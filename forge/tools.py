"""Tool Registry and core tools for ForgeAI.

Provides a generic ToolRegistry and four built-in tools:
1. csv_lookup: Search CSV records by column/value within a bounded folder.
2. calculator: Safely evaluate basic arithmetic expressions via AST.
3. weather_api: Query keyless Open-Meteo API with timeout and error handling.
4. current_date: Return the current local date in ISO format.
"""

import ast
from dataclasses import dataclass, field
import datetime
import operator as op
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import csv
import requests


# ---------------------------------------------------------------------------
# 1. CSV Lookup Tool
# ---------------------------------------------------------------------------

def resolve_secure_csv_path(base_folder: Union[str, Path], relative_path: str) -> Path:
    """Resolve a CSV path ensuring it strictly stays within the base document folder."""
    base = Path(base_folder).resolve()
    target = (base / relative_path).resolve()

    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(
            f"Path traversal denied: '{relative_path}' resolves outside allowed directory."
        )

    if not target.exists():
        raise FileNotFoundError(f"CSV file not found: '{relative_path}'")
    if not target.is_file():
        raise ValueError(f"Path is not a regular file: '{relative_path}'")
    if target.suffix.lower() != ".csv":
        raise ValueError(f"Target file must be a CSV file (.csv), got: '{target.name}'")

    return target


def csv_lookup(
    filename: str,
    column: str,
    value: str,
    document_folder: Union[str, Path],
    exact_match: bool = False,
) -> Dict[str, Any]:
    """Search CSV records by column and value within the bounded document folder."""
    csv_path = resolve_secure_csv_path(document_folder, filename)

    content: Optional[str] = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            content = csv_path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise ValueError(f"Unable to decode CSV file: '{filename}'")

    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"CSV file is empty: '{filename}'")

    reader = csv.DictReader(lines)
    if reader.fieldnames is None:
        raise ValueError(f"CSV missing column headers: '{filename}'")

    # Find matching column name (case-insensitive if exact match not found)
    target_col = None
    for col in reader.fieldnames:
        if col == column or col.strip().lower() == column.strip().lower():
            target_col = col
            break

    if target_col is None:
        raise ValueError(
            f"Column '{column}' not found in CSV '{filename}'. "
            f"Available columns: {list(reader.fieldnames)}"
        )

    clean_query = value.strip().lower()
    matches: List[Dict[str, Any]] = []

    for row_idx, row in enumerate(reader, start=1):
        cell_val = row.get(target_col, "")
        cell_str = str(cell_val).strip() if cell_val is not None else ""

        is_match = (
            cell_str.lower() == clean_query
            if exact_match
            else clean_query in cell_str.lower()
        )

        if is_match:
            matches.append({
                "row_number": row_idx,
                "source": filename,
                "record": dict(row),
            })

    return {
        "filename": filename,
        "searched_column": target_col,
        "searched_value": value,
        "total_matches": len(matches),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# 2. Safe Calculator Tool (AST-based, never uses eval/exec)
# ---------------------------------------------------------------------------

AST_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_ast_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluate an AST expression node with strict whitelist."""
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    if isinstance(node, ast.UnaryOp):
        operator_type = type(node.op)
        if operator_type not in AST_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {operator_type.__name__}")
        operand = _eval_ast_node(node.operand)
        return AST_OPERATORS[operator_type](operand)

    if isinstance(node, ast.BinOp):
        operator_type = type(node.op)
        if operator_type not in AST_OPERATORS:
            raise ValueError(f"Unsupported binary operator: {operator_type.__name__}")

        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)

        if operator_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ZeroDivisionError("Division or modulo by zero is not permitted.")

        if operator_type is ast.Pow:
            if right > 1000 or (abs(left) > 1000 and right > 10):
                raise ValueError("Exponentiation operands exceed safe limits.")

        return AST_OPERATORS[operator_type](left, right)

    raise ValueError(f"Unsafe or unsupported expression syntax: {type(node).__name__}")


def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluate basic arithmetic expressions using AST without eval or exec."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string.")

    expr = expression.strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression syntax: {exc}") from exc

    result = _eval_ast_node(tree)
    # Normalize float to int if exact integer
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return {
        "expression": expr,
        "result": result,
    }


# ---------------------------------------------------------------------------
# 3. Weather API Tool (Open-Meteo keyless API)
# ---------------------------------------------------------------------------

def weather_api(
    latitude: float,
    longitude: float,
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """Query current weather data from Open-Meteo keyless forecast API."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Coordinates must be valid numbers: {exc}") from exc

    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude must be between -90.0 and 90.0, got: {lat}")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude must be between -180.0 and 180.0, got: {lon}")

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
    }

    url = "https://api.open-meteo.com/v1/forecast"
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(f"Weather API request timed out after {timeout} seconds: {exc}") from exc
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Failed to connect to Weather API service at {url}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Weather API request failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Weather API returned HTTP {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except Exception as exc:
        raise ValueError(f"Malformed JSON response from Weather API: {exc}") from exc

    current = data.get("current", {})
    units = data.get("current_units", {})

    return {
        "latitude": data.get("latitude", lat),
        "longitude": data.get("longitude", lon),
        "timezone": data.get("timezone", "auto"),
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units.get("temperature_2m", "°C"),
        "relative_humidity": current.get("relative_humidity_2m"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
        "weather_code": current.get("weather_code"),
        "time": current.get("time"),
    }


# ---------------------------------------------------------------------------
# 4. Current Date Tool
# ---------------------------------------------------------------------------

def current_date() -> Dict[str, str]:
    """Return the current local date in ISO format."""
    now = datetime.datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "iso_timestamp": now.isoformat(),
        "day_of_week": now.strftime("%A"),
    }


# ---------------------------------------------------------------------------
# 5. Tool Definition and Tool Registry
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    """Metadata and execution callable for a registered tool."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]

    def to_schema(self) -> Dict[str, Any]:
        """Return JSON-serializable tool schema for LLM tool selection."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Generic tool registry with argument validation and authorization control."""

    def __init__(self, document_folder: Optional[Union[str, Path]] = None) -> None:
        self.document_folder = Path(document_folder).resolve() if document_folder else None
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the four standard tools."""
        self.register(
            name="csv_lookup",
            description="Search CSV records by column name and value within the document repository.",
            parameters={
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative path to CSV file within document folder (e.g. 'equipment/pumps.csv').",
                    },
                    "column": {
                        "type": "string",
                        "description": "Column name to search.",
                    },
                    "value": {
                        "type": "string",
                        "description": "Target value to match in the specified column.",
                    },
                    "exact_match": {
                        "type": "boolean",
                        "description": "Whether to require an exact match instead of substring matching. Defaults to false.",
                    },
                },
                "required": ["filename", "column", "value"],
            },
            func=self._wrap_csv_lookup,
        )

        self.register(
            name="calculator",
            description="Safely evaluate basic arithmetic expressions (+, -, *, /, //, %, **). Never uses eval.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to compute (e.g. '450 / 60 * 1.25').",
                    },
                },
                "required": ["expression"],
            },
            func=calculator,
        )

        self.register(
            name="weather_api",
            description="Fetch real-time weather parameters using coordinates (latitude, longitude) from Open-Meteo.",
            parameters={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "Latitude coordinate (-90.0 to 90.0).",
                    },
                    "longitude": {
                        "type": "number",
                        "description": "Longitude coordinate (-180.0 to 180.0).",
                    },
                },
                "required": ["latitude", "longitude"],
            },
            func=weather_api,
        )

        self.register(
            name="current_date",
            description="Get the current local date in ISO format (YYYY-MM-DD).",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            func=current_date,
        )

    def _wrap_csv_lookup(
        self,
        filename: str,
        column: str,
        value: str,
        document_folder: Optional[Union[str, Path]] = None,
        exact_match: bool = False,
    ) -> Dict[str, Any]:
        """Wrapper injecting default document_folder if omitted."""
        folder = document_folder or self.document_folder
        if not folder:
            raise ValueError(
                "Document folder is not configured. Supply 'document_folder' argument or set on ToolRegistry."
            )
        return csv_lookup(
            filename=filename,
            column=column,
            value=value,
            document_folder=folder,
            exact_match=exact_match,
        )

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable[..., Any],
    ) -> None:
        """Register a tool definition."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if not callable(func):
            raise ValueError(f"Tool function for '{name}' must be callable.")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description or "",
            parameters=parameters or {"type": "object", "properties": {}},
            function=func,
        )

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve tool definition by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """Return list of all registered tool names."""
        return sorted(self._tools.keys())

    def get_tool_schemas(
        self,
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return JSON-serializable tool schemas for LLM tool selection."""
        allowed_set = set(allowed_tools) if allowed_tools is not None else None
        schemas: List[Dict[str, Any]] = []

        for name in sorted(self._tools.keys()):
            if allowed_set is not None and name not in allowed_set:
                continue
            schemas.append(self._tools[name].to_schema())

        return schemas

    def _validate_arguments(
        self,
        tool_def: ToolDefinition,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate argument presence against required parameter list."""
        if not isinstance(arguments, dict):
            raise ValueError(f"Arguments must be a dictionary, got {type(arguments).__name__}")

        schema = tool_def.parameters
        required = schema.get("required", [])
        missing = [param for param in required if param not in arguments]
        if missing:
            raise ValueError(f"Missing required parameter(s) for tool '{tool_def.name}': {missing}")

        return arguments

    def execute(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool by name with validated arguments and authorization check.

        Returns structured result:
            {"success": bool, "tool": str, "result": Any, "error": Optional[str]}
        """
        args = arguments if arguments is not None else {}

        # 1. Authorization check
        if allowed_tools is not None and name not in allowed_tools:
            return {
                "success": False,
                "tool": name,
                "result": None,
                "error": f"Unauthorized tool execution: Tool '{name}' is not authorized. Allowed tools: {list(allowed_tools)}",
            }

        # 2. Registration check
        if name not in self._tools:
            return {
                "success": False,
                "tool": name,
                "result": None,
                "error": f"Tool '{name}' is not registered in ToolRegistry. Available: {self.list_tools()}",
            }

        tool_def = self._tools[name]

        # 3. Parameter validation
        try:
            validated_args = self._validate_arguments(tool_def, args)
        except Exception as exc:
            return {
                "success": False,
                "tool": name,
                "result": None,
                "error": f"Invalid arguments for '{name}': {exc}",
            }

        # 4. Tool execution
        try:
            res = tool_def.function(**validated_args)
            return {
                "success": True,
                "tool": name,
                "result": res,
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "tool": name,
                "result": None,
                "error": f"Execution error in tool '{name}': {exc}",
            }
