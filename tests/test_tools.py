"""Unit tests for ForgeAI Tool Registry and Core Tools (forge/tools.py)."""

import datetime
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch
import requests

from forge.tools import (
    ToolRegistry,
    calculator,
    csv_lookup,
    current_date,
    resolve_secure_csv_path,
    weather_api,
)


class TestCalculatorTool(unittest.TestCase):
    """Tests for safe AST-based calculator tool."""

    def test_valid_arithmetic_expressions(self) -> None:
        """Test evaluation of standard arithmetic operations."""
        cases = [
            ("2 + 2", 4),
            ("10 - 4.5", 5.5),
            ("3 * 7", 21),
            ("100 / 4", 25),
            ("17 // 5", 3),
            ("17 % 5", 2),
            ("2 ** 8", 256),
            ("(5 + 3) * (10 - 2)", 64),
            ("-15 + 5", -10),
            ("+42", 42),
            ("2 * (3 + 4 * (2 - 1))", 14),
        ]
        for expr, expected in cases:
            res = calculator(expr)
            self.assertEqual(res["result"], expected, f"Failed for expr: {expr}")
            self.assertEqual(res["expression"], expr)

    def test_division_by_zero(self) -> None:
        """Test division and modulo by zero are safely rejected."""
        with self.assertRaises(ZeroDivisionError):
            calculator("10 / 0")
        with self.assertRaises(ZeroDivisionError):
            calculator("10 // 0")
        with self.assertRaises(ZeroDivisionError):
            calculator("10 % 0")

    def test_unsafe_and_arbitrary_code_rejected(self) -> None:
        """Test that non-arithmetic syntax, imports, function calls, and assignments are rejected."""
        unsafe_expressions = [
            "__import__('os').system('echo pwned')",
            "open('passwords.txt', 'r')",
            "eval('2 + 2')",
            "exec('x = 1')",
            "x = 5",
            "[x for x in range(10)]",
            "{'a': 1}",
            "import os",
            "lambda x: x + 1",
            "print('hello')",
        ]
        for expr in unsafe_expressions:
            with self.assertRaises(ValueError, msg=f"Expression should be rejected: {expr}"):
                calculator(expr)

    def test_invalid_syntax_and_empty_expression(self) -> None:
        """Test malformed syntax and empty inputs."""
        with self.assertRaises(ValueError):
            calculator("")
        with self.assertRaises(ValueError):
            calculator("   ")
        with self.assertRaises(ValueError):
            calculator("2 + * 3")

    def test_oversized_exponent_rejected(self) -> None:
        """Test exponentiation limits to prevent DoS."""
        with self.assertRaises(ValueError):
            calculator("2 ** 10001")


class TestCSVLookupTool(unittest.TestCase):
    """Tests for CSV lookup tool and directory boundary enforcement."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.doc_folder = Path(self.temp_dir.name)

        # Create a sample CSV
        self.csv_file = self.doc_folder / "equipment.csv"
        self.csv_file.write_text(
            "equipment_id,name,location,operating_temp\n"
            "P-101,Feed Pump,Building A,68.5\n"
            "P-102,Slurry Pump,Building B,85.0\n"
            "V-201,Relief Valve,Building A,45.2\n",
            encoding="utf-8",
        )

        # Create a subfolder with duplicate named CSV
        sub = self.doc_folder / "subfolder"
        sub.mkdir()
        (sub / "equipment.csv").write_text(
            "equipment_id,name,location,operating_temp\n"
            "P-303,High-Pressure Pump,Building C,92.0\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_csv_lookup_matching_records(self) -> None:
        """Test searching records by column and value."""
        res = csv_lookup(
            filename="equipment.csv",
            column="location",
            value="Building A",
            document_folder=self.doc_folder,
        )
        self.assertEqual(res["total_matches"], 2)
        matches = res["matches"]
        self.assertEqual(matches[0]["record"]["equipment_id"], "P-101")
        self.assertEqual(matches[0]["row_number"], 1)
        self.assertEqual(matches[1]["record"]["equipment_id"], "V-201")
        self.assertEqual(matches[1]["row_number"], 3)

    def test_csv_lookup_in_subfolder(self) -> None:
        """Test searching CSV in a nested subfolder."""
        res = csv_lookup(
            filename="subfolder/equipment.csv",
            column="equipment_id",
            value="P-303",
            document_folder=self.doc_folder,
        )
        self.assertEqual(res["total_matches"], 1)
        self.assertEqual(res["matches"][0]["record"]["name"], "High-Pressure Pump")

    def test_csv_lookup_path_traversal_prevention(self) -> None:
        """Verify path traversal outside document folder raises PermissionError."""
        outside_attempts = [
            "../secret.csv",
            "../../etc/passwd",
            "..\\..\\windows\\system32\\cmd.exe",
        ]
        for attempt in outside_attempts:
            with self.assertRaises(PermissionError):
                csv_lookup(
                    filename=attempt,
                    column="name",
                    value="test",
                    document_folder=self.doc_folder,
                )

    def test_csv_lookup_missing_column_or_file(self) -> None:
        """Test error handling for nonexistent files or columns."""
        with self.assertRaises(FileNotFoundError):
            csv_lookup(
                filename="nonexistent.csv",
                column="name",
                value="test",
                document_folder=self.doc_folder,
            )

        with self.assertRaises(ValueError):
            csv_lookup(
                filename="equipment.csv",
                column="nonexistent_column",
                value="test",
                document_folder=self.doc_folder,
            )


class TestWeatherAPITool(unittest.TestCase):
    """Tests for weather_api tool with mocked HTTP responses."""

    @patch("forge.tools.requests.get")
    def test_weather_api_successful_response(self, mock_get: MagicMock) -> None:
        """Test weather API parsing using a mocked successful response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "latitude": 52.52,
            "longitude": 13.41,
            "timezone": "Europe/Berlin",
            "current": {
                "temperature_2m": 18.5,
                "relative_humidity_2m": 65,
                "wind_speed_10m": 12.3,
                "weather_code": 1,
                "time": "2026-11-27T12:00",
            },
            "current_units": {
                "temperature_2m": "°C",
                "wind_speed_10m": "km/h",
            },
        }
        mock_get.return_value = mock_response

        res = weather_api(latitude=52.52, longitude=13.41, timeout=5.0)

        mock_get.assert_called_once()
        self.assertEqual(res["latitude"], 52.52)
        self.assertEqual(res["longitude"], 13.41)
        self.assertEqual(res["temperature"], 18.5)
        self.assertEqual(res["temperature_unit"], "°C")
        self.assertEqual(res["wind_speed"], 12.3)

    def test_weather_api_invalid_coordinates(self) -> None:
        """Test validation of out-of-range coordinates."""
        with self.assertRaises(ValueError):
            weather_api(latitude=95.0, longitude=10.0)
        with self.assertRaises(ValueError):
            weather_api(latitude=45.0, longitude=200.0)
        with self.assertRaises(ValueError):
            weather_api(latitude="invalid", longitude=10.0)  # type: ignore

    @patch("forge.tools.requests.get")
    def test_weather_api_timeout_and_network_errors(self, mock_get: MagicMock) -> None:
        """Test explicit error handling on network timeouts and connection errors."""
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        with self.assertRaises(TimeoutError):
            weather_api(latitude=52.52, longitude=13.41, timeout=1.0)

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        with self.assertRaises(ConnectionError):
            weather_api(latitude=52.52, longitude=13.41, timeout=1.0)


class TestCurrentDateTool(unittest.TestCase):
    """Tests for current_date tool."""

    def test_current_date_iso_format(self) -> None:
        """Test date format matches YYYY-MM-DD."""
        res = current_date()
        self.assertIn("date", res)
        self.assertIn("iso_timestamp", res)
        self.assertIn("day_of_week", res)

        # Validate format parses as ISO date
        parsed = datetime.date.fromisoformat(res["date"])
        self.assertEqual(parsed.strftime("%Y-%m-%d"), res["date"])


class TestToolRegistry(unittest.TestCase):
    """Tests for ToolRegistry management, schema exposure, and authorization."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.doc_folder = Path(self.temp_dir.name)
        (self.doc_folder / "test.csv").write_text(
            "sensor,reading\ns1,42\n", encoding="utf-8"
        )
        self.registry = ToolRegistry(document_folder=self.doc_folder)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_registered_tools_list_and_schemas(self) -> None:
        """Test default tool registration and JSON-serializable schemas."""
        tools = self.registry.list_tools()
        expected_tools = ["calculator", "csv_lookup", "current_date", "weather_api"]
        self.assertEqual(sorted(tools), expected_tools)

        schemas = self.registry.get_tool_schemas()
        self.assertEqual(len(schemas), 4)
        names = [s["name"] for s in schemas]
        self.assertEqual(sorted(names), expected_tools)

        # Verify schema structure
        for s in schemas:
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertIn("parameters", s)
            self.assertIsInstance(s["parameters"], dict)

    def test_execution_calculator_success(self) -> None:
        """Test executing calculator via registry."""
        res = self.registry.execute("calculator", {"expression": "25 * 4"})
        self.assertTrue(res["success"])
        self.assertEqual(res["result"]["result"], 100)
        self.assertIsNone(res["error"])

    def test_execution_csv_lookup_success(self) -> None:
        """Test executing csv_lookup with automatic document_folder injection."""
        res = self.registry.execute(
            "csv_lookup",
            {"filename": "test.csv", "column": "sensor", "value": "s1"},
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["result"]["total_matches"], 1)

    def test_execution_missing_arguments(self) -> None:
        """Test execution fails when required parameters are omitted."""
        # calculator requires 'expression'
        res = self.registry.execute("calculator", {})
        self.assertFalse(res["success"])
        self.assertIn("Missing required parameter", res["error"])

    def test_unauthorized_tool_execution(self) -> None:
        """Test that tools not in allowed_tools list are rejected."""
        allowed = ["calculator", "current_date"]

        # calculator is authorized
        res_calc = self.registry.execute(
            "calculator",
            {"expression": "1 + 1"},
            allowed_tools=allowed,
        )
        self.assertTrue(res_calc["success"])

        # csv_lookup is not authorized for this agent
        res_csv = self.registry.execute(
            "csv_lookup",
            {"filename": "test.csv", "column": "sensor", "value": "s1"},
            allowed_tools=allowed,
        )
        self.assertFalse(res_csv["success"])
        self.assertIn("not authorized", res_csv["error"])

    def test_unregistered_tool_execution(self) -> None:
        """Test executing nonexistent tool returns explicit error."""
        res = self.registry.execute("nonexistent_tool", {})
        self.assertFalse(res["success"])
        self.assertIn("not registered", res["error"])

    def test_allowed_tools_filters_schemas(self) -> None:
        """Test get_tool_schemas respects allowed_tools filter."""
        filtered = self.registry.get_tool_schemas(allowed_tools=["calculator"])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["name"], "calculator")


if __name__ == "__main__":
    unittest.main()
