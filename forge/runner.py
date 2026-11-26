"""Generic Agent Execution Runner for ForgeAI.

Executes a generated agent definition against a document folder and ToolRegistry,
enforcing grounding, citation validation, tool authorization, refusal policies,
and the machine-readable output contract.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from forge.documents import Document, DocumentLoader
from forge.generator import validate_agent_definition
from forge.groundability import validate_citation
from forge.llm import OllamaClient
from forge.retrieval import RetrievalResult, RetrieverConfig, TFIDFRetriever
from forge.tools import ToolRegistry


RUNNER_PROMPT_TEMPLATE = """{system_prompt}

### RETRIEVED DOCUMENT EVIDENCE:
{retrieved_evidence}

### TOOL RESULTS (IF ANY):
{tool_results}

### USER QUERY:
{query}

### MANDATORY INSTRUCTIONS:
1. Adhere strictly to the required output contract JSON format:
   {{
     "status": "answered" | "refused" | "clarification_needed",
     "answer": "Clear factual answer or refusal explanation",
     "citations": [
       {{
         "source": "exact_source_path",
         "chunk_id": "exact_chunk_id",
         "excerpt": "verbatim text excerpt from evidence"
       }}
     ]
   }}
2. If the user query is about a forbidden topic or lacks sufficient factual evidence, set status to "refused".
3. Every factual claim must be backed by a verified citation with exact source, chunk_id, and excerpt.
4. If tool execution is needed, format answer with requested tool or return the final answer.
"""


TOOL_DECISION_PROMPT = """{system_prompt}

You have access to the following authorized tools:
{tool_schemas}

Retrieved Document Evidence:
{retrieved_evidence}

User Query:
{query}

Determine if an authorized tool execution is strictly necessary to answer the query.
If a tool is needed, output JSON:
{{
  "action": "call_tool",
  "tool_name": "name_of_tool",
  "tool_args": {{ "param": "value" }}
}}
If no tool is needed, output JSON:
{{
  "action": "proceed_to_answer"
}}
"""


@dataclass
class AgentExecutionResult:
    """Standardized output structure for agent execution."""
    status: str  # "answered", "refused", "clarification_needed"
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    retrieved_evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_executions: List[Dict[str, Any]] = field(default_factory=list)
    execution_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_evidence": self.retrieved_evidence,
            "tool_executions": self.tool_executions,
            "execution_errors": self.execution_errors,
        }


class AgentRunner:
    """Loads and executes an agent definition against documents and tools."""

    def __init__(
        self,
        agent_definition: Union[Dict[str, Any], str, Path],
        document_folder: Union[str, Path],
        client: Optional[OllamaClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.document_folder = Path(document_folder).resolve()

        # 1. Load and validate agent definition
        if isinstance(agent_definition, (str, Path)):
            def_path = Path(agent_definition).resolve()
            if not def_path.exists():
                raise FileNotFoundError(f"Agent definition file not found: {def_path}")
            self.definition = json.loads(def_path.read_text(encoding="utf-8"))
        else:
            self.definition = dict(agent_definition)

        self.tool_registry = tool_registry or ToolRegistry(document_folder=self.document_folder)
        validate_agent_definition(self.definition, self.tool_registry)

        # 2. Extract allowed tools
        self.allowed_tools = [
            t["name"] for t in self.definition.get("tools", [])
            if isinstance(t, dict) and "name" in t
        ]

        # 3. Load documents and build retrieval index
        loader = DocumentLoader()
        self.documents: List[Document] = loader.load(self.document_folder)

        retrieval_cfg_dict = self.definition.get("retrieval_config", {})
        self.retrieval_config = RetrieverConfig.from_dict(retrieval_cfg_dict)
        self.retrieval_config.validate()

        self.retriever = TFIDFRetriever(
            config=self.retrieval_config,
            documents=self.documents,
        )

    def _format_evidence_chunks(self, chunks: Sequence[RetrievalResult]) -> str:
        """Format retrieved chunks for prompt context within bounds."""
        if not chunks:
            return "No relevant document evidence retrieved."

        lines: List[str] = []
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            lines.append(f"[Source: {src} | Chunk ID: {c.chunk_id}]\n{c.text}")
        return "\n\n".join(lines)

    def answer(self, query: str) -> AgentExecutionResult:
        """Execute the agent to answer a user query with full tracing."""
        errors: List[str] = []
        tool_executions: List[Dict[str, Any]] = []

        # 1. Retrieve evidence
        retrieved_results = self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_config.top_k,
            minimum_similarity_score=self.retrieval_config.minimum_similarity_score,
        )
        evidence_dicts = [
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "metadata": r.metadata,
                "score": r.score,
            }
            for r in retrieved_results
        ]

        # 2. Tool execution loop (at most 2 rounds)
        tool_results_history: List[str] = []
        if self.allowed_tools:
            for round_idx in range(2):
                tool_schemas = self.tool_registry.get_tool_schemas(allowed_tools=self.allowed_tools)
                decision_prompt = (
                    TOOL_DECISION_PROMPT.replace("{system_prompt}", self.definition.get("system_prompt", ""))
                    .replace("{tool_schemas}", json.dumps(tool_schemas, indent=2))
                    .replace("{retrieved_evidence}", self._format_evidence_chunks(retrieved_results))
                    .replace("{query}", query)
                )

                try:
                    decision = self.client.generate_json(
                        prompt=decision_prompt,
                        system_prompt="Determine if a tool call is strictly necessary. Return JSON.",
                    )
                except Exception as exc:
                    errors.append(f"Tool decision error: {exc}")
                    break

                if decision.get("action") == "call_tool":
                    t_name = str(decision.get("tool_name", "")).strip()
                    t_args = decision.get("tool_args", {})
                    if not isinstance(t_args, dict):
                        t_args = {}

                    tool_run = self.tool_registry.execute(
                        name=t_name,
                        arguments=t_args,
                        allowed_tools=self.allowed_tools,
                    )
                    tool_executions.append({
                        "round": round_idx + 1,
                        "tool": t_name,
                        "arguments": t_args,
                        "outcome": tool_run,
                    })

                    if tool_run.get("success"):
                        tool_results_history.append(
                            f"Tool '{t_name}' result: {json.dumps(tool_run.get('result'))}"
                        )
                    else:
                        err_msg = tool_run.get("error", "Unknown tool error")
                        errors.append(f"Tool '{t_name}' error: {err_msg}")
                        tool_results_history.append(f"Tool '{t_name}' failed: {err_msg}")
                else:
                    # Model decided no tool is needed
                    break

        # 3. Generate answer using system prompt & output contract
        tool_results_str = "\n".join(tool_results_history) if tool_results_history else "None."
        answer_prompt = (
            RUNNER_PROMPT_TEMPLATE.replace("{system_prompt}", self.definition.get("system_prompt", ""))
            .replace("{retrieved_evidence}", self._format_evidence_chunks(retrieved_results))
            .replace("{tool_results}", tool_results_str)
            .replace("{query}", query)
        )

        try:
            raw_response = self.client.generate_json(
                prompt=answer_prompt,
                system_prompt="Return output adhering strictly to the JSON output contract.",
            )
        except Exception as exc:
            return AgentExecutionResult(
                status="refused",
                answer="Failed to generate structured response from model.",
                citations=[],
                retrieved_evidence=evidence_dicts,
                tool_executions=tool_executions,
                execution_errors=errors + [f"Generation error: {exc}"],
            )

        status = str(raw_response.get("status", "answered")).strip().lower()
        if status not in {"answered", "refused", "clarification_needed"}:
            status = "refused"

        answer_text = str(raw_response.get("answer", "")).strip()
        raw_citations = raw_response.get("citations", [])

        # 4. Validate citations against actual retrieved source, chunk_id, and excerpt
        validated_citations: List[Dict[str, str]] = []
        if isinstance(raw_citations, list) and status == "answered":
            for cit in raw_citations:
                if not isinstance(cit, dict):
                    continue

                c_dict = {
                    "source": str(cit.get("source", "")).strip(),
                    "chunk_id": str(cit.get("chunk_id", "")).strip(),
                    "excerpt": str(cit.get("excerpt", "")).strip(),
                }

                if validate_citation(c_dict, retrieved_results):
                    validated_citations.append(c_dict)
                else:
                    errors.append(
                        f"Fabricated or invalid citation rejected: source='{c_dict['source']}', "
                        f"chunk_id='{c_dict['chunk_id']}', excerpt='{c_dict['excerpt'][:40]}...'"
                    )

            # If answer status is answered but citations were all fabricated or missing when needed
            if not validated_citations and not tool_executions and retrieved_results:
                errors.append("All proposed citations failed verification against actual chunks.")

        return AgentExecutionResult(
            status=status,
            answer=answer_text,
            citations=validated_citations,
            retrieved_evidence=evidence_dicts,
            tool_executions=tool_executions,
            execution_errors=errors,
        )
