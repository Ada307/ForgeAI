"""Unit and live tests for Agent Execution Runner (forge/runner.py)."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from forge.generator import OUTPUT_CONTRACT_SCHEMA
from forge.llm import OllamaClient
from forge.runner import AgentRunner


class MockOllamaClient:
    """Mock client returning pre-configured responses for runner unit tests."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return {"status": "answered", "answer": "Mock answer", "citations": []}


class TestAgentRunnerUnit(unittest.TestCase):
    """Mocked unit tests for citation validation, unauthorized tools, and response handling."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.doc_folder = Path(self.temp_dir.name)

        # Create a sample document
        (self.doc_folder / "pump_spec.txt").write_text(
            "Slurry Pump P-302 standard operating temperature limit is 85 degrees Celsius.\n"
            "Maintenance interval is 250 operating hours.",
            encoding="utf-8",
        )

        self.agent_def = {
            "agent_name": "TestPumpAgent",
            "description": "Technical assistant for Slurry Pump P-302.",
            "system_prompt": "You are TestPumpAgent. Answer using provided evidence.",
            "scope": "Guidance on Slurry Pump P-302.",
            "audience": "Maintenance technicians",
            "tone": "Direct",
            "must_answer_topics": ["operating temperature limit"],
            "must_refuse_topics": ["financial revenue [Refusal: Unsupported]"],
            "retrieval_config": {
                "chunk_size": 180,
                "chunk_overlap": 30,
                "top_k": 3,
                "minimum_similarity_score": 0.0,
            },
            "tools": [],
            "output_contract": OUTPUT_CONTRACT_SCHEMA,
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_citation_validation_catches_fabricated_citation(self) -> None:
        """Verify fabricated citations are rejected and recorded in execution_errors."""
        mock_response = {
            "status": "answered",
            "answer": "The pump operates up to 85 C.",
            "citations": [
                {
                    "source": "pump_spec.txt",
                    "chunk_id": "pump_spec.txt:c0",
                    "excerpt": "THIS EXCERPT DOES NOT EXIST IN DOCUMENT",
                }
            ],
        }

        client = MockOllamaClient([mock_response])
        runner = AgentRunner(
            agent_definition=self.agent_def,
            document_folder=self.doc_folder,
            client=client,
        )

        result = runner.answer("operating temperature limit")

        self.assertEqual(result.status, "answered")
        # Fabricated citation should be excluded
        self.assertEqual(result.citations, [])
        # Error should be recorded
        self.assertTrue(any("Fabricated or invalid citation" in err for err in result.execution_errors))

    def test_valid_citation_is_accepted(self) -> None:
        """Verify genuine citation matching source, chunk_id, and excerpt is accepted."""
        mock_response = {
            "status": "answered",
            "answer": "Operating temperature limit is 85 degrees Celsius.",
            "citations": [
                {
                    "source": "pump_spec.txt",
                    "chunk_id": "pump_spec.txt:c0",
                    "excerpt": "standard operating temperature limit is 85 degrees Celsius",
                }
            ],
        }

        client = MockOllamaClient([mock_response])
        runner = AgentRunner(
            agent_definition=self.agent_def,
            document_folder=self.doc_folder,
            client=client,
        )

        result = runner.answer("operating temperature limit")

        self.assertEqual(result.status, "answered")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["source"], "pump_spec.txt")
        self.assertEqual(result.citations[0]["chunk_id"], "pump_spec.txt:c0")

    def test_unauthorized_tool_execution_blocked(self) -> None:
        """Verify that an agent with no authorized tools rejects tool calls."""
        # Definition has tools: []
        # Simulate LLM trying to call calculator
        mock_tool_decision = {
            "action": "call_tool",
            "tool_name": "calculator",
            "tool_args": {"expression": "2 + 2"},
        }
        mock_final_answer = {
            "status": "answered",
            "answer": "Computed result",
            "citations": [],
        }

        # Runner with empty allowed_tools does not even enter tool execution loop
        client = MockOllamaClient([mock_final_answer])
        runner = AgentRunner(
            agent_definition=self.agent_def,
            document_folder=self.doc_folder,
            client=client,
        )

        self.assertEqual(runner.allowed_tools, [])
        result = runner.answer("calculate 2 + 2")
        self.assertEqual(result.tool_executions, [])

    def test_refusal_for_unsupported_question(self) -> None:
        """Verify refusal response format."""
        mock_response = {
            "status": "refused",
            "answer": "I cannot answer queries regarding company financial revenue as it is outside my verified scope.",
            "citations": [],
        }

        client = MockOllamaClient([mock_response])
        runner = AgentRunner(
            agent_definition=self.agent_def,
            document_folder=self.doc_folder,
            client=client,
        )

        result = runner.answer("What is the quarterly revenue?")
        self.assertEqual(result.status, "refused")
        self.assertIn("financial revenue", result.answer)
        self.assertEqual(result.citations, [])

    def test_reusable_answer_method_output_structure(self) -> None:
        """Verify AgentExecutionResult contains all required fields for future evaluation."""
        client = MockOllamaClient([{"status": "answered", "answer": "Test answer", "citations": []}])
        runner = AgentRunner(
            agent_definition=self.agent_def,
            document_folder=self.doc_folder,
            client=client,
        )

        res = runner.answer("test query")
        data = res.to_dict()

        for key in ("status", "answer", "citations", "retrieved_evidence", "tool_executions", "execution_errors"):
            self.assertIn(key, data)
        self.assertEqual(data["status"], "answered")


class TestLiveAgentRunner(unittest.TestCase):
    """Live smoke test executing answer() against local Ollama instance with qwen2.5:3b."""

    def test_live_runner_smoke(self) -> None:
        client = OllamaClient(timeout=45.0)

        # Check connectivity
        try:
            client.generate("Ping", max_tokens=2)
        except Exception as e:
            self.skipTest(f"Ollama server not reachable: {e}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            docs_dir = Path(tmp_dir) / "docs"
            docs_dir.mkdir()

            (docs_dir / "pump_spec.txt").write_text(
                "Slurry Pump P-302 Operational Specifications:\n"
                "Standard operating temperature limit: 85 degrees Celsius.\n"
                "Maximum flow rate capacity: 450 GPM.\n"
                "Routine maintenance inspection interval: 250 operating hours.\n",
                encoding="utf-8",
            )

            agent_def = {
                "agent_name": "PumpTechAssistant",
                "description": "Technical guidance for Slurry Pump P-302.",
                "system_prompt": (
                    "You are PumpTechAssistant. Answer queries regarding Slurry Pump P-302 strictly using retrieved evidence. "
                    "Refuse queries regarding company finances or unverified topics. "
                    "Output JSON adhering to the output contract: {\"status\": \"answered\" | \"refused\", \"answer\": \"...\", \"citations\": [...]}"
                ),
                "scope": "Technical guidance on Slurry Pump P-302 operating temperature limits and maintenance inspection intervals.",
                "audience": "Maintenance technicians",
                "tone": "Professional and concise",
                "must_answer_topics": [
                    "operating temperature limit of Slurry Pump P-302",
                    "standard maintenance inspection interval hours",
                ],
                "must_refuse_topics": [
                    "company financial revenue and profits [Refusal: Unsupported]",
                ],
                "retrieval_config": {
                    "chunk_size": 180,
                    "chunk_overlap": 30,
                    "top_k": 3,
                    "minimum_similarity_score": 0.0,
                },
                "tools": [],
                "output_contract": OUTPUT_CONTRACT_SCHEMA,
            }

            runner = AgentRunner(
                agent_definition=agent_def,
                document_folder=docs_dir,
                client=client,
            )

            # 1. Answerable question
            res_answerable = runner.answer("What is the operating temperature limit of Slurry Pump P-302?")
            self.assertEqual(res_answerable.status, "answered")
            self.assertIn("85", res_answerable.answer)
            self.assertGreaterEqual(len(res_answerable.citations), 1)

            # 2. Unsupported / refusal question
            res_refused = runner.answer("What is the quarterly revenue and profit of the company?")
            self.assertEqual(res_refused.status, "refused")
            self.assertEqual(res_refused.citations, [])


if __name__ == "__main__":
    unittest.main()
