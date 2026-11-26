"""Unit and live verification tests for Agent Generator (forge/generator.py)."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from forge.documents import Document
from forge.generator import (
    OUTPUT_CONTRACT_SCHEMA,
    AgentGenerator,
    save_agent_definition,
    validate_agent_definition,
)
from forge.llm import OllamaClient
from forge.tools import ToolRegistry


class MockOllamaClient:
    """Mock Ollama client for deterministic generator unit tests."""

    def __init__(self, response: Dict[str, Any]) -> None:
        self.response = response
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return dict(self.response)


class TestAgentGeneratorUnit(unittest.TestCase):
    """Unit tests for agent definition validation, tool rejection, and safety guardrails."""

    def setUp(self) -> None:
        self.registry = ToolRegistry()
        self.valid_def: Dict[str, Any] = {
            "agent_name": "FurnaceMonitorAgent",
            "description": "Specialized blast furnace monitoring assistant.",
            "system_prompt": "You are FurnaceMonitorAgent. Ground claims in documents. Refuse unauthorized topics.",
            "scope": "Blast furnace operating parameter diagnostics and calculation.",
            "audience": "Furnace operators",
            "tone": "Technical and concise",
            "must_answer_topics": ["hearth temperature", "calculate thermal gradient"],
            "must_refuse_topics": ["corporate revenue [Refusal: Unsupported]"],
            "retrieval_config": {
                "chunk_size": 180,
                "chunk_overlap": 30,
                "top_k": 4,
                "minimum_similarity_score": 0.0,
            },
            "tools": [{"name": "calculator", "justification": "Calculate thermal gradients"}],
            "output_contract": OUTPUT_CONTRACT_SCHEMA,
        }

    def test_valid_definition_passes(self) -> None:
        """Verify that a well-formed agent definition passes validation."""
        validated = validate_agent_definition(self.valid_def, self.registry)
        self.assertEqual(validated["agent_name"], "FurnaceMonitorAgent")
        self.assertEqual(len(validated["tools"]), 1)
        self.assertEqual(validated["tools"][0]["name"], "calculator")

    def test_missing_required_fields_raises_error(self) -> None:
        """Verify missing required string or schema fields raise ValueError."""
        for field in ("agent_name", "description", "system_prompt", "scope", "audience", "tone"):
            broken = dict(self.valid_def)
            broken[field] = ""
            with self.assertRaises(ValueError, msg=f"Should reject empty {field}"):
                validate_agent_definition(broken, self.registry)

    def test_invalid_retrieval_settings_rejected(self) -> None:
        """Verify that invalid retrieval parameters fail validation via RetrieverConfig."""
        invalid_cases = [
            {"chunk_size": 0, "chunk_overlap": 0, "top_k": 4, "minimum_similarity_score": 0.0},
            {"chunk_size": 100, "chunk_overlap": 150, "top_k": 4, "minimum_similarity_score": 0.0},
            {"chunk_size": 100, "chunk_overlap": 20, "top_k": -1, "minimum_similarity_score": 0.0},
            {"chunk_size": 100, "chunk_overlap": 20, "top_k": 4, "minimum_similarity_score": 1.5},
        ]
        for cfg in invalid_cases:
            broken = dict(self.valid_def)
            broken["retrieval_config"] = cfg
            with self.assertRaises(ValueError):
                validate_agent_definition(broken, self.registry)

    def test_irrelevant_or_unknown_tool_rejection(self) -> None:
        """Verify that tools not registered in ToolRegistry are rejected."""
        broken = dict(self.valid_def)
        broken["tools"] = [{"name": "unauthorized_sql_exec", "justification": "Direct DB access"}]
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(broken, self.registry)
        self.assertIn("not registered in ToolRegistry", str(ctx.exception))

    def test_empty_tool_list_allowed(self) -> None:
        """Verify that an empty tool list is valid when no tools are justified."""
        no_tools_def = dict(self.valid_def)
        no_tools_def["tools"] = []
        validated = validate_agent_definition(no_tools_def, self.registry)
        self.assertEqual(validated["tools"], [])

    def test_preservation_of_groundability_restrictions(self) -> None:
        """Verify that unsupported topics remain in must_refuse_topics and cannot be in must_answer."""
        # Overlapping must_answer and must_refuse
        conflicted = dict(self.valid_def)
        conflicted["must_answer_topics"] = ["corporate revenue"]
        conflicted["must_refuse_topics"] = ["corporate revenue"]
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(conflicted, self.registry)
        self.assertIn("cannot be in both", str(ctx.exception))

    def test_generator_synthesizes_definition_and_injects_guardrails(self) -> None:
        """Verify AgentGenerator builds complete definition and reinforces safety constraints."""
        mock_output = {
            "agent_name": "SynthesizedAgent",
            "description": "Assistant for pump diagnostics",
            "system_prompt": "You are SynthesizedAgent. Assist technicians with pump parameters.",
            "selected_tools": [
                {"name": "calculator", "justification": "Flow rate calculation"},
                {"name": "nonexistent_tool", "justification": "Fake tool"},  # should be discarded
            ],
            "retrieval_config": {
                "chunk_size": 200,
                "chunk_overlap": 40,
                "top_k": 5,
                "minimum_similarity_score": 0.05,
            },
        }

        client = MockOllamaClient(mock_output)
        generator = AgentGenerator(client=client, tool_registry=self.registry)

        reqs = {
            "permitted_scope": "Diagnostics for slurry pump operations.",
            "audience": "Technicians",
            "tone": "Direct and factual",
            "must_answer_topics": ["calculate pump flow rate", "impeller alloy"],
            "must_refuse_topics": ["company financial statements [Refusal: Unsupported]"],
        }

        agent_def = generator.generate_agent_definition(requirements=reqs)

        # Check fields
        self.assertEqual(agent_def["agent_name"], "SynthesizedAgent")
        self.assertEqual(agent_def["scope"], "Diagnostics for slurry pump operations.")
        self.assertEqual(agent_def["must_answer_topics"], ["calculate pump flow rate", "impeller alloy"])
        self.assertIn("company financial statements [Refusal: Unsupported]", agent_def["must_refuse_topics"])

        # Tool filtering: nonexistent_tool rejected, calculator retained
        tool_names = [t["name"] for t in agent_def["tools"]]
        self.assertIn("calculator", tool_names)
        self.assertNotIn("nonexistent_tool", tool_names)

        # Safety guardrails reinforced in system prompt
        sys_prompt = agent_def["system_prompt"]
        self.assertIn("INDIRECT INJECTION DEFENSE", sys_prompt)
        self.assertIn("GROUNDING & CITATIONS", sys_prompt)
        self.assertIn("REFUSAL POLICY", sys_prompt)

    def test_unnecessary_registered_tools_rejected(self) -> None:
        """Verify unnecessary registered tools like current_date and calculator are rejected."""
        # Non-calculation task with calculator tool
        broken_calc = dict(self.valid_def)
        broken_calc["must_answer_topics"] = ["pump operating temperature limit"]
        broken_calc["scope"] = "Pump temperature guidance."
        broken_calc["tools"] = [{"name": "calculator", "justification": "General math"}]
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(broken_calc, self.registry)
        self.assertIn("Unnecessary tool 'calculator' rejected", str(ctx.exception))

        # Contextual / speculative current_date
        broken_date = dict(self.valid_def)
        broken_date["must_answer_topics"] = ["pump operating temperature limit"]
        broken_date["scope"] = "Pump temperature guidance."
        broken_date["tools"] = [
            {"name": "current_date", "justification": "To provide current date for context, though not strictly required"}
        ]
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(broken_date, self.registry)
        self.assertIn("Unnecessary tool 'current_date' rejected", str(ctx.exception))

        # csv_lookup without explicit lookup requirement
        broken_csv = dict(self.valid_def)
        broken_csv["must_answer_topics"] = ["pump operating temperature limit"]
        broken_csv["scope"] = "Pump temperature guidance."
        broken_csv["tools"] = [{"name": "csv_lookup", "justification": "Because CSV documents exist"}]
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(broken_csv, self.registry)
        self.assertIn("Unnecessary tool 'csv_lookup' rejected", str(ctx.exception))

    def test_positive_case_genuinely_required_tool_retained(self) -> None:
        """Verify positive cases where tools are genuinely required by the requested tasks."""
        # Genuine calculator requirement
        calc_def = dict(self.valid_def)
        calc_def["scope"] = "Thermodynamic calculation engine."
        calc_def["must_answer_topics"] = ["calculate thermal expansion and flow rate"]
        calc_def["tools"] = [{"name": "calculator", "justification": "Calculate thermal expansion"}]
        validated_calc = validate_agent_definition(calc_def, self.registry)
        self.assertEqual(len(validated_calc["tools"]), 1)
        self.assertEqual(validated_calc["tools"][0]["name"], "calculator")

        # Genuine current_date requirement
        date_def = dict(self.valid_def)
        date_def["scope"] = "Maintenance expiration tracker."
        date_def["must_answer_topics"] = ["check maintenance due date against current date"]
        date_def["tools"] = [{"name": "current_date", "justification": "Check current date"}]
        validated_date = validate_agent_definition(date_def, self.registry)
        self.assertEqual(len(validated_date["tools"]), 1)
        self.assertEqual(validated_date["tools"][0]["name"], "current_date")

    def test_citation_schema_requires_chunk_id(self) -> None:
        """Verify output contract citation specification requires chunk_id alongside source and excerpt."""
        broken_contract_def = dict(self.valid_def)
        broken_contract_def["output_contract"] = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "answer": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"source": {"type": "string"}, "excerpt": {"type": "string"}},
                        "required": ["source", "excerpt"],  # missing chunk_id!
                    },
                },
            },
            "required": ["status", "answer", "citations"],
        }
        with self.assertRaises(ValueError) as ctx:
            validate_agent_definition(broken_contract_def, self.registry)
        self.assertIn("must require 'chunk_id'", str(ctx.exception))

    def test_save_agent_definition_json(self) -> None:
        """Test persisting agent definition to agent_definition.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "agent_definition.json"
            save_agent_definition(self.valid_def, out_file)
            self.assertTrue(out_file.exists())

            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["agent_name"], "FurnaceMonitorAgent")
            self.assertEqual(loaded["tools"][0]["name"], "calculator")


class TestLiveOllamaAgentGenerator(unittest.TestCase):
    """Live smoke test against local Ollama instance with real qwen2.5:3b."""

    def test_live_agent_generator_smoke(self) -> None:
        client = OllamaClient(timeout=45.0)

        # Check connectivity
        try:
            client.generate("Ping", max_tokens=2)
        except Exception as e:
            self.skipTest(f"Ollama server not reachable: {e}")

        registry = ToolRegistry()
        generator = AgentGenerator(client=client, tool_registry=registry)

        # Realistic verified requirements from previous steps
        reqs = {
            "permitted_scope": "Technical guidance on Slurry Pump P-302 operating temperature limits and maintenance inspection intervals.",
            "audience": "Maintenance technicians",
            "tone": "Professional, direct and concise",
            "must_answer_topics": [
                "operating temperature limit of Slurry Pump P-302",
                "standard maintenance inspection interval hours",
            ],
            "must_refuse_topics": [
                "historical maintenance logs and automated scheduling [Refusal: Unrequested]",
                "quarterly revenue of pump manufacturing division [Refusal: Unsupported]",
            ],
        }

        docs = [
            Document(
                doc_id="equipment/slurry_pump_spec.txt",
                text="Operating maximum flow rate: 450 GPM. Standard operating temperature limit: 85 degrees Celsius. Inspection interval: 250 operating hours.",
                metadata={"source": "equipment/slurry_pump_spec.txt"},
            )
        ]

        definition = generator.generate_agent_definition(requirements=reqs, documents=docs)

        # Verify definition validity
        self.assertTrue(isinstance(definition["agent_name"], str) and len(definition["agent_name"]) > 0)
        self.assertTrue(isinstance(definition["description"], str) and len(definition["description"]) > 0)
        self.assertTrue(isinstance(definition["system_prompt"], str) and len(definition["system_prompt"]) > 0)

        # Verify groundability restrictions preserved
        self.assertEqual(definition["must_answer_topics"], reqs["must_answer_topics"])
        for refuse_item in reqs["must_refuse_topics"]:
            self.assertIn(refuse_item, definition["must_refuse_topics"])

        # Verify retrieval config is valid
        self.assertIn("chunk_size", definition["retrieval_config"])
        self.assertIn("top_k", definition["retrieval_config"])
        self.assertGreater(definition["retrieval_config"]["chunk_size"], 0)

        # Verify tools are empty as no tool is requested or justified for simple Q&A
        self.assertEqual(definition["tools"], [])

        # Verify output contract requires chunk_id
        self.assertEqual(definition["output_contract"], OUTPUT_CONTRACT_SCHEMA)
        self.assertIn(
            "chunk_id",
            definition["output_contract"]["properties"]["citations"]["items"]["required"],
        )


if __name__ == "__main__":
    unittest.main()
