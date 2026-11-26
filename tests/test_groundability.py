"""Unit and live verification tests for Groundability Guard (forge/groundability.py)."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from forge.documents import Document
from forge.groundability import (
    GroundabilityGuard,
    save_requirements,
    validate_citation_excerpt,
)
from forge.llm import OllamaClient
from forge.retrieval import RetrievalResult, TFIDFRetriever


class MockOllamaClient:
    """Mock OllamaClient for deterministic groundability unit tests."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return {"status": "unsupported", "reason": "Default mock unsupported response"}


class TestGroundabilityGuardUnit(unittest.TestCase):
    """Mocked unit tests for Groundability Guard logic and citation validation."""

    def setUp(self) -> None:
        self.doc = Document(
            doc_id="specs/valve.txt",
            text="The emergency pressure relief valve opens automatically at 150 PSI.",
            metadata={"source": "specs/valve.txt"},
        )
        self.retriever = TFIDFRetriever(documents=[self.doc])

    def test_supported_topic_with_valid_citation(self) -> None:
        """Verify supported topic with genuine citation excerpt remains permitted."""
        mock_response = {
            "status": "supported",
            "reason": "Directly stated in specification.",
            "citations": [
                {
                    "source": "specs/valve.txt",
                    "chunk_id": "specs/valve.txt:c0",
                    "excerpt": "relief valve opens automatically at 150 PSI",
                }
            ],
            "knowledge_gaps": [],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))

        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["emergency relief valve opening pressure"],
            "must_refuse_topics": [],
        }

        checked = guard.check_requirements(reqs)
        self.assertIn("emergency relief valve opening pressure", checked["must_answer_topics"])
        self.assertEqual(checked["original_must_answer_topics"], ["emergency relief valve opening pressure"])
        self.assertEqual(len(checked["groundability_report"]), 1)
        self.assertEqual(checked["groundability_report"][0]["status"], "supported")
        self.assertTrue(checked["groundability_report"][0]["evidence_validated"])

    def test_unsupported_topic_detection_and_refusal_routing(self) -> None:
        """Verify unsupported topics are excluded from must_answer and routed to refusal."""
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([]))

        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["chemical composition of lubrication oil"],
            "must_refuse_topics": ["financial budgets"],
        }

        # Topic has zero matching chunks in corpus
        checked = guard.check_requirements(reqs)

        # Must not be permitted
        self.assertEqual(checked["must_answer_topics"], [])
        # Routed to must_refuse_topics
        self.assertTrue(any("chemical composition" in t for t in checked["must_refuse_topics"]))
        # Original preserved
        self.assertEqual(checked["original_must_answer_topics"], ["chemical composition of lubrication oil"])
        # Knowledge gaps recorded
        self.assertGreater(len(checked["knowledge_gaps"]), 0)

    def test_citation_validation_catches_hallucinated_excerpt(self) -> None:
        """Verify that LLM claims of support with hallucinated excerpts are rejected."""
        mock_response = {
            "status": "supported",
            "reason": "The valve operates at 5000 RPM using magnetic bearings.",
            "citations": [
                {
                    "source": "specs/valve.txt",
                    "chunk_id": "specs/valve.txt:c0",
                    "excerpt": "valve operates at 5000 RPM using magnetic bearings",
                }
            ],
            "knowledge_gaps": [],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))

        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["valve rotation speed"],
            "must_refuse_topics": [],
        }

        checked = guard.check_requirements(reqs)

        # Hallucinated excerpt was not found in chunk -> downgraded to unsupported
        self.assertEqual(checked["must_answer_topics"], [])
        report = checked["groundability_report"][0]
        self.assertEqual(report["status"], "unsupported")
        self.assertFalse(report["evidence_validated"])
        self.assertIn("Citation validation failed", report["reason"])

    def test_similarity_scores_alone_do_not_equal_support(self) -> None:
        """Verify that lexical match with irrelevant facts is rejected as unsupported."""
        # Query matches words "valve" or "pressure", but does not cover the topic
        mock_response = {
            "status": "unsupported",
            "reason": "Chunks discuss pressure relief but do not state the valve manufacturer or warranty.",
            "citations": [],
            "knowledge_gaps": ["Manufacturer warranty details are absent."],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))

        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["valve manufacturer warranty terms"],
            "must_refuse_topics": [],
        }

        checked = guard.check_requirements(reqs)
        self.assertEqual(checked["must_answer_topics"], [])
        self.assertTrue(any("warranty" in t for t in checked["must_refuse_topics"]))

    def test_save_requirements_json(self) -> None:
        """Test persisting requirements to requirements.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "requirements.json"
            data = {
                "scope": "Test Scope",
                "audience": "Test Audience",
                "tone": "Professional",
                "must_answer_topics": ["Topic A"],
                "must_refuse_topics": ["Topic B"],
                "groundability_report": [],
            }
            save_requirements(data, out_file)
            self.assertTrue(out_file.exists())

            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["scope"], "Test Scope")
            self.assertEqual(loaded["must_answer_topics"], ["Topic A"])

    def test_citation_mismatched_source_or_chunk_id_fails(self) -> None:
        """Verify citations pointing to wrong chunk ID or wrong source are rejected even if excerpt exists elsewhere."""
        # Text is in specs/valve.txt:c0, but citation claims specs/valve.txt:c99
        mock_response = {
            "status": "supported",
            "reason": "Directly stated.",
            "citations": [
                {
                    "source": "specs/valve.txt",
                    "chunk_id": "specs/valve.txt:c99",  # wrong chunk_id
                    "excerpt": "relief valve opens automatically at 150 PSI",
                }
            ],
            "knowledge_gaps": [],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))
        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["emergency relief valve opening pressure"],
            "must_refuse_topics": [],
        }
        checked = guard.check_requirements(reqs)
        self.assertEqual(checked["must_answer_topics"], [])
        self.assertEqual(checked["groundability_report"][0]["status"], "unsupported")
        self.assertFalse(checked["groundability_report"][0]["evidence_validated"])

    def test_partially_supported_topic_permits_only_supported_portion(self) -> None:
        """Verify partially supported topics permit only the explicitly supported portion."""
        mock_response = {
            "status": "partially_supported",
            "supported_portion": "Opening pressure of 150 PSI only",
            "reason": "Opening pressure is documented, but flow rate is absent.",
            "citations": [
                {
                    "source": "specs/valve.txt",
                    "chunk_id": "specs/valve.txt:c0",
                    "excerpt": "relief valve opens automatically at 150 PSI",
                }
            ],
            "knowledge_gaps": ["Flow rate discharge capacity is not documented."],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))
        reqs = {
            "scope": "Valve monitor",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["relief valve pressure and discharge flow rate"],
            "must_refuse_topics": [],
        }
        checked = guard.check_requirements(reqs)
        self.assertEqual(checked["must_answer_topics"], ["Opening pressure of 150 PSI only"])
        self.assertTrue(any("discharge flow rate" in r for r in checked["must_refuse_topics"]))
        self.assertIn("Flow rate discharge capacity is not documented.", checked["knowledge_gaps"])

    def test_permitted_scope_constrained_and_original_scope_preserved(self) -> None:
        """Verify scope is constrained to verified capabilities while original scope is preserved."""
        mock_response = {
            "status": "supported",
            "reason": "Opening pressure is documented.",
            "citations": [
                {
                    "source": "specs/valve.txt",
                    "chunk_id": "specs/valve.txt:c0",
                    "excerpt": "relief valve opens automatically at 150 PSI",
                }
            ],
            "knowledge_gaps": [],
        }
        guard = GroundabilityGuard(retriever=self.retriever, client=MockOllamaClient([mock_response]))
        reqs = {
            "scope": "Comprehensive valve assistant covering telemetry, predictive analytics, and budgeting.",
            "audience": "Engineers",
            "tone": "Technical",
            "must_answer_topics": ["valve opening pressure"],
            "must_refuse_topics": [],
        }
        checked = guard.check_requirements(reqs)
        self.assertEqual(
            checked["original_scope"],
            "Comprehensive valve assistant covering telemetry, predictive analytics, and budgeting.",
        )
        self.assertIn("valve opening pressure", checked["scope"])
        self.assertNotIn("budgeting", checked["scope"])


class TestLiveOllamaGroundability(unittest.TestCase):
    """Live integration smoke test against local Ollama instance with real qwen2.5:3b."""

    def test_live_groundability_smoke(self) -> None:
        client = OllamaClient(timeout=30.0)

        # Check connectivity
        try:
            client.generate("Ping", max_tokens=2)
        except Exception as e:
            self.skipTest(f"Ollama server not reachable: {e}")

        # Temporary document with specific facts
        docs = [
            Document(
                doc_id="furnace_manual.txt",
                text=(
                    "Blast Furnace No. 3 Operating Parameters:\n"
                    "Maximum hearth temperature: 1650 degrees Celsius.\n"
                    "Hot blast pressure: 3.5 bar.\n"
                    "Daily pig iron tapping capacity: 4200 metric tons."
                ),
                metadata={"source": "furnace_manual.txt", "file_type": "txt"},
            )
        ]
        retriever = TFIDFRetriever(documents=docs)
        guard = GroundabilityGuard(retriever=retriever, client=client)

        reqs = {
            "scope": "Blast furnace operating assistant",
            "audience": "Furnace operators",
            "tone": "Technical and concise",
            "must_answer_topics": [
                "maximum hearth temperature of Blast Furnace No. 3",  # Supported
                "annual profit margin of the steel mill",             # Unsupported
            ],
            "must_refuse_topics": ["employee payroll data"],
        }

        enriched = guard.check_requirements(reqs)

        # Verify that original topics and scope are preserved
        self.assertEqual(len(enriched["original_must_answer_topics"]), 2)
        self.assertEqual(enriched["original_scope"], "Blast furnace operating assistant")
        self.assertIn("hearth temperature", enriched["scope"].lower())

        # Verify supported topic is retained
        self.assertTrue(
            any("hearth temperature" in t.lower() for t in enriched["must_answer_topics"]),
            f"Expected hearth temperature in {enriched['must_answer_topics']}",
        )

        # Verify unsupported topic is excluded from must_answer and present in must_refuse
        self.assertFalse(
            any("profit margin" in t.lower() for t in enriched["must_answer_topics"]),
            f"Profit margin should not be permitted in {enriched['must_answer_topics']}",
        )
        self.assertTrue(
            any("profit margin" in t.lower() for t in enriched["must_refuse_topics"]),
            f"Profit margin should be in refusal topics: {enriched['must_refuse_topics']}",
        )

        # Verify groundability report structure
        report = enriched["groundability_report"]
        self.assertEqual(len(report), 2)
        statuses = {r["topic"]: r["status"] for r in report}

        self.assertIn(statuses["maximum hearth temperature of Blast Furnace No. 3"], {"supported", "partially_supported"})
        self.assertEqual(statuses["annual profit margin of the steel mill"], "unsupported")


if __name__ == "__main__":
    unittest.main()
