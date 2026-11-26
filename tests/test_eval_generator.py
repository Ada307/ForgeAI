"""Unit and live tests for Automatic Evaluation Generator (forge/eval_generator.py)."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import unittest

from forge.documents import Document
from forge.eval_generator import (
    DEFAULT_TARGET_COUNTS,
    EvaluationCase,
    EvaluationSetGenerator,
    build_documents_lookup,
    validate_evaluation_case,
)
from forge.llm import OllamaClient


class MockOllamaClient:
    """Mock client returning sequential JSON responses for evaluation generator."""

    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return {"cases": []}


class TestEvaluationGeneratorUnit(unittest.TestCase):
    """Mocked unit tests for case validation, evidence checking, and constraints."""

    def setUp(self) -> None:
        self.doc1 = Document(
            doc_id="pump_spec.txt",
            text="Slurry Pump P-302 standard operating temperature limit is 85 degrees Celsius.",
            metadata={"source": "pump_spec.txt", "filename": "pump_spec.txt"},
        )
        self.doc2 = Document(
            doc_id="maintenance_guide.txt",
            text="Slurry Pump P-302 routine maintenance inspection interval is 250 operating hours.",
            metadata={"source": "maintenance_guide.txt", "filename": "maintenance_guide.txt"},
        )
        self.documents = [self.doc1, self.doc2]
        self.doc_lookup = build_documents_lookup(self.documents)

        self.requirements = {
            "scope": "Technical guidance for Slurry Pump P-302.",
            "must_answer_topics": ["operating temperature limit", "maintenance inspection interval"],
            "must_refuse_topics": ["company financial revenue and profits [Refusal: Unsupported]"],
        }

    def test_valid_answerable_case_passes(self) -> None:
        case = EvaluationCase(
            id="eval_001",
            category="answerable",
            question="What is the operating temperature limit for Slurry Pump P-302?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius"],
            expected_sources=["pump_spec.txt"],
            expected_evidence=[
                {
                    "source": "pump_spec.txt",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                }
            ],
        )
        is_valid, reason = validate_evaluation_case(case, self.doc_lookup, set())
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_fabricated_evidence_rejected(self) -> None:
        case = EvaluationCase(
            id="eval_002",
            category="answerable",
            question="What is the operating temperature limit?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius"],
            expected_sources=["pump_spec.txt"],
            expected_evidence=[
                {
                    "source": "pump_spec.txt",
                    "excerpt": "Nonexistent text not in document at all",
                }
            ],
        )
        is_valid, reason = validate_evaluation_case(case, self.doc_lookup, set())
        self.assertFalse(is_valid)
        self.assertIn("Fabricated excerpt", str(reason))

    def test_duplicate_question_rejected(self) -> None:
        case = EvaluationCase(
            id="eval_003",
            category="answerable",
            question="What is the operating temperature limit for Slurry Pump P-302?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius"],
            expected_sources=["pump_spec.txt"],
            expected_evidence=[
                {
                    "source": "pump_spec.txt",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                }
            ],
        )
        seen = {"whatistheoperatingtemperaturelimitforslurrypumpp302"}
        is_valid, reason = validate_evaluation_case(case, self.doc_lookup, seen)
        self.assertFalse(is_valid)
        self.assertIn("Duplicate question detected", str(reason))

    def test_cross_document_requires_two_distinct_sources(self) -> None:
        # Case citing only 1 document should fail cross_document validation
        case_single_source = EvaluationCase(
            id="eval_004",
            category="cross_document",
            question="Compare limits in pump spec.",
            expected_status="answered",
            expected_key_facts=["Fact 1"],
            expected_sources=["pump_spec.txt"],
            expected_evidence=[
                {
                    "source": "pump_spec.txt",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                }
            ],
        )
        is_valid, reason = validate_evaluation_case(case_single_source, self.doc_lookup, set())
        self.assertFalse(is_valid)
        self.assertIn("requires evidence from at least 2 distinct files", str(reason))

        # Case citing 2 distinct documents with valid excerpts passes
        case_two_sources = EvaluationCase(
            id="eval_005",
            category="cross_document",
            question="What are the operating limit and inspection interval for P-302?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius limit", "250 operating hours interval"],
            expected_sources=["pump_spec.txt", "maintenance_guide.txt"],
            expected_evidence=[
                {
                    "source": "pump_spec.txt",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                },
                {
                    "source": "maintenance_guide.txt",
                    "excerpt": "routine maintenance inspection interval is 250 operating hours",
                },
            ],
        )
        is_valid, reason = validate_evaluation_case(case_two_sources, self.doc_lookup, set())
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_insufficient_documents_for_cross_document_raises_honestly(self) -> None:
        generator = EvaluationSetGenerator(client=MockOllamaClient([]))
        # Only 1 document supplied
        single_doc_corpus = [self.doc1]
        with self.assertRaises(RuntimeError) as ctx:
            generator.generate_eval_set(
                documents=single_doc_corpus,
                requirements=self.requirements,
                target_counts={"cross_document": 2},
            )
        self.assertIn("Insufficient documents for cross-document evaluation", str(ctx.exception))

    def test_save_eval_set_preserves_structure_and_counts(self) -> None:
        cases = [
            EvaluationCase(
                id="eval_001",
                category="answerable",
                question="What is the temp limit?",
                expected_status="answered",
                expected_key_facts=["85 C"],
                expected_sources=["pump_spec.txt"],
                expected_evidence=[{"source": "pump_spec.txt", "excerpt": "85 degrees Celsius"}],
            ),
            EvaluationCase(
                id="eval_002",
                category="out_of_scope",
                question="What is quarterly revenue?",
                expected_status="refused",
                expected_key_facts=["Financial revenue is unsupported"],
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "eval_set.json"
            generator = EvaluationSetGenerator(client=MockOllamaClient([]))
            saved_path = generator.save_eval_set(cases, out_file)
            self.assertTrue(saved_path.exists())

            data = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(data["total_cases"], 2)
            self.assertEqual(data["category_distribution"]["answerable"], 1)
            self.assertEqual(data["category_distribution"]["out_of_scope"], 1)
            self.assertEqual(len(data["cases"]), 2)


class TestLiveEvaluationGenerator(unittest.TestCase):
    """Live test generating full evaluation set against local Ollama instance with qwen2.5:3b."""

    def test_live_generation_smoke(self) -> None:
        client = OllamaClient(timeout=60.0)
        try:
            client.generate("Ping", max_tokens=2)
        except Exception as e:
            self.skipTest(f"Ollama server not reachable: {e}")

        # Create two rich test documents with ample specifications
        doc1 = Document(
            doc_id="pump_spec.txt",
            text=(
                "Slurry Pump P-302 Operational Specifications:\n"
                "Standard operating temperature limit is 85 degrees Celsius.\n"
                "Maximum flow rate capacity is 450 GPM.\n"
                "Design pressure rating is 150 PSI.\n"
                "Impeller diameter is 12 inches.\n"
                "Motor power rating is 75 horsepower.\n"
                "Suction nozzle flange diameter is 6 inches.\n"
                "Discharge nozzle flange diameter is 4 inches.\n"
                "Casing material is high-chrome white iron.\n"
                "Shaft sleeve material is ceramic-coated stainless steel.\n"
                "Maximum allowable operating speed is 1750 RPM.\n"
                "Minimum continuous stable flow is 80 GPM.\n"
            ),
            metadata={"source": "pump_spec.txt", "filename": "pump_spec.txt"},
        )
        doc2 = Document(
            doc_id="maintenance_guide.txt",
            text=(
                "Slurry Pump P-302 Maintenance Procedures:\n"
                "Routine maintenance inspection interval is 250 operating hours.\n"
                "Mechanical seal lubricant must be changed every 500 hours.\n"
                "Recommended bearing lubrication grease is Mobilith SHC 220.\n"
                "Shaft alignment tolerance is 0.05 mm.\n"
                "Vibration velocity warning threshold is 4.5 mm/s RMS.\n"
                "Bearing operating temperature limit is 90 degrees Celsius.\n"
                "Mechanical seal flush water pressure must be 15 PSI above stuffing box pressure.\n"
                "Casing bolt tightening torque requirement is 320 Nm.\n"
                "Impeller clearance adjustment should be set to 0.75 mm.\n"
                "Gland packing replacement is required after 1000 operating hours.\n"
            ),
            metadata={"source": "maintenance_guide.txt", "filename": "maintenance_guide.txt"},
        )

        requirements = {
            "scope": "Technical guidance on Slurry Pump P-302 operational specifications and maintenance procedures.",
            "must_answer_topics": [
                "operating temperature limits and pressure ratings",
                "routine maintenance inspection intervals and lubricants",
            ],
            "must_refuse_topics": [
                "company financial revenue and stock valuation [Refusal: Unsupported]",
                "executive compensation and employee payroll [Refusal: Unsupported]",
            ],
        }

        # Target: 10 answerable, 4 out_of_scope, 3 ambiguous, 3 cross_document = 20 cases
        generator = EvaluationSetGenerator(client=client, batch_size=5)
        cases = generator.generate_eval_set(
            documents=[doc1, doc2],
            requirements=requirements,
            target_counts=DEFAULT_TARGET_COUNTS,
            max_retries=3,
        )

        # Assert total case count is at least 20
        self.assertGreaterEqual(len(cases), 20)

        # Check distribution
        counts: Dict[str, int] = {}
        for c in cases:
            counts[c.category] = counts.get(c.category, 0) + 1

        self.assertEqual(counts.get("answerable", 0), 10)
        self.assertEqual(counts.get("out_of_scope", 0), 4)
        self.assertEqual(counts.get("ambiguous", 0), 3)
        self.assertEqual(counts.get("cross_document", 0), 3)

        # Verify all cross-document cases genuinely have >= 2 distinct sources
        for c in cases:
            if c.category == "cross_document":
                sources = {ev["source"] for ev in c.expected_evidence}
                self.assertGreaterEqual(len(sources), 2)

        # Test saving
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "eval_set.json"
            saved_path = generator.save_eval_set(cases, out_file)
            self.assertTrue(saved_path.exists())


if __name__ == "__main__":
    unittest.main()
