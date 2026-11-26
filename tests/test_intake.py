"""Unit tests for Intake Interviewer (forge/intake.py)."""

from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock

from forge.documents import Document
from forge.intake import (
    IntakeInterviewer,
    build_document_preview,
    validate_requirements,
)


class MockOllamaClient:
    """Mock client for deterministic intake unit tests."""

    def __init__(self, json_responses: List[Dict[str, Any]]) -> None:
        self.json_responses = list(json_responses)
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.json_responses:
            return self.json_responses.pop(0)
        return {}


class TestIntakeInterviewer(unittest.TestCase):
    """Test suite for IntakeInterviewer."""

    def test_five_question_limit(self) -> None:
        """Verify that at most 5 clarifying questions are generated even if LLM returns more."""
        # Mock returning 8 questions
        mock_response = {
            "questions": [
                {"question": f"Question {i}?", "default_assumption": f"Assumption {i}"}
                for i in range(1, 9)
            ]
        }
        client = MockOllamaClient([mock_response])
        interviewer = IntakeInterviewer(client=client)

        docs = [Document(doc_id="doc1.txt", text="Industrial cooling system manual.", metadata={"source": "doc1.txt"})]
        questions = interviewer.generate_clarifying_questions(brief="Build cooling assistant", documents=docs)

        self.assertLessEqual(len(questions), 5)
        self.assertEqual(len(questions), 5)
        self.assertEqual(questions[0]["question"], "Question 1?")
        self.assertEqual(questions[4]["question"], "Question 5?")

    def test_unattended_mode_and_no_hallucinated_answers(self) -> None:
        """Verify unattended mode records explicit assumptions and never fabricates answers."""
        question_items = [
            {"question": "What is the operating pressure range?", "default_assumption": "Assume 100-150 PSI."},
            {"question": "Who is the primary operator?", "default_assumption": "Assume certified maintenance technician."},
        ]

        interviewer = IntakeInterviewer(client=MockOllamaClient([]))

        # Unattended mode without user interaction
        answers, assumptions = interviewer.collect_answers(
            questions=question_items,
            interactive=False,
            answers=None,
        )

        self.assertEqual(answers, {})
        self.assertEqual(len(assumptions), 2)
        self.assertEqual(assumptions["What is the operating pressure range?"], "Assume 100-150 PSI.")
        self.assertEqual(assumptions["Who is the primary operator?"], "Assume certified maintenance technician.")

    def test_conservative_assumptions_block_unrequested_capabilities(self) -> None:
        """Verify questions proposing unrequested capabilities default to conservative exclusions in unattended mode."""
        question_items = [
            {
                "question": "Should the assistant include historical maintenance logs and automated alerts?",
                "default_assumption": "Yes, to provide comprehensive support.",
            }
        ]
        interviewer = IntakeInterviewer(client=MockOllamaClient([]))
        answers, assumptions = interviewer.collect_answers(question_items, interactive=False)

        self.assertEqual(answers, {})
        assump = assumptions["Should the assistant include historical maintenance logs and automated alerts?"]
        self.assertTrue(assump.lower().startswith("no"), f"Expected conservative exclusion starting with 'No', got: {assump}")
        self.assertNotIn("yes", assump.lower())

    def test_user_provided_answers_merged_with_assumptions(self) -> None:
        """Verify provided answers are recorded and remaining questions record assumptions."""
        question_items = [
            {"question": "Q1?", "default_assumption": "A1_default"},
            {"question": "Q2?", "default_assumption": "A2_default"},
        ]
        interviewer = IntakeInterviewer(client=MockOllamaClient([]))

        provided = {"Q1?": "Specific user answer"}
        answers, assumptions = interviewer.collect_answers(
            questions=question_items,
            interactive=False,
            answers=provided,
        )

        self.assertEqual(answers["Q1?"], "Specific user answer")
        self.assertNotIn("Q1?", assumptions)
        self.assertEqual(assumptions["Q2?"], "A2_default")
        self.assertNotIn("Q2?", answers)

    def test_requirements_validation(self) -> None:
        """Verify schema and type validation for synthesized requirements."""
        valid_data = {
            "scope": "Cooling system diagnostic advisor",
            "audience": "Maintenance engineers",
            "tone": "Direct and authoritative",
            "must_answer_topics": ["pressure regulation", "temperature threshold"],
            "must_refuse_topics": ["financial advice", "unrelated systems"],
        }
        validated = validate_requirements(valid_data)
        self.assertEqual(validated["scope"], "Cooling system diagnostic advisor")

        # Missing required string field
        invalid_scope = dict(valid_data)
        invalid_scope["scope"] = ""
        with self.assertRaises(ValueError):
            validate_requirements(invalid_scope)

        # Empty must_answer_topics list
        invalid_topics = dict(valid_data)
        invalid_topics["must_answer_topics"] = []
        with self.assertRaises(ValueError):
            validate_requirements(invalid_topics)

        # Non-list must_refuse_topics
        invalid_refuse = dict(valid_data)
        invalid_refuse["must_refuse_topics"] = "not a list"
        with self.assertRaises(ValueError):
            validate_requirements(invalid_refuse)

    def test_document_inspection_and_context_bounds(self) -> None:
        """Verify document content is inspected and omitted documents are tracked honestly."""
        docs = [
            Document(doc_id=f"doc_{i}.txt", text=f"Document content sentence for file {i}.", metadata={"source": f"doc_{i}.txt"})
            for i in range(10)
        ]

        # Use small word budget to trigger omission tracking
        preview, inspected, omitted = build_document_preview(docs, max_words_per_doc=10, total_word_budget=35)

        self.assertGreater(len(inspected), 0)
        self.assertGreater(len(omitted), 0)
        self.assertEqual(len(inspected) + len(omitted), len(docs))
        self.assertIn("NOTE: Inspected", preview)
        self.assertIn("Omitted documents", preview)


if __name__ == "__main__":
    unittest.main()
