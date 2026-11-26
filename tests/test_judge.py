"""Mocked unit tests for Automated Judge (forge/judge.py)."""

from typing import Any, Dict, List, Optional
import unittest

from forge.eval_generator import EvaluationCase
from forge.judge import AutomatedJudge, CaseJudgment, EvaluationScorecard
from forge.runner import AgentExecutionResult


class MockOllamaJudgeClient:
    """Mock LLM client returning pre-configured judge responses."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return {
            "factual_score": 1.0,
            "grounding_score": 1.0,
            "passed": True,
            "reason": "Default mock judge response: passed.",
        }


class TestAutomatedJudgeUnit(unittest.TestCase):
    """Essential mocked unit tests for AutomatedJudge."""

    def setUp(self) -> None:
        self.mock_client = MockOllamaJudgeClient()
        self.judge = AutomatedJudge(client=self.mock_client)

        self.answerable_case = EvaluationCase(
            id="eval_001",
            category="answerable",
            question="What is the operating temperature limit of Slurry Pump P-302?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius"],
            expected_sources=["pump_spec.txt"],
            expected_evidence=[{"source": "pump_spec.txt", "excerpt": "85 degrees Celsius"}],
        )

        self.out_of_scope_case = EvaluationCase(
            id="eval_002",
            category="out_of_scope",
            question="What is the company's quarterly revenue?",
            expected_status="refused",
            expected_key_facts=["Financial information is out of scope"],
            expected_sources=[],
            expected_evidence=[],
        )

        self.cross_doc_case = EvaluationCase(
            id="eval_003",
            category="cross_document",
            question="What are the temperature limit and inspection interval for P-302?",
            expected_status="answered",
            expected_key_facts=["85 degrees Celsius limit", "250 operating hours interval"],
            expected_sources=["pump_spec.txt", "maintenance_guide.txt"],
            expected_evidence=[
                {"source": "pump_spec.txt", "excerpt": "85 degrees Celsius"},
                {"source": "maintenance_guide.txt", "excerpt": "250 operating hours"},
            ],
        )

    def test_answerable_case_passes_with_correct_facts_and_citations(self) -> None:
        """Verify compliant answerable response with verified citations passes."""
        result = AgentExecutionResult(
            status="answered",
            answer="The standard operating temperature limit is 85 degrees Celsius.",
            citations=[
                {
                    "source": "pump_spec.txt",
                    "chunk_id": "pump_spec.txt:c0",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                }
            ],
            retrieved_evidence=[
                {
                    "metadata": {"source": "pump_spec.txt"},
                    "text": "operating temperature limit is 85 degrees Celsius",
                }
            ],
            tool_executions=[],
            execution_errors=[],
        )

        self.mock_client.responses = [
            {
                "factual_score": 1.0,
                "grounding_score": 1.0,
                "passed": True,
                "reason": "Directly supported by pump_spec.txt evidence and states 85 degrees Celsius.",
            }
        ]

        judgment = self.judge.judge_case(self.answerable_case, result)

        self.assertTrue(judgment.passed)
        self.assertTrue(judgment.status_match)
        self.assertTrue(judgment.citation_valid)
        self.assertEqual(judgment.factual_score, 1.0)
        self.assertEqual(judgment.grounding_score, 1.0)
        self.assertEqual(len(self.mock_client.calls), 1)

    def test_status_mismatch_fails_deterministically(self) -> None:
        """Verify status mismatch fails immediately without relying on LLM judge."""
        # Expected 'refused', but agent returned 'answered'
        result = AgentExecutionResult(
            status="answered",
            answer="The quarterly revenue was 10 million dollars.",
            citations=[],
            retrieved_evidence=[],
            tool_executions=[],
            execution_errors=[],
        )

        judgment = self.judge.judge_case(self.out_of_scope_case, result)

        self.assertFalse(judgment.passed)
        self.assertFalse(judgment.status_match)
        self.assertIn("Status mismatch", judgment.reason)
        # LLM judge should NOT even be called for deterministic failure
        self.assertEqual(len(self.mock_client.calls), 0)

    def test_execution_errors_fail_deterministically(self) -> None:
        """Verify execution errors trigger deterministic failure."""
        result = AgentExecutionResult(
            status="answered",
            answer="The pump limit is 85 C.",
            citations=[],
            retrieved_evidence=[],
            tool_executions=[],
            execution_errors=["Fabricated or invalid citation: chunk not found"],
        )

        judgment = self.judge.judge_case(self.answerable_case, result)

        self.assertFalse(judgment.passed)
        self.assertIn("Execution errors occurred", judgment.reason)
        self.assertEqual(len(self.mock_client.calls), 0)

    def test_cross_document_single_source_citation_fails(self) -> None:
        """Verify cross-document case fails citation check if evidence spans only 1 file."""
        result = AgentExecutionResult(
            status="answered",
            answer="Limit is 85 C and interval is 250 hours.",
            citations=[
                {
                    "source": "pump_spec.txt",
                    "chunk_id": "pump_spec.txt:c0",
                    "excerpt": "85 degrees Celsius",
                }
            ],
            retrieved_evidence=[
                {"metadata": {"source": "pump_spec.txt"}, "text": "85 degrees Celsius"}
            ],
            tool_executions=[],
            execution_errors=[],
        )

        judgment = self.judge.judge_case(self.cross_doc_case, result)

        self.assertFalse(judgment.passed)
        self.assertFalse(judgment.citation_valid)
        self.assertIn("requires citations from at least 2 distinct files", judgment.reason)
        self.assertEqual(len(self.mock_client.calls), 0)

    def test_malformed_judge_response_fails_gracefully(self) -> None:
        """Verify malformed or invalid LLM judge output is never silently passed."""
        result = AgentExecutionResult(
            status="answered",
            answer="The limit is 85 degrees Celsius.",
            citations=[{"source": "pump_spec.txt", "chunk_id": "c0", "excerpt": "85 C"}],
            retrieved_evidence=[],
            tool_executions=[],
            execution_errors=[],
        )

        # Mock LLM returns garbage string instead of valid dict
        self.mock_client.responses = ["not a dict"]  # type: ignore

        judgment = self.judge.judge_case(self.answerable_case, result)

        self.assertFalse(judgment.passed)
        self.assertIn("Malformed judge response", judgment.reason)

    def test_scorecard_aggregation(self) -> None:
        """Verify scorecard aggregates totals, pass rates, category breakdown, and failed IDs."""
        case1 = self.answerable_case
        case2 = self.out_of_scope_case

        # Result 1: passes
        res1 = AgentExecutionResult(
            status="answered",
            answer="85 degrees Celsius.",
            citations=[{"source": "pump_spec.txt", "chunk_id": "c0", "excerpt": "85 degrees"}],
            retrieved_evidence=[],
            tool_executions=[],
            execution_errors=[],
        )
        # Result 2: fails due to status mismatch
        res2 = AgentExecutionResult(
            status="answered",
            answer="Here are company profits.",
            citations=[],
            retrieved_evidence=[],
            tool_executions=[],
            execution_errors=[],
        )

        self.mock_client.responses = [
            {"factual_score": 1.0, "grounding_score": 1.0, "passed": True, "reason": "Accurate"}
        ]

        scorecard = self.judge.judge_evaluation_set([case1, case2], [res1, res2])

        self.assertEqual(scorecard.total_cases, 2)
        self.assertEqual(scorecard.passed_cases, 1)
        self.assertEqual(scorecard.failed_cases, 1)
        self.assertEqual(scorecard.pass_rate, 0.5)
        self.assertEqual(scorecard.failed_case_ids, ["eval_002"])
        self.assertEqual(scorecard.category_scores["answerable"]["passed"], 1)
        self.assertEqual(scorecard.category_scores["out_of_scope"]["passed"], 0)


if __name__ == "__main__":
    unittest.main()
