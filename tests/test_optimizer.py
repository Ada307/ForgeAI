"""Focused mocked unit tests for Critic-Optimizer Loop (forge/optimizer.py)."""

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import unittest

from forge.eval_generator import EvaluationCase
from forge.generator import OUTPUT_CONTRACT_SCHEMA
from forge.judge import AutomatedJudge, CaseJudgment, EvaluationScorecard
from forge.optimizer import AgentOptimizer, OptimizationResult
from forge.runner import AgentExecutionResult


class MockOllamaOptimizerClient:
    """Mock client providing sequential JSON responses for optimizer unit tests."""

    def __init__(self, responses: Optional[List[Dict[str, Any]]] = None) -> None:
        self.responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt})
        if self.responses:
            return self.responses.pop(0)
        return {
            "status": "answered",
            "answer": "Default mock response",
            "citations": [],
        }


class MockJudge(AutomatedJudge):
    """Configurable mock judge returning controlled scorecards."""

    def __init__(self, scorecards: List[EvaluationScorecard]) -> None:
        self.scorecards = list(scorecards)
        self.call_count = 0

    def judge_evaluation_set(
        self, cases: Any, results: Any
    ) -> EvaluationScorecard:
        self.call_count += 1
        if self.scorecards:
            return self.scorecards.pop(0)
        # Default perfect scorecard
        return EvaluationScorecard(
            total_cases=len(cases),
            passed_cases=len(cases),
            failed_cases=0,
            pass_rate=1.0,
            category_scores={},
            failed_case_ids=[],
            judgments=[
                CaseJudgment(
                    case_id=c.id,
                    category=c.category,
                    passed=True,
                    status_match=True,
                    grounding_score=1.0,
                    factual_score=1.0,
                    citation_valid=True,
                    reason="Passed",
                )
                for c in cases
            ],
        )


class TestAgentOptimizerUnit(unittest.TestCase):
    """Focused mocked unit tests for AgentOptimizer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.docs_path = Path(self.temp_dir.name)
        (self.docs_path / "pump.txt").write_text(
            "Slurry Pump P-302 standard operating temperature limit is 85 degrees Celsius.",
            encoding="utf-8",
        )

        self.initial_def = {
            "agent_name": "PumpAgent",
            "description": "Assistant for Slurry Pump P-302",
            "system_prompt": "You are PumpAgent. Answer using provided evidence.",
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

        self.cases = [
            EvaluationCase(
                id="eval_001",
                category="answerable",
                question="What is the operating temperature limit?",
                expected_status="answered",
                expected_key_facts=["85 degrees Celsius"],
                expected_sources=["pump.txt"],
                expected_evidence=[{"source": "pump.txt", "excerpt": "85 degrees Celsius"}],
            ),
            EvaluationCase(
                id="eval_002",
                category="out_of_scope",
                question="What is company revenue?",
                expected_status="refused",
                expected_key_facts=["Financial information is out of scope"],
            ),
        ]

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_failure_classification_categories(self) -> None:
        """Verify critic classifies failed cases into OceanAI's four standard categories."""
        client = MockOllamaOptimizerClient(
            responses=[
                {
                    "classifications": [
                        {
                            "case_id": "eval_001",
                            "category": "retrieval_miss",
                            "diagnosis": "Missing evidence chunk in context",
                        },
                        {
                            "case_id": "eval_002",
                            "category": "wrong_refusal",
                            "diagnosis": "Refused an in-scope question",
                        },
                    ],
                    "proposed_action": {
                        "action_type": "update_retrieval_config",
                        "updated_retrieval_config": {
                            "chunk_size": 200,
                            "chunk_overlap": 40,
                            "top_k": 5,
                            "minimum_similarity_score": 0.0,
                        },
                        "reason": "Increase top_k to 5",
                    },
                }
            ]
        )

        optimizer = AgentOptimizer(client=client)

        failed_cases = self.cases
        failed_results = [
            AgentExecutionResult(
                status="answered",
                answer="I don't know",
                citations=[],
                retrieved_evidence=[],
            ),
            AgentExecutionResult(
                status="refused",
                answer="Refused",
                citations=[],
                retrieved_evidence=[],
            ),
        ]
        failed_judgments = [
            CaseJudgment(
                case_id="eval_001",
                category="answerable",
                passed=False,
                status_match=True,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=False,
                reason="Missing facts",
            ),
            CaseJudgment(
                case_id="eval_002",
                category="out_of_scope",
                passed=False,
                status_match=False,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=False,
                reason="Wrong refusal",
            ),
        ]

        classifications, action = optimizer.classify_and_diagnose(
            failed_cases, failed_results, failed_judgments, self.initial_def
        )

        self.assertEqual(len(classifications), 2)
        cat_map = {c["case_id"]: c["category"] for c in classifications}
        self.assertEqual(cat_map["eval_001"], "retrieval_miss")
        self.assertEqual(cat_map["eval_002"], "wrong_refusal")
        self.assertEqual(action["action_type"], "update_retrieval_config")

    def test_targeted_retrieval_config_fix_applied_without_full_regeneration(self) -> None:
        """Verify targeted retrieval_config update preserves all other agent attributes."""
        optimizer = AgentOptimizer(client=MockOllamaOptimizerClient())
        proposed_action = {
            "action_type": "update_retrieval_config",
            "updated_retrieval_config": {
                "chunk_size": 220,
                "chunk_overlap": 40,
                "top_k": 6,
                "minimum_similarity_score": 0.1,
            },
            "reason": "Expand retrieval parameters to capture evidence.",
        }

        new_def, action_type, summary, _reason = optimizer.apply_targeted_fix(
            self.initial_def, proposed_action
        )

        self.assertEqual(action_type, "update_retrieval_config")
        self.assertEqual(new_def["retrieval_config"]["top_k"], 6)
        self.assertEqual(new_def["retrieval_config"]["chunk_size"], 220)

        # Preserved fields
        self.assertEqual(new_def["agent_name"], self.initial_def["agent_name"])
        self.assertEqual(new_def["system_prompt"], self.initial_def["system_prompt"])
        self.assertEqual(new_def["scope"], self.initial_def["scope"])
        self.assertEqual(new_def["must_refuse_topics"], self.initial_def["must_refuse_topics"])
        self.assertEqual(new_def["tools"], self.initial_def["tools"])

    def test_targeted_system_prompt_fix_applied_without_full_regeneration(self) -> None:
        """Verify targeted system prompt refinement preserves configuration and schema."""
        optimizer = AgentOptimizer(client=MockOllamaOptimizerClient())
        new_prompt = self.initial_def["system_prompt"] + "\nDirective: strictly cite verified evidence."
        proposed_action = {
            "action_type": "update_system_prompt",
            "updated_system_prompt": new_prompt,
            "reason": "Strengthen citation instruction.",
        }

        new_def, action_type, _summary, _reason = optimizer.apply_targeted_fix(
            self.initial_def, proposed_action
        )

        self.assertEqual(action_type, "update_system_prompt")
        self.assertEqual(new_def["system_prompt"], new_prompt)
        self.assertEqual(new_def["retrieval_config"], self.initial_def["retrieval_config"])

    def test_iteration_limit_enforced(self) -> None:
        """Verify the loop strictly stops at max_iterations (default 3) when failures persist."""
        # Scorecard that consistently fails case 1
        failing_scorecard = EvaluationScorecard(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            pass_rate=0.5,
            category_scores={},
            failed_case_ids=["eval_001"],
            judgments=[
                CaseJudgment(
                    case_id="eval_001",
                    category="answerable",
                    passed=False,
                    status_match=True,
                    grounding_score=0.0,
                    factual_score=0.0,
                    citation_valid=False,
                    reason="Incomplete answer",
                ),
                CaseJudgment(
                    case_id="eval_002",
                    category="out_of_scope",
                    passed=True,
                    status_match=True,
                    grounding_score=1.0,
                    factual_score=1.0,
                    citation_valid=True,
                    reason="Refused",
                ),
            ],
        )

        # Baseline + 3 iterations = 4 scorecard checks
        judge = MockJudge([failing_scorecard, failing_scorecard, failing_scorecard, failing_scorecard])

        # Alternating distinct proposed actions so it doesn't halt due to identical definitions
        critic_responses = [
            {
                "classifications": [{"case_id": "eval_001", "category": "retrieval_miss"}],
                "proposed_action": {
                    "action_type": "update_retrieval_config",
                    "updated_retrieval_config": {"chunk_size": 180, "chunk_overlap": 30, "top_k": 4, "minimum_similarity_score": 0.0},
                },
            },
            {
                "classifications": [{"case_id": "eval_001", "category": "retrieval_miss"}],
                "proposed_action": {
                    "action_type": "update_retrieval_config",
                    "updated_retrieval_config": {"chunk_size": 180, "chunk_overlap": 30, "top_k": 5, "minimum_similarity_score": 0.0},
                },
            },
            {
                "classifications": [{"case_id": "eval_001", "category": "retrieval_miss"}],
                "proposed_action": {
                    "action_type": "update_retrieval_config",
                    "updated_retrieval_config": {"chunk_size": 180, "chunk_overlap": 30, "top_k": 6, "minimum_similarity_score": 0.0},
                },
            },
        ]

        client = MockOllamaOptimizerClient(responses=critic_responses)
        optimizer = AgentOptimizer(client=client, judge=judge, max_iterations=3)

        result: OptimizationResult = optimizer.optimize(
            self.initial_def, self.docs_path, self.cases
        )

        self.assertEqual(result.iterations_completed, 3)
        self.assertFalse(result.early_stopped)
        self.assertEqual(len(result.history), 3)

    def test_early_stopping_on_perfect_score(self) -> None:
        """Verify optimization stops early immediately when all cases pass."""
        perfect_scorecard = EvaluationScorecard(
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            pass_rate=1.0,
            category_scores={},
            failed_case_ids=[],
            judgments=[
                CaseJudgment(
                    case_id="eval_001",
                    category="answerable",
                    passed=True,
                    status_match=True,
                    grounding_score=1.0,
                    factual_score=1.0,
                    citation_valid=True,
                    reason="Passed",
                ),
                CaseJudgment(
                    case_id="eval_002",
                    category="out_of_scope",
                    passed=True,
                    status_match=True,
                    grounding_score=1.0,
                    factual_score=1.0,
                    citation_valid=True,
                    reason="Passed",
                ),
            ],
        )

        judge = MockJudge([perfect_scorecard])
        client = MockOllamaOptimizerClient()
        optimizer = AgentOptimizer(client=client, judge=judge, max_iterations=3)

        result = optimizer.optimize(self.initial_def, self.docs_path, self.cases)

        self.assertTrue(result.early_stopped)
        self.assertEqual(result.iterations_completed, 0)
        self.assertEqual(result.best_scorecard.pass_rate, 1.0)
        # Should have stopped at baseline (0 iterations, 0 critic calls, runner answered the 2 baseline queries)
        self.assertEqual(len(client.calls), 2)

    def test_regression_tracking_and_history_logging(self) -> None:
        """Verify regressions and score deltas are recorded in iteration logs."""
        # Baseline: case 1 passed, case 2 failed (50%)
        baseline_sc = EvaluationScorecard(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            pass_rate=0.5,
            category_scores={},
            failed_case_ids=["eval_002"],
            judgments=[
                CaseJudgment(case_id="eval_001", category="answerable", passed=True, status_match=True, grounding_score=1.0, factual_score=1.0, citation_valid=True, reason=""),
                CaseJudgment(case_id="eval_002", category="out_of_scope", passed=False, status_match=False, grounding_score=0.0, factual_score=0.0, citation_valid=False, reason=""),
            ],
        )

        # Iteration 1: case 1 now fails (regression!), case 2 passes (50%)
        iter1_sc = EvaluationScorecard(
            total_cases=2,
            passed_cases=1,
            failed_cases=1,
            pass_rate=0.5,
            category_scores={},
            failed_case_ids=["eval_001"],
            judgments=[
                CaseJudgment(case_id="eval_001", category="answerable", passed=False, status_match=False, grounding_score=0.0, factual_score=0.0, citation_valid=False, reason=""),
                CaseJudgment(case_id="eval_002", category="out_of_scope", passed=True, status_match=True, grounding_score=1.0, factual_score=1.0, citation_valid=True, reason=""),
            ],
        )

        # Iteration 2: both pass (100%)
        iter2_sc = EvaluationScorecard(
            total_cases=2,
            passed_cases=2,
            failed_cases=0,
            pass_rate=1.0,
            category_scores={},
            failed_case_ids=[],
            judgments=[
                CaseJudgment(case_id="eval_001", category="answerable", passed=True, status_match=True, grounding_score=1.0, factual_score=1.0, citation_valid=True, reason=""),
                CaseJudgment(case_id="eval_002", category="out_of_scope", passed=True, status_match=True, grounding_score=1.0, factual_score=1.0, citation_valid=True, reason=""),
            ],
        )

        judge = MockJudge([baseline_sc, iter1_sc, iter2_sc])
        critic_responses = [
            {
                "classifications": [{"case_id": "eval_002", "category": "prompt_defect"}],
                "proposed_action": {
                    "action_type": "update_system_prompt",
                    "updated_system_prompt": self.initial_def["system_prompt"] + "\nRefined.",
                },
            },
            {
                "classifications": [{"case_id": "eval_001", "category": "retrieval_miss"}],
                "proposed_action": {
                    "action_type": "update_retrieval_config",
                    "updated_retrieval_config": {"chunk_size": 180, "chunk_overlap": 30, "top_k": 5, "minimum_similarity_score": 0.0},
                },
            },
        ]

        client = MockOllamaOptimizerClient(responses=critic_responses)
        optimizer = AgentOptimizer(client=client, judge=judge, max_iterations=3)

        result = optimizer.optimize(self.initial_def, self.docs_path, self.cases)

        self.assertEqual(len(result.history), 2)
        # Iteration 1 had eval_001 regress
        self.assertEqual(result.history[0].regressions, ["eval_001"])
        self.assertEqual(result.history[0].before_pass_rate, 0.5)
        self.assertEqual(result.history[0].after_pass_rate, 0.5)

        # Iteration 2 reached 100% and halted early
        self.assertEqual(result.history[1].after_pass_rate, 1.0)
        self.assertTrue(result.early_stopped)
        self.assertEqual(result.best_scorecard.pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
