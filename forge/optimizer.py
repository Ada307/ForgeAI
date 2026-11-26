"""Critic-Optimizer Loop for ForgeAI.

Diagnoses benchmark evaluation failures into OceanAI root-cause categories
(retrieval_miss, prompt_defect, hallucination, wrong_refusal), applies targeted
configuration or prompt adjustments without full regeneration, and iteratively
improves agent performance across bounded iterations.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from forge.eval_generator import EvaluationCase
from forge.generator import validate_agent_definition
from forge.judge import AutomatedJudge, CaseJudgment, EvaluationScorecard
from forge.llm import OllamaClient
from forge.retrieval import RetrieverConfig
from forge.runner import AgentExecutionResult, AgentRunner


DEFAULT_CRITIC_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "critic.txt"
)

VALID_FAILURE_CATEGORIES = {
    "retrieval_miss",
    "prompt_defect",
    "hallucination",
    "wrong_refusal",
}


def load_critic_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    """Load critic prompt template from file."""
    path = Path(prompt_path) if prompt_path else DEFAULT_CRITIC_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Critic prompt file not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for line in content.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@dataclass
class OptimizationIterationLog:
    """Audit log of a single optimization iteration."""

    iteration: int
    action_type: str
    change_summary: str
    reason: str
    addressed_case_ids: List[str]
    failure_categories: Dict[str, int]
    before_pass_rate: float
    after_pass_rate: float
    regressions: List[str]
    scorecard: EvaluationScorecard

    def to_dict(self) -> Dict[str, Any]:
        """Convert iteration log to dictionary."""
        d = asdict(self)
        d["scorecard"] = self.scorecard.to_dict()
        return d


@dataclass
class OptimizationResult:
    """Complete summary of the optimization loop execution."""

    best_agent_definition: Dict[str, Any]
    best_scorecard: EvaluationScorecard
    baseline_scorecard: EvaluationScorecard
    iterations_completed: int
    early_stopped: bool
    early_stop_reason: Optional[str]
    history: List[OptimizationIterationLog]

    def to_dict(self) -> Dict[str, Any]:
        """Convert optimization result to dictionary."""
        return {
            "best_agent_definition": self.best_agent_definition,
            "best_scorecard": self.best_scorecard.to_dict(),
            "baseline_scorecard": self.baseline_scorecard.to_dict(),
            "iterations_completed": self.iterations_completed,
            "early_stopped": self.early_stopped,
            "early_stop_reason": self.early_stop_reason,
            "history": [h.to_dict() for h in self.history],
        }


class AgentOptimizer:
    """Orchestrates the critic-optimizer loop on generated agents."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        judge: Optional[AutomatedJudge] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        max_iterations: int = 3,
        batch_size: int = 5,
    ) -> None:
        self.client = client or OllamaClient()
        self.judge = judge or AutomatedJudge(client=self.client)
        self.prompt_template = load_critic_prompt(prompt_path)
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def evaluate(
        self,
        runner: AgentRunner,
        cases: Sequence[EvaluationCase],
    ) -> Tuple[List[AgentExecutionResult], EvaluationScorecard]:
        """Execute all evaluation cases against the runner and score with judge."""
        results: List[AgentExecutionResult] = []
        for case in cases:
            res = runner.answer(case.question)
            results.append(res)

        scorecard = self.judge.judge_evaluation_set(cases, results)
        return results, scorecard

    def classify_and_diagnose(
        self,
        failed_cases: Sequence[EvaluationCase],
        failed_results: Sequence[AgentExecutionResult],
        failed_judgments: Sequence[CaseJudgment],
        current_definition: Dict[str, Any],
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Diagnose failures into OceanAI categories in bounded batches and propose targeted fix."""
        all_classifications: List[Dict[str, str]] = []
        proposed_action: Dict[str, Any] = {}

        system_prompt = current_definition.get("system_prompt", "")
        retrieval_config_str = json.dumps(current_definition.get("retrieval_config", {}), indent=2)

        # Batch failed cases to respect context boundaries and limit LLM calls
        for i in range(0, len(failed_cases), self.batch_size):
            batch_cases = failed_cases[i : i + self.batch_size]
            batch_results = failed_results[i : i + self.batch_size]
            batch_judgments = failed_judgments[i : i + self.batch_size]

            snippets: List[str] = []
            for case, res, j in zip(batch_cases, batch_results, batch_judgments):
                ev_count = len(res.retrieved_evidence)
                snippet = (
                    f"--- Case ID: {case.id} (Category: {case.category}) ---\n"
                    f"Question: {case.question}\n"
                    f"Expected Status: {case.expected_status}\n"
                    f"Candidate Status: {res.status}\n"
                    f"Candidate Answer: {res.answer[:250]}\n"
                    f"Retrieved Chunks Count: {ev_count}\n"
                    f"Judge Finding: {j.reason}\n"
                )
                snippets.append(snippet)

            batch_context = "\n".join(snippets)

            prompt = (
                self.prompt_template.replace("{system_prompt}", system_prompt)
                .replace("{retrieval_config}", retrieval_config_str)
                .replace("{failed_cases_context}", batch_context)
            )

            critic_sys = (
                "You are an expert diagnostic critic for an AI retrieval agent. "
                "Classify every failed case into exactly one category: "
                "retrieval_miss | prompt_defect | hallucination | wrong_refusal. "
                "Output ONLY valid JSON adhering strictly to the schema."
            )

            try:
                res_json = self.client.generate_json(prompt, system_prompt=critic_sys)
            except Exception:
                res_json = {}

            # Parse classifications
            raw_cls = res_json.get("classifications", []) if isinstance(res_json, dict) else []
            case_cls_map = {}
            if isinstance(raw_cls, list):
                for item in raw_cls:
                    if isinstance(item, dict) and "case_id" in item:
                        cat = str(item.get("category", "")).strip().lower()
                        if cat in VALID_FAILURE_CATEGORIES:
                            case_cls_map[item["case_id"]] = {
                                "case_id": item["case_id"],
                                "category": cat,
                                "diagnosis": str(item.get("diagnosis", "")),
                            }

            # Fallback for any unclassified or misclassified case in batch
            for case, res, j in zip(batch_cases, batch_results, batch_judgments):
                if case.id in case_cls_map:
                    all_classifications.append(case_cls_map[case.id])
                else:
                    # Deterministic fallback classification based on symptoms
                    fallback_cat = self._deterministic_fallback_classification(case, res, j)
                    all_classifications.append(
                        {
                            "case_id": case.id,
                            "category": fallback_cat,
                            "diagnosis": j.reason,
                        }
                    )

            # Keep last or most specific proposed action
            if isinstance(res_json, dict) and isinstance(res_json.get("proposed_action"), dict):
                proposed_action = res_json["proposed_action"]

        # Default fallback proposed action if LLM provided none
        if not proposed_action or not proposed_action.get("action_type"):
            proposed_action = self._default_fallback_action(all_classifications, current_definition)

        return all_classifications, proposed_action

    def _deterministic_fallback_classification(
        self,
        case: EvaluationCase,
        result: AgentExecutionResult,
        judgment: CaseJudgment,
    ) -> str:
        """Deterministic fallback classification when LLM response is malformed."""
        if not result.retrieved_evidence or "citation" in judgment.reason.lower() and not result.citations:
            return "retrieval_miss"
        if case.expected_status == "answered" and result.status == "refused":
            return "wrong_refusal"
        if "fabricated" in judgment.reason.lower() or not judgment.grounding_score:
            return "hallucination"
        return "prompt_defect"

    def _default_fallback_action(
        self,
        classifications: List[Dict[str, str]],
        current_def: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Propose a safe targeted action if LLM critic failed to generate one."""
        cat_counts: Dict[str, int] = {}
        for c in classifications:
            cat = c.get("category", "prompt_defect")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "prompt_defect"

        if top_cat == "retrieval_miss":
            cfg = dict(current_def.get("retrieval_config", {}))
            cfg["top_k"] = min(cfg.get("top_k", 3) + 2, 8)
            cfg["minimum_similarity_score"] = 0.0
            return {
                "action_type": "update_retrieval_config",
                "updated_retrieval_config": cfg,
                "reason": "Increased top_k to capture missing evidence chunks.",
            }
        elif top_cat == "hallucination":
            prompt = current_def.get("system_prompt", "")
            refined = prompt + "\nStrict Directive: Do not speculate or introduce unverified facts. Rely strictly on retrieved chunks."
            return {
                "action_type": "update_system_prompt",
                "updated_system_prompt": refined,
                "reason": "Strengthened strict grounding directives to prevent hallucination.",
            }
        elif top_cat == "wrong_refusal":
            prompt = current_def.get("system_prompt", "")
            refined = prompt + "\nClarification: Answer all permitted in-scope questions if any retrieved evidence is present."
            return {
                "action_type": "update_system_prompt",
                "updated_system_prompt": refined,
                "reason": "Clarified permitted answer boundaries to prevent unnecessary refusals.",
            }
        else:
            prompt = current_def.get("system_prompt", "")
            refined = prompt + "\nDirective: Ensure responses are direct, factual, and strictly cite verified sources."
            return {
                "action_type": "update_system_prompt",
                "updated_system_prompt": refined,
                "reason": "Clarified system prompt formatting and citation instructions.",
            }

    def apply_targeted_fix(
        self,
        current_definition: Dict[str, Any],
        proposed_action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, str, str]:
        """Apply targeted modifications and validate against agent definition schema."""
        action_type = str(proposed_action.get("action_type", ""))
        reason = str(proposed_action.get("reason", "Applied targeted fix."))

        new_def = json.loads(json.dumps(current_definition))  # deep copy

        if action_type == "update_retrieval_config":
            new_cfg = proposed_action.get("updated_retrieval_config", {})
            # Validate with RetrieverConfig
            validated_cfg = RetrieverConfig.from_dict(new_cfg)
            new_def["retrieval_config"] = validated_cfg.to_dict()
            change_summary = (
                f"Updated retrieval config: chunk_size={validated_cfg.chunk_size}, "
                f"chunk_overlap={validated_cfg.chunk_overlap}, top_k={validated_cfg.top_k}, "
                f"min_score={validated_cfg.minimum_similarity_score}"
            )

        elif action_type == "update_system_prompt":
            new_prompt = str(proposed_action.get("updated_system_prompt", "")).strip()
            if not new_prompt:
                raise ValueError("Proposed updated_system_prompt is empty.")
            new_def["system_prompt"] = new_prompt
            change_summary = "Updated system prompt with targeted directives."

        else:
            raise ValueError(f"Unsupported action_type: '{action_type}'")

        # Re-validate complete definition
        validate_agent_definition(new_def)

        return new_def, action_type, change_summary, reason

    def optimize(
        self,
        agent_definition: Dict[str, Any],
        document_folder: Union[str, Path],
        evaluation_cases: Sequence[EvaluationCase],
    ) -> OptimizationResult:
        """Execute the complete critic-optimizer loop up to max_iterations."""
        current_def = json.loads(json.dumps(agent_definition))
        validate_agent_definition(current_def)

        # Baseline evaluation (Iteration 0)
        runner = AgentRunner(
            agent_definition=current_def,
            document_folder=document_folder,
            client=self.client,
        )
        results, baseline_scorecard = self.evaluate(runner, evaluation_cases)

        best_def = current_def
        best_scorecard = baseline_scorecard
        history: List[OptimizationIterationLog] = []

        passed_case_ids_prev: Set[str] = {
            j.case_id for j in baseline_scorecard.judgments if j.passed
        }

        # Early exit if baseline is already perfect
        if baseline_scorecard.pass_rate >= 1.0:
            return OptimizationResult(
                best_agent_definition=best_def,
                best_scorecard=best_scorecard,
                baseline_scorecard=baseline_scorecard,
                iterations_completed=0,
                early_stopped=True,
                early_stop_reason="All evaluation cases passed on baseline evaluation.",
                history=history,
            )

        early_stopped = False
        early_stop_reason = None

        for iteration in range(1, self.max_iterations + 1):
            # Identify failed cases
            failed_cases: List[EvaluationCase] = []
            failed_results: List[AgentExecutionResult] = []
            failed_judgments: List[CaseJudgment] = []

            for case, res, j in zip(evaluation_cases, results, best_scorecard.judgments):
                if not j.passed:
                    failed_cases.append(case)
                    failed_results.append(res)
                    failed_judgments.append(j)

            if not failed_cases:
                early_stopped = True
                early_stop_reason = "All evaluation cases passed."
                break

            # Diagnose failures
            classifications, proposed_action = self.classify_and_diagnose(
                failed_cases=failed_cases,
                failed_results=failed_results,
                failed_judgments=failed_judgments,
                current_definition=current_def,
            )

            cat_counts: Dict[str, int] = {}
            for c in classifications:
                cat = c["category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

            # Apply targeted fix
            try:
                new_def, action_type, change_summary, reason = self.apply_targeted_fix(
                    current_def, proposed_action
                )
            except Exception as e:
                early_stopped = True
                early_stop_reason = f"Targeted fix could not be applied or validated: {e}"
                break

            # If no actual change was made, halt early
            if new_def == current_def:
                early_stopped = True
                early_stop_reason = "No valid targeted fix remains; proposed definition was identical."
                break

            # Rebuild runner and re-evaluate
            current_def = new_def
            runner = AgentRunner(
                agent_definition=current_def,
                document_folder=document_folder,
                client=self.client,
            )
            results, new_scorecard = self.evaluate(runner, evaluation_cases)

            # Compute regressions (cases passed in previous iteration but failed in this iteration)
            current_passed_ids = {j.case_id for j in new_scorecard.judgments if j.passed}
            regressions = sorted(passed_case_ids_prev - current_passed_ids)

            log_entry = OptimizationIterationLog(
                iteration=iteration,
                action_type=action_type,
                change_summary=change_summary,
                reason=reason,
                addressed_case_ids=[c.id for c in failed_cases],
                failure_categories=cat_counts,
                before_pass_rate=best_scorecard.pass_rate,
                after_pass_rate=new_scorecard.pass_rate,
                regressions=regressions,
                scorecard=new_scorecard,
            )
            history.append(log_entry)

            # Retain best evaluated definition
            if new_scorecard.pass_rate > best_scorecard.pass_rate:
                best_scorecard = new_scorecard
                best_def = current_def

            passed_case_ids_prev = current_passed_ids

            if new_scorecard.pass_rate >= 1.0:
                early_stopped = True
                early_stop_reason = f"All evaluation cases passed at iteration {iteration}."
                break

        return OptimizationResult(
            best_agent_definition=best_def,
            best_scorecard=best_scorecard,
            baseline_scorecard=baseline_scorecard,
            iterations_completed=len(history),
            early_stopped=early_stopped,
            early_stop_reason=early_stop_reason,
            history=history,
        )
