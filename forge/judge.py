"""Automated Judge module for ForgeAI.

Scores agent execution results against benchmark evaluation cases for grounding,
factual correctness, and refusal/clarification compliance using combined
deterministic verification and LLM-as-a-judge semantic assessment.

Bias Mitigations:
1. Position Bias: Evaluates cases individually using independent criteria-based
   rubrics rather than relative or pairwise rankings where order influences preference.
2. Verbosity Bias: Explicitly checks presence and accuracy of defined key facts;
   flowery, verbose explanations lacking expected facts are penalized.
3. Self-Preference: Candidate responses are anonymized (agent persona and model
   identity are withheld from the prompt) and anchored against deterministic
   document retrieval evidence.

Documented Remaining Limitations:
- Single-judge calibration: Smaller local models (e.g. 3B parameters) may exhibit
  occasional boundary variance on nuanced semantic edge cases compared to large models.
- Subtle paraphrase drift: Paraphrased technical facts may require careful prompt
  guidance to avoid false negative factual penalties.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from forge.eval_generator import EvaluationCase
from forge.llm import OllamaClient
from forge.runner import AgentExecutionResult


DEFAULT_JUDGE_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "judge.txt"
)


def load_judge_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    """Load judge prompt template from file."""
    path = Path(prompt_path) if prompt_path else DEFAULT_JUDGE_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Judge prompt file not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for line in content.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


@dataclass
class CaseJudgment:
    """Detailed evaluation score and rationale for a single evaluation case."""

    case_id: str
    category: str
    passed: bool
    status_match: bool
    grounding_score: float
    factual_score: float
    citation_valid: bool
    reason: str
    evidence_used: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert judgment to dictionary."""
        return asdict(self)


@dataclass
class EvaluationScorecard:
    """Aggregate benchmark scorecard across all evaluated cases."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    category_scores: Dict[str, Dict[str, Any]]
    failed_case_ids: List[str]
    judgments: List[CaseJudgment]

    def to_dict(self) -> Dict[str, Any]:
        """Convert scorecard to dictionary."""
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "pass_rate": round(self.pass_rate, 4),
            "category_scores": self.category_scores,
            "failed_case_ids": self.failed_case_ids,
            "judgments": [j.to_dict() for j in self.judgments],
        }


class AutomatedJudge:
    """Automated benchmark evaluation judge for ForgeAI."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        passing_threshold: float = 0.7,
    ) -> None:
        self.client = client or OllamaClient()
        self.prompt_template = load_judge_prompt(prompt_path)
        self.passing_threshold = passing_threshold

    def judge_case(
        self,
        case: EvaluationCase,
        result: AgentExecutionResult,
    ) -> CaseJudgment:
        """Evaluate a single agent execution result against an evaluation case.

        Performs deterministic checks first, followed by at most one LLM judge call.
        """
        # 1. Deterministic Status Validation
        status_match = (result.status == case.expected_status)
        if not status_match:
            return CaseJudgment(
                case_id=case.id,
                category=case.category,
                passed=False,
                status_match=False,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=False,
                reason=(
                    f"Status mismatch: expected '{case.expected_status}', "
                    f"got '{result.status}'."
                ),
                evidence_used=result.retrieved_evidence,
            )

        # 2. Deterministic Execution Errors Validation
        if result.execution_errors:
            return CaseJudgment(
                case_id=case.id,
                category=case.category,
                passed=False,
                status_match=True,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=False,
                reason=(
                    f"Execution errors occurred during generation: "
                    f"{'; '.join(result.execution_errors)}"
                ),
                evidence_used=result.retrieved_evidence,
            )

        # 3. Deterministic Citation Validation
        citation_valid = True
        if case.category in ("answerable", "cross_document"):
            if not result.citations:
                return CaseJudgment(
                    case_id=case.id,
                    category=case.category,
                    passed=False,
                    status_match=True,
                    grounding_score=0.0,
                    factual_score=0.0,
                    citation_valid=False,
                    reason=f"Category '{case.category}' requires valid supporting citations, but none were returned.",
                    evidence_used=result.retrieved_evidence,
                )

            if case.category == "cross_document":
                cited_sources = {
                    str(c.get("source", "")).strip().replace("\\", "/")
                    for c in result.citations
                    if c.get("source")
                }
                if len(cited_sources) < 2:
                    return CaseJudgment(
                        case_id=case.id,
                        category=case.category,
                        passed=False,
                        status_match=True,
                        grounding_score=0.5,
                        factual_score=0.5,
                        citation_valid=False,
                        reason=(
                            f"Cross-document case requires citations from at least 2 distinct files, "
                            f"got {len(cited_sources)}: {sorted(cited_sources)}"
                        ),
                        evidence_used=result.retrieved_evidence,
                    )

        # 4. LLM-as-a-Judge Evaluation (At most one call per case)
        # Format evidence summary
        evidence_snippets: List[str] = []
        for idx, ev in enumerate(result.retrieved_evidence[:4]):
            src = ev.get("metadata", {}).get("source") or ev.get("metadata", {}).get("filename", f"doc_{idx}")
            txt = ev.get("text", "").strip()[:300]
            evidence_snippets.append(f"[{src}]: {txt}")

        evidence_str = "\n".join(evidence_snippets) if evidence_snippets else "No retrieved document evidence."
        expected_facts_str = "\n".join(f"- {f}" for f in case.expected_key_facts) if case.expected_key_facts else "None specified."

        prompt = (
            self.prompt_template.replace("{question}", case.question)
            .replace("{expected_status}", case.expected_status)
            .replace("{candidate_status}", result.status)
            .replace("{candidate_answer}", result.answer.strip())
            .replace("{expected_key_facts}", expected_facts_str)
            .replace("{retrieved_evidence}", evidence_str)
        )

        system_prompt = (
            "You are a strict, objective AI benchmark judge. "
            "Score candidate responses strictly against expected facts and evidence. "
            "Output ONLY valid JSON adhering to the schema."
        )

        try:
            res = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:
            return CaseJudgment(
                case_id=case.id,
                category=case.category,
                passed=False,
                status_match=True,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=citation_valid,
                reason=f"Judge LLM evaluation failed: {exc}",
                evidence_used=result.retrieved_evidence,
            )

        if not isinstance(res, dict):
            return CaseJudgment(
                case_id=case.id,
                category=case.category,
                passed=False,
                status_match=True,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=citation_valid,
                reason="Malformed judge response: output was not a valid JSON dictionary.",
                evidence_used=result.retrieved_evidence,
            )

        try:
            factual_score = float(res.get("factual_score", 0.0))
            grounding_score = float(res.get("grounding_score", 0.0))
            llm_passed = bool(res.get("passed", False))
            reason = str(res.get("reason", "")).strip() or "Evaluated by judge."
        except (ValueError, TypeError) as conv_err:
            return CaseJudgment(
                case_id=case.id,
                category=case.category,
                passed=False,
                status_match=True,
                grounding_score=0.0,
                factual_score=0.0,
                citation_valid=citation_valid,
                reason=f"Malformed judge response numerical fields: {conv_err}",
                evidence_used=result.retrieved_evidence,
            )

        overall_passed = (
            status_match
            and citation_valid
            and llm_passed
            and factual_score >= self.passing_threshold
            and grounding_score >= self.passing_threshold
        )

        return CaseJudgment(
            case_id=case.id,
            category=case.category,
            passed=overall_passed,
            status_match=status_match,
            grounding_score=grounding_score,
            factual_score=factual_score,
            citation_valid=citation_valid,
            reason=reason,
            evidence_used=result.retrieved_evidence,
        )

    def judge_evaluation_set(
        self,
        cases: Sequence[EvaluationCase],
        results: Sequence[AgentExecutionResult],
    ) -> EvaluationScorecard:
        """Evaluate a sequence of cases and results to produce an aggregate scorecard."""
        if len(cases) != len(results):
            raise ValueError(
                f"Number of evaluation cases ({len(cases)}) does not match "
                f"number of execution results ({len(results)})."
            )

        judgments: List[CaseJudgment] = []
        category_stats: Dict[str, Dict[str, int]] = {}
        failed_ids: List[str] = []

        for case, res in zip(cases, results):
            j = self.judge_case(case, res)
            judgments.append(j)

            cat = case.category
            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "passed": 0}
            category_stats[cat]["total"] += 1

            if j.passed:
                category_stats[cat]["passed"] += 1
            else:
                failed_ids.append(case.id)

        total = len(cases)
        passed = sum(1 for j in judgments if j.passed)
        failed = total - passed
        pass_rate = (passed / total) if total > 0 else 0.0

        category_scores: Dict[str, Dict[str, Any]] = {}
        for cat, stats in category_stats.items():
            c_total = stats["total"]
            c_passed = stats["passed"]
            category_scores[cat] = {
                "total": c_total,
                "passed": c_passed,
                "pass_rate": round(c_passed / c_total, 4) if c_total > 0 else 0.0,
            }

        return EvaluationScorecard(
            total_cases=total,
            passed_cases=passed,
            failed_cases=failed,
            pass_rate=pass_rate,
            category_scores=category_scores,
            failed_case_ids=failed_ids,
            judgments=judgments,
        )
