"""Honest Reporting and Artifact Generation module for ForgeAI.

Produces transparent, comprehensive markdown reports (final_report.md) and
persists structured stage artifacts across all pipeline phases.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from forge.documents import Document, SkippedFile
from forge.eval_generator import EvaluationCase
from forge.optimizer import OptimizationResult


def generate_markdown_report(
    status: str,
    brief_text: str,
    documents: Sequence[Document],
    skipped_files: Sequence[SkippedFile],
    requirements: Optional[Dict[str, Any]] = None,
    initial_definition: Optional[Dict[str, Any]] = None,
    final_definition: Optional[Dict[str, Any]] = None,
    evaluation_cases: Optional[Sequence[EvaluationCase]] = None,
    optimization_result: Optional[OptimizationResult] = None,
    error_message: Optional[str] = None,
) -> str:
    """Generate comprehensive, honest markdown report for the factory run."""
    lines: List[str] = []

    # Title & Executive Header
    lines.append("# ForgeAI Meta-Agent Factory Run Report\n")
    lines.append("## 1. Executive Summary")

    is_success = (status.upper() == "SUCCESS")
    status_badge = "✅ COMPLETED_SUCCESSFULLY" if is_success else f"⚠️ INCOMPLETE / FAILED"
    lines.append(f"- **Pipeline Status**: {status_badge}")
    if error_message:
        lines.append(f"- **Termination Error**: `{error_message}`")

    lines.append(f"- **Ingested Documents Count**: {len(documents)}")
    lines.append(f"- **Skipped Files Count**: {len(skipped_files)}")

    if optimization_result:
        b_rate = optimization_result.baseline_scorecard.pass_rate * 100.0
        f_rate = optimization_result.best_scorecard.pass_rate * 100.0
        lines.append(f"- **Baseline Benchmark Pass Rate**: {b_rate:.1f}%")
        lines.append(f"- **Final Benchmark Pass Rate**: {f_rate:.1f}%")
        lines.append(f"- **Optimization Iterations Completed**: {optimization_result.iterations_completed}")
        if optimization_result.early_stopped:
            lines.append(f"- **Early Termination Reason**: {optimization_result.early_stop_reason}")
    else:
        lines.append("- **Benchmark Evaluation**: Not reached or incomplete")

    lines.append("")

    # 2. Requester Brief & Document Ingestion
    lines.append("## 2. Ingested Documents & Skipped Files")
    lines.append("### Requester Brief Preview")
    brief_preview = brief_text.strip()[:400] + ("..." if len(brief_text.strip()) > 400 else "")
    lines.append(f"> {brief_preview}\n")

    lines.append("### Usable Documents")
    if documents:
        lines.append("| Document ID | Source Path | Characters | Chunks |")
        lines.append("|---|---|---|---|")
        for doc in documents:
            src = str(doc.metadata.get("source", doc.doc_id))
            lines.append(f"| `{doc.doc_id}` | `{src}` | {len(doc.text)} | {doc.metadata.get('chunk_index', 1)} |")
    else:
        lines.append("No usable documents were ingested.")

    lines.append("")
    lines.append("### Skipped Files")
    if skipped_files:
        lines.append("| File Path | Skip Reason |")
        lines.append("|---|---|")
        for sk in skipped_files:
            lines.append(f"| `{sk.path}` | {sk.reason} |")
    else:
        lines.append("None. All discovered files were successfully processed.")

    lines.append("")

    # 3. Intake Scope & Groundability Guard
    lines.append("## 3. Intake Requirements & Groundability Guard")
    if requirements:
        lines.append(f"- **Scope**: {requirements.get('scope', 'N/A')}")
        lines.append(f"- **Audience**: {requirements.get('audience', 'N/A')}")
        lines.append(f"- **Tone**: {requirements.get('tone', 'N/A')}")

        lines.append("\n### Permitted Must-Answer Topics (Supported by Evidence)")
        for t in requirements.get("must_answer_topics", []):
            lines.append(f"- {t}")

        lines.append("\n### Must-Refuse Topics (Unsupported Knowledge Gaps & Out of Scope)")
        for t in requirements.get("must_refuse_topics", []):
            lines.append(f"- {t}")

        assumptions = requirements.get("assumptions", [])
        if assumptions:
            lines.append("\n### Explicit Unattended Assumptions")
            for a in assumptions:
                lines.append(f"- {a}")
    else:
        lines.append("Intake requirements stage not reached.")

    lines.append("")

    # 4. Agent Architecture & Selected Tools
    lines.append("## 4. Generated Agent Architecture")
    active_def = final_definition or initial_definition
    if active_def:
        lines.append(f"- **Agent Name**: `{active_def.get('agent_name')}`")
        lines.append(f"- **Description**: {active_def.get('description')}")

        tools = active_def.get("tools", [])
        lines.append(f"- **Attached Tools Count**: {len(tools)}")
        if tools:
            for t in tools:
                lines.append(f"  - `{t.get('name')}`: {t.get('justification', 'Authorized')}")
        else:
            lines.append("  *(No tools attached; purely document evidence retrieval)*")

        rc = active_def.get("retrieval_config", {})
        lines.append(
            f"- **Retrieval Parameters**: chunk_size={rc.get('chunk_size')}, "
            f"chunk_overlap={rc.get('chunk_overlap')}, top_k={rc.get('top_k')}, "
            f"min_similarity={rc.get('minimum_similarity_score')}"
        )

        lines.append("\n### System Prompt Preview")
        sp = active_def.get("system_prompt", "").strip()
        lines.append(f"```text\n{sp[:600]}...\n```")
    else:
        lines.append("Agent generation stage not reached.")

    lines.append("")

    # 5. Evaluation Benchmark Distribution
    lines.append("## 5. Evaluation Benchmark Set")
    if evaluation_cases:
        lines.append(f"- **Total Test Cases**: {len(evaluation_cases)}")
        cat_counts: Dict[str, int] = {}
        for c in evaluation_cases:
            cat_counts[c.category] = cat_counts.get(c.category, 0) + 1

        lines.append("| Category | Target | Generated Count |")
        lines.append("|---|---|---|")
        for cat, cnt in sorted(cat_counts.items()):
            lines.append(f"| `{cat}` | {cnt} | {cnt} |")
    else:
        lines.append("Evaluation benchmark generation stage not reached.")

    lines.append("")

    # 6. Optimization History & Critic Insights
    lines.append("## 6. Critic–Optimizer Loop Performance")
    if optimization_result:
        lines.append("### Scorecard Progress by Iteration")
        lines.append("| Phase | Pass Rate | Passed Cases | Failed Cases | Action Applied |")
        lines.append("|---|---|---|---|---|")
        b_sc = optimization_result.baseline_scorecard
        lines.append(f"| **Baseline (0)** | {b_sc.pass_rate * 100:.1f}% | {b_sc.passed_cases}/{b_sc.total_cases} | {b_sc.failed_cases} | Initial Agent Definition |")

        for log in optimization_result.history:
            lines.append(
                f"| **Iteration {log.iteration}** | {log.after_pass_rate * 100:.1f}% | "
                f"{log.scorecard.passed_cases}/{log.scorecard.total_cases} | "
                f"{log.scorecard.failed_cases} | `{log.action_type}`: {log.change_summary} |"
            )

        lines.append("\n### Critic Diagnostic Classifications Across Iterations")
        lines.append("| Iteration | Diagnosed Failure Categories | Regressions | Reason |")
        lines.append("|---|---|---|---|")
        for log in optimization_result.history:
            cats = ", ".join(f"{k}: {v}" for k, v in log.failure_categories.items()) or "None"
            regs = ", ".join(log.regressions) if log.regressions else "0"
            lines.append(f"| {log.iteration} | {cats} | {regs} | {log.reason} |")

        # 7. Unresolved Failures
        lines.append("\n## 7. Unresolved Benchmark Failures")
        best_sc = optimization_result.best_scorecard
        if best_sc.failed_case_ids:
            lines.append(f"Total unresolved failures: **{len(best_sc.failed_case_ids)}**\n")
            lines.append("| Case ID | Category | Failure Reason |")
            lines.append("|---|---|---|")
            judg_map = {j.case_id: j for j in best_sc.judgments}
            for fid in best_sc.failed_case_ids:
                j = judg_map.get(fid)
                r_txt = j.reason if j else "Failed evaluation"
                c_cat = j.category if j else "unknown"
                lines.append(f"| `{fid}` | `{c_cat}` | {r_txt} |")
        else:
            lines.append("🎉 **Zero unresolved failures.** All benchmark evaluation cases passed.")
    else:
        lines.append("Optimization stage not reached.")

    lines.append("")

    # 8. Known Limitations
    lines.append("## 8. System Limitations & Boundaries")
    lines.append("- **Local Context Bounds**: Operating within local 4096-token GPU context; retrieval chunks and previews are strictly bounded.")
    lines.append("- **Single-Judge Rubric Variance**: Local 3B judge models may exhibit subtle calibration drift on nuanced paraphrasing.")
    lines.append("- **Optimization Horizon**: Bounded to maximum 3 iterations per factory run to preserve compute.")
    lines.append("- **Grounding Anchor**: Unsupported topics remain strictly routed to must-refuse topics; never relaxed.")

    return "\n".join(lines).strip() + "\n"


def save_stage_artifacts(
    output_dir: Union[str, Path],
    requirements: Optional[Dict[str, Any]] = None,
    initial_definition: Optional[Dict[str, Any]] = None,
    final_definition: Optional[Dict[str, Any]] = None,
    evaluation_cases: Optional[Sequence[EvaluationCase]] = None,
    optimization_result: Optional[OptimizationResult] = None,
    evaluation_results: Optional[List[Dict[str, Any]]] = None,
    final_report_md: Optional[str] = None,
) -> Dict[str, Path]:
    """Persist all generated artifacts to the output directory immediately."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    def _write_json(filename: str, data: Any) -> Path:
        p = out / filename
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        saved[filename] = p
        return p

    if requirements is not None:
        _write_json("requirements.json", requirements)

    if initial_definition is not None:
        _write_json("initial_agent_definition.json", initial_definition)

    # agent_definition.json is the active/best definition for chat.py
    active_def = final_definition or initial_definition
    if active_def is not None:
        _write_json("agent_definition.json", active_def)

    if evaluation_cases is not None:
        eval_payload = {
            "total_cases": len(evaluation_cases),
            "cases": [c.to_dict() for c in evaluation_cases],
        }
        _write_json("eval_set.json", eval_payload)

    if optimization_result is not None:
        # iteration_logs.json
        iter_logs = [log.to_dict() for log in optimization_result.history]
        _write_json("iteration_logs.json", iter_logs)

        # scorecards.json
        scorecards_payload = {
            "baseline": optimization_result.baseline_scorecard.to_dict(),
            "best": optimization_result.best_scorecard.to_dict(),
            "iterations": [log.scorecard.to_dict() for log in optimization_result.history],
        }
        _write_json("scorecards.json", scorecards_payload)

    if evaluation_results is not None:
        _write_json("eval_results.json", evaluation_results)

    if final_report_md is not None:
        p = out / "final_report.md"
        p.write_text(final_report_md, encoding="utf-8")
        saved["final_report.md"] = p

    return saved
