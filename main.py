"""ForgeAI Meta-Agent Factory CLI Entrypoint and Orchestration Pipeline.

Orchestrates the complete factory workflow:
Document Ingestion -> Intake Interview -> Groundability Guard ->
Agent Generation -> Evaluation Generation -> Baseline Evaluation ->
Critic-Optimizer Loop -> Honest Final Reporting.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

from forge.documents import DocumentLoader
from forge.eval_generator import EvaluationSetGenerator
from forge.generator import AgentGenerator
from forge.groundability import GroundabilityGuard
from forge.intake import IntakeInterviewer
from forge.judge import AutomatedJudge
from forge.llm import OllamaClient
from forge.optimizer import AgentOptimizer, OptimizationResult
from forge.reporting import generate_markdown_report, save_stage_artifacts
from forge.retrieval import TFIDFRetriever
from forge.tools import ToolRegistry


def run_factory(
    brief_path: Union[str, Path],
    docs_path: Union[str, Path],
    output_path: Union[str, Path],
    max_iterations: int = 3,
    interactive: bool = False,
    client: Optional[OllamaClient] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end ForgeAI meta-agent factory pipeline."""
    brief_file = Path(brief_path)
    docs_folder = Path(docs_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = client or OllamaClient()

    # Track intermediate objects for honest progressive artifact persistence
    brief_text = ""
    documents = []
    skipped_files = []
    requirements = None
    initial_definition = None
    final_definition = None
    evaluation_cases = None
    optimization_result = None
    eval_results = []

    try:
        if not brief_file.exists():
            raise FileNotFoundError(f"Requester brief file not found: '{brief_path}'")
        if not docs_folder.exists() or not docs_folder.is_dir():
            raise FileNotFoundError(f"Document folder not found or not a directory: '{docs_path}'")

        brief_text = brief_file.read_text(encoding="utf-8").strip()
        # =====================================================================
        # Stage 1: Document Ingestion
        # =====================================================================
        print(f"\n[Stage 1/7] Ingesting documents from '{docs_folder}'...")
        loader = DocumentLoader()
        documents = loader.load(docs_folder)
        skipped_files = loader.skipped_files
        print(f"  -> Ingested {len(documents)} document chunk(s); {len(skipped_files)} skipped.")

        # =====================================================================
        # Stage 2: Intake Interview
        # =====================================================================
        mode_str = "interactive" if interactive else "unattended"
        print(f"\n[Stage 2/7] Running intake interview ({mode_str} mode)...")
        interviewer = IntakeInterviewer(client=client)
        raw_requirements = interviewer.run_intake(
            brief=brief_text,
            documents=documents,
            interactive=interactive,
        )
        print(f"  -> Synthesized scope: {raw_requirements.get('scope', '')[:70]}...")

        # =====================================================================
        # Stage 3: Groundability Guard
        # =====================================================================
        print("\n[Stage 3/7] Verifying groundability against document evidence...")
        retriever = TFIDFRetriever()
        retriever.index_documents(documents)
        guard = GroundabilityGuard(retriever=retriever, client=client)

        requirements = guard.check_requirements(raw_requirements)

        # Immediately save Stage 3 artifact
        save_stage_artifacts(output_dir, requirements=requirements)
        print(f"  -> Verified topics. Saved 'requirements.json'.")

        # =====================================================================
        # Stage 4: Agent Generation
        # =====================================================================
        print("\n[Stage 4/7] Generating initial agent definition...")
        tool_registry = ToolRegistry(document_folder=docs_folder)
        generator = AgentGenerator(client=client, tool_registry=tool_registry)
        initial_definition = generator.generate_agent_definition(
            requirements=requirements,
            documents=documents,
        )
        final_definition = initial_definition

        # Immediately save Stage 4 artifacts
        save_stage_artifacts(
            output_dir,
            initial_definition=initial_definition,
            final_definition=final_definition,
        )
        print(f"  -> Generated agent '{initial_definition.get('agent_name')}'. Saved 'agent_definition.json'.")

        # =====================================================================
        # Stage 5: Evaluation Benchmark Generation
        # =====================================================================
        print("\n[Stage 5/7] Generating 20-case evaluation benchmark set...")
        eval_generator = EvaluationSetGenerator(client=client)

        distinct_docs = {str(doc.metadata.get("source", doc.doc_id)) for doc in documents}
        from forge.eval_generator import DEFAULT_TARGET_COUNTS
        target_counts = dict(DEFAULT_TARGET_COUNTS)
        if len(distinct_docs) < 2:
            omitted = target_counts.pop("cross_document", 0)
            target_counts["answerable"] = target_counts.get("answerable", 10) + omitted
            print(f"  -> Single source document detected; reallocated {omitted} cross-doc cases to answerable.")

        evaluation_cases = eval_generator.generate_eval_set(
            documents=documents,
            requirements=requirements,
            target_counts=target_counts,
        )

        # Immediately save Stage 5 artifact
        save_stage_artifacts(output_dir, evaluation_cases=evaluation_cases)
        print(f"  -> Generated {len(evaluation_cases)} benchmark cases. Saved 'eval_set.json'.")

        # =====================================================================
        # Stage 6: Baseline Evaluation & Critic-Optimizer Loop
        # =====================================================================
        print(f"\n[Stage 6/7] Running baseline evaluation and critic-optimizer loop (max {max_iterations} iterations)...")
        judge = AutomatedJudge(client=client)
        optimizer = AgentOptimizer(client=client, judge=judge, max_iterations=max_iterations)
        optimization_result = optimizer.optimize(
            agent_definition=initial_definition,
            document_folder=docs_folder,
            evaluation_cases=evaluation_cases,
        )
        final_definition = optimization_result.best_agent_definition

        # Format detailed evaluation results preserving baseline and every iteration
        eval_results_payload = {
            "baseline": {
                "scorecard": optimization_result.baseline_scorecard.to_dict(),
                "judgments": [j.to_dict() for j in optimization_result.baseline_scorecard.judgments],
            },
            "iterations": [
                {
                    "iteration": log.iteration,
                    "action_type": log.action_type,
                    "change_summary": log.change_summary,
                    "scorecard": log.scorecard.to_dict(),
                    "judgments": [j.to_dict() for j in log.scorecard.judgments],
                }
                for log in optimization_result.history
            ],
            "best": {
                "scorecard": optimization_result.best_scorecard.to_dict(),
                "judgments": [j.to_dict() for j in optimization_result.best_scorecard.judgments],
            },
        }

        # Immediately save Stage 6 artifacts
        save_stage_artifacts(
            output_dir,
            final_definition=final_definition,
            optimization_result=optimization_result,
            evaluation_results=eval_results_payload,
        )
        print(f"  -> Optimization complete. Best Pass Rate: {optimization_result.best_scorecard.pass_rate * 100:.1f}%.")
        print("  -> Saved 'iteration_logs.json', 'scorecards.json', and 'eval_results.json'.")

        # =====================================================================
        # Stage 7: Reporting
        # =====================================================================
        print("\n[Stage 7/7] Generating final audit report...")
        report_md = generate_markdown_report(
            status="SUCCESS",
            brief_text=brief_text,
            documents=documents,
            skipped_files=skipped_files,
            requirements=requirements,
            initial_definition=initial_definition,
            final_definition=final_definition,
            evaluation_cases=evaluation_cases,
            optimization_result=optimization_result,
        )
        saved = save_stage_artifacts(output_dir, final_report_md=report_md)
        print("  -> Saved 'final_report.md'.")

        print("\n" + "=" * 65)
        print("🎉 FORGEAI META-AGENT FACTORY COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        print(f"Output Artifacts Directory: {output_dir}")
        print(f"Agent Definition:          {output_dir / 'agent_definition.json'}")
        print(f"Comprehensive Report:      {output_dir / 'final_report.md'}")
        print(f"Baseline Pass Rate:        {optimization_result.baseline_scorecard.pass_rate * 100:.1f}%")
        print(f"Final Best Pass Rate:      {optimization_result.best_scorecard.pass_rate * 100:.1f}%")
        print(f"\nTo chat with the generated agent, run:")
        print(f"  python chat.py --definition {output_dir / 'agent_definition.json'} --docs {docs_folder}")
        print("=" * 65 + "\n")

        return {
            "status": "SUCCESS",
            "output_dir": str(output_dir),
            "best_pass_rate": optimization_result.best_scorecard.pass_rate,
            "baseline_pass_rate": optimization_result.baseline_scorecard.pass_rate,
            "artifacts": {k: str(v) for k, v in saved.items()},
        }

    except Exception as exc:
        print(f"\n[ERROR] Pipeline halted with error: {exc}", file=sys.stderr)
        # Honestly record incomplete run without destroying previous stage artifacts
        report_md = generate_markdown_report(
            status="INCOMPLETE",
            brief_text=brief_text,
            documents=documents,
            skipped_files=skipped_files,
            requirements=requirements,
            initial_definition=initial_definition,
            final_definition=final_definition,
            evaluation_cases=evaluation_cases,
            optimization_result=optimization_result,
            error_message=str(exc),
        )
        save_stage_artifacts(output_dir, final_report_md=report_md)
        return {
            "status": "INCOMPLETE",
            "error": str(exc),
            "output_dir": str(output_dir),
        }


def main() -> None:
    """CLI parser and runner."""
    parser = argparse.ArgumentParser(
        description="ForgeAI Meta-Agent Factory: End-to-end automated agent generator and optimizer."
    )
    parser.add_argument(
        "--brief",
        required=True,
        help="Path to requester brief text file.",
    )
    parser.add_argument(
        "--docs",
        required=True,
        help="Path to document folder containing domain files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to persist all generated artifacts and reports.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum critic-optimizer iterations (default: 3).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive CLI intake interview for ambiguous briefs.",
    )

    args = parser.parse_args()

    result = run_factory(
        brief_path=args.brief,
        docs_path=args.docs,
        output_path=args.output,
        max_iterations=args.max_iterations,
        interactive=args.interactive,
    )

    if result.get("status") != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()
