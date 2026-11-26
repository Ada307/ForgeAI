"""Mocked orchestration tests for ForgeAI Factory Pipeline (main.py, forge/reporting.py)."""

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import unittest

from forge.generator import OUTPUT_CONTRACT_SCHEMA
from forge.llm import OllamaClient
from forge.runner import AgentRunner
from main import run_factory


class MockFactoryOllamaClient:
    """Mock client providing controlled responses for factory orchestration stages."""

    def __init__(self) -> None:
        self.call_count = 0
        self.eval_counter = 0

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        return "Mock plain text"

    def generate_json(self, prompt: str, system_prompt: str = "") -> Dict[str, Any]:
        self.call_count += 1
        sys_lower = system_prompt.lower()
        prompt_lower = prompt.lower()

        # 1. Intake: Clarifying questions
        if "up to 5 clarifying questions" in sys_lower or "clarifying questions" in sys_lower:
            return {
                "questions": [
                    {
                        "question": "What is the expected operating temperature limit?",
                        "default_assumption": "85 degrees Celsius",
                    }
                ]
            }

        # 2. Intake: Requirements synthesis
        if "scope, audience, tone" in sys_lower:
            return {
                "scope": "Technical guidance on Slurry Pump P-302.",
                "audience": "Maintenance technicians",
                "tone": "Direct",
                "must_answer_topics": ["operating temperature limit"],
                "must_refuse_topics": ["company financial revenue [Refusal: Unsupported]"],
                "assumptions": ["Standard operating environment assumed"],
                "clarifying_questions": [],
            }

        # 3. Groundability Guard
        if "groundability" in sys_lower or "verification" in sys_lower:
            return {
                "topic": "operating temperature limit",
                "status": "supported",
                "supported_portion": "operating temperature limit",
                "reason": "Directly stated in pump_spec.txt",
                "citations": [
                    {
                        "source": "pump_spec.txt",
                        "chunk_id": "pump_spec.txt:c0",
                        "excerpt": "operating temperature limit is 85 degrees Celsius",
                    }
                ],
                "knowledge_gaps": [],
            }

        # 4. Evaluation Benchmark Generator
        if "evaluation benchmark" in sys_lower:
            self.eval_counter += 1
            if "target category: out_of_scope" in prompt_lower:
                cat = "out_of_scope"
                exp_status = "refused"
                evidence = []
                sources = []
            elif "target category: ambiguous" in prompt_lower:
                cat = "ambiguous"
                exp_status = "clarification_needed"
                evidence = []
                sources = []
            elif "target category: cross_document" in prompt_lower:
                cat = "cross_document"
                exp_status = "answered"
                evidence = [
                    {"source": "pump_spec.txt", "excerpt": "operating temperature limit is 85 degrees Celsius"},
                    {"source": "maintenance_guide.txt", "excerpt": "Mobilith SHC 220"},
                ]
                sources = ["pump_spec.txt", "maintenance_guide.txt"]
            else:
                cat = "answerable"
                exp_status = "answered"
                evidence = [{"source": "pump_spec.txt", "excerpt": "operating temperature limit is 85 degrees Celsius"}]
                sources = ["pump_spec.txt"]

            return {
                "cases": [
                    {
                        "category": cat,
                        "question": f"Evaluation Question {cat} {self.eval_counter}_{idx} for P-302?",
                        "expected_status": exp_status,
                        "expected_key_facts": ["Standard verified fact"],
                        "expected_sources": sources,
                        "expected_evidence": evidence,
                    }
                    for idx in range(5)
                ]
            }

        # 5. Agent Generator
        if "meta-agent" in sys_lower or "system prompt" in sys_lower:
            return {
                "agent_name": "PumpSpecAgent",
                "description": "Assistant for Slurry Pump P-302",
                "system_prompt": "You are PumpSpecAgent. Answer strictly using retrieved evidence.",
                "selected_tools": [],
                "retrieval_config": {
                    "chunk_size": 180,
                    "chunk_overlap": 30,
                    "top_k": 3,
                    "minimum_similarity_score": 0.0,
                },
                "guardrails": ["Refuse forbidden topics"],
            }

        # 6. Judge
        if "judge" in sys_lower:
            return {
                "factual_score": 1.0,
                "grounding_score": 1.0,
                "passed": True,
                "reason": "Directly verified against pump_spec.txt evidence.",
            }

        # 7. Default Runner Answer
        return {
            "status": "answered",
            "answer": "The operating temperature limit is 85 degrees Celsius.",
            "citations": [
                {
                    "source": "pump_spec.txt",
                    "chunk_id": "pump_spec.txt:c0",
                    "excerpt": "operating temperature limit is 85 degrees Celsius",
                }
            ],
        }


class TestFactoryOrchestration(unittest.TestCase):
    """Essential mocked tests for full pipeline execution, reporting, and artifact persistence."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

        # Create brief file
        self.brief_file = self.root_path / "brief.txt"
        self.brief_file.write_text(
            "Build an assistant for Slurry Pump P-302 operating limits for maintenance staff.",
            encoding="utf-8",
        )

        # Create docs folder with two distinct documents
        self.docs_folder = self.root_path / "docs"
        self.docs_folder.mkdir()
        (self.docs_folder / "pump_spec.txt").write_text(
            "Slurry Pump P-302 Operational Specifications:\n"
            "Standard operating temperature limit is 85 degrees Celsius.\n"
            "Routine maintenance interval is 250 operating hours.\n",
            encoding="utf-8",
        )
        (self.docs_folder / "maintenance_guide.txt").write_text(
            "Slurry Pump P-302 Maintenance Procedures:\n"
            "Bearing lubricant grease is Mobilith SHC 220.\n",
            encoding="utf-8",
        )

        self.output_dir = self.root_path / "output"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pipeline_orchestration_mocked_success(self) -> None:
        """Verify complete pipeline executes, returns SUCCESS, and creates all 8 artifacts."""
        mock_client = MockFactoryOllamaClient()

        result = run_factory(
            brief_path=self.brief_file,
            docs_path=self.docs_folder,
            output_path=self.output_dir,
            max_iterations=2,
            interactive=False,
            client=mock_client,  # type: ignore
        )

        self.assertEqual(result["status"], "SUCCESS")
        self.assertGreater(result["best_pass_rate"], 0.0)

        # Verify all 8 required artifacts are saved
        expected_artifacts = [
            "requirements.json",
            "initial_agent_definition.json",
            "agent_definition.json",
            "eval_set.json",
            "iteration_logs.json",
            "scorecards.json",
            "eval_results.json",
            "final_report.md",
        ]
        for art in expected_artifacts:
            art_path = self.output_dir / art
            self.assertTrue(art_path.exists(), f"Missing expected artifact: {art}")
            self.assertGreater(art_path.stat().st_size, 0, f"Artifact {art} is empty")

        # Verify final_report.md structure
        report_text = (self.output_dir / "final_report.md").read_text(encoding="utf-8")
        self.assertIn("Pipeline Status", report_text)
        self.assertIn("Ingested Documents", report_text)
        self.assertIn("Intake Requirements", report_text)
        self.assertIn("Generated Agent Architecture", report_text)
        self.assertIn("Critic–Optimizer Loop Performance", report_text)
        self.assertIn("Limitations & Boundaries", report_text)

    def test_pipeline_stage_failure_preserves_artifacts_and_marks_incomplete(self) -> None:
        """Verify failures midway preserve earlier stage artifacts and record INCOMPLETE report."""
        # Nonexistent documents path triggers early failure
        bad_docs = self.root_path / "nonexistent_docs"

        result = run_factory(
            brief_path=self.brief_file,
            docs_path=bad_docs,
            output_path=self.output_dir,
            max_iterations=1,
            interactive=False,
        )

        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("error", result)

        # Report must be written and marked incomplete
        report_path = self.output_dir / "final_report.md"
        self.assertTrue(report_path.exists())
        content = report_path.read_text(encoding="utf-8")
        self.assertIn("INCOMPLETE", content)
        self.assertIn("Termination Error", content)

    def test_chat_compatibility_with_final_agent_definition(self) -> None:
        """Verify final generated agent_definition.json can be loaded and executed by AgentRunner."""
        mock_client = MockFactoryOllamaClient()

        run_factory(
            brief_path=self.brief_file,
            docs_path=self.docs_folder,
            output_path=self.output_dir,
            max_iterations=1,
            interactive=False,
            client=mock_client,  # type: ignore
        )

        agent_def_path = self.output_dir / "agent_definition.json"
        self.assertTrue(agent_def_path.exists())

        # Instantiate runner as chat.py does
        runner = AgentRunner(
            agent_definition=agent_def_path,
            document_folder=self.docs_folder,
            client=mock_client,  # type: ignore
        )

        ans = runner.answer("What is the operating temperature limit?")
        self.assertEqual(ans.status, "answered")
        self.assertIn("85 degrees Celsius", ans.answer)

    def test_cli_argument_parsing(self) -> None:
        """Verify main.py CLI parser accepts required and optional flags."""
        import subprocess

        cmd = [
            ".venv\\Scripts\\python.exe",
            "main.py",
            "--help",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("--brief", res.stdout)
        self.assertIn("--docs", res.stdout)
        self.assertIn("--output", res.stdout)
        self.assertIn("--max-iterations", res.stdout)
        self.assertIn("--interactive", res.stdout)


if __name__ == "__main__":
    unittest.main()
