"""ForgeAI — Professional Meta-Agent Factory, Evaluation & Chat UI.

A clean, professional Streamlit interface providing:
1. Agent Factory: Requester brief input, document upload, and automated pipeline execution.
2. Evaluation Dashboard: Inspection of baseline vs final scores, category results, iteration logs, and knowledge gaps.
3. Chat Playground: Interactive real-time agent execution with verifiable citations and refusal monitoring.
"""

from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st

# ==============================================================================
# Page Configuration & Professional Dark Styling
# ==============================================================================
st.set_page_config(
    page_title="ForgeAI — Meta-Agent Studio",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Hide Streamlit hamburger menu, deploy button, status widget, and footer */
    #MainMenu { visibility: hidden; display: none !important; }
    footer { visibility: hidden; display: none !important; }
    [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    button[title="View fullscreen"] { visibility: hidden; }

    /* Ensure sidebar expand/collapse controls remain fully visible and accessible */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: block !important;
        z-index: 1000001 !important;
    }
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: block !important;
    }

    /* Compact Layout Overrides */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 95% !important;
    }

    .main-header {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
        color: #F8FAFC;
    }
    .header-sub {
        color: #94A3B8;
        font-size: 0.9rem;
        margin-bottom: 1.0rem;
    }

    /* Professional Blue & Slate Badges - No Red/Pink except genuine error */
    .forge-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .badge-answered {
        background-color: #172554;
        color: #93C5FD;
        border: 1px solid #2563EB;
    }
    .badge-refused {
        background-color: #1E293B;
        color: #CBD5E1;
        border: 1px solid #475569;
    }
    .badge-clarification {
        background-color: #0F172A;
        color: #60A5FA;
        border: 1px solid #1D4ED8;
    }
    .badge-tool {
        background-color: #1E3A8A;
        color: #BFDBFE;
        border: 1px solid #3B82F6;
    }

    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .citation-box {
        background-color: #0F172A;
        border-left: 3px solid #2563EB;
        padding: 0.5rem 0.8rem;
        border-radius: 0 4px 4px 0;
        margin-top: 0.4rem;
        font-size: 0.85rem;
        color: #E2E8F0;
    }

    .agent-summary-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# Helper Utilities & Robust Data Loaders
# ==============================================================================
WORKSPACE_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
DEMOS_DIR = WORKSPACE_ROOT / "demos"


def load_json_safe(path: Optional[Path]) -> Optional[Any]:
    """Load JSON file safely returning None if missing or invalid."""
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_text_safe(path: Optional[Path]) -> Optional[str]:
    """Load text file safely returning None if missing or unreadable."""
    if not path or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def discover_artifact_directories() -> List[Path]:
    """Discover directories containing generated agent artifacts."""
    discovered = []
    if ARTIFACTS_DIR.exists():
        for item in sorted(ARTIFACTS_DIR.iterdir()):
            if item.is_dir() and (
                (item / "agent_definition.json").exists()
                or (item / "final_report.md").exists()
                or (item / "requirements.json").exists()
                or (item / "scorecards.json").exists()
            ):
                discovered.append(item)

    if DEMOS_DIR.exists():
        for item in sorted(DEMOS_DIR.iterdir()):
            if item.is_dir() and (
                (item / "agent_definition.json").exists()
                or (item / "final_report.md").exists()
            ):
                discovered.append(item)

    return discovered


def parse_final_report_metrics(report_md: str) -> Dict[str, Any]:
    """Parse key metrics honestly from final_report.md when JSON files are missing."""
    metrics: Dict[str, Any] = {
        "pipeline_status": "UNKNOWN",
        "baseline_pass_rate": None,
        "final_pass_rate": None,
        "iterations_completed": 0,
        "total_test_cases": 0,
    }

    status_match = re.search(r"Pipeline Status\*\*:\s*([^\n]+)", report_md)
    if status_match:
        raw_status = re.sub(r"[^\w\s\-_/]", "", status_match.group(1)).strip()
        metrics["pipeline_status"] = raw_status

    baseline_match = re.search(r"Baseline Benchmark Pass Rate\*\*:\s*([\d\.]+)%", report_md)
    if baseline_match:
        metrics["baseline_pass_rate"] = float(baseline_match.group(1))

    final_match = re.search(r"Final Benchmark Pass Rate\*\*:\s*([\d\.]+)%", report_md)
    if final_match:
        metrics["final_pass_rate"] = float(final_match.group(1))

    iter_match = re.search(r"Optimization Iterations Completed\*\*:\s*(\d+)", report_md)
    if iter_match:
        metrics["iterations_completed"] = int(iter_match.group(1))

    cases_match = re.search(r"Total Test Cases\*\*:\s*(\d+)", report_md)
    if cases_match:
        metrics["total_test_cases"] = int(cases_match.group(1))

    return metrics


def extract_scorecard_data(
    scorecards_obj: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """Safely extract (baseline_card, best_card, iterations_list) from scorecards structure.

    Handles both dictionary format {'baseline': ..., 'best': ..., 'iterations': [...]}
    and list format [baseline_card, iteration_1, ...], preventing any KeyError or AttributeError.
    """
    if not scorecards_obj:
        return None, None, []

    if isinstance(scorecards_obj, dict):
        baseline = scorecards_obj.get("baseline")
        best = scorecards_obj.get("best")
        iterations = scorecards_obj.get("iterations", [])
        if not isinstance(iterations, list):
            iterations = []
        if not best and iterations:
            best = iterations[-1]
        elif not best:
            best = baseline
        return baseline, best, iterations

    elif isinstance(scorecards_obj, list):
        if not scorecards_obj:
            return None, None, []
        baseline = scorecards_obj[0] if isinstance(scorecards_obj[0], dict) else None
        best = scorecards_obj[-1] if isinstance(scorecards_obj[-1], dict) else None
        iterations = [item for item in scorecards_obj[1:] if isinstance(item, dict)]
        return baseline, best, iterations

    return None, None, []


# ==============================================================================
# Sidebar Navigation & System Status (No Emojis)
# ==============================================================================
st.sidebar.markdown(
    """
    <div style="margin-bottom: 0.5rem;">
        <div style="font-weight: 800; font-size: 1.25rem; letter-spacing: -0.02em; color: #F8FAFC;">ForgeAI</div>
        <div style="color: #60A5FA; font-size: 0.82rem; font-weight: 500;">Meta-Agent Studio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.caption("Deterministic Grounding • Grounding & Refusal Safeguards")
st.sidebar.divider()

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Agent Factory",
        "Evaluation Dashboard",
        "Chat Playground",
    ],
    index=1 if discover_artifact_directories() else 0,
)

st.sidebar.divider()
st.sidebar.subheader("System Status")
st.sidebar.markdown("- **Core Model**: `Qwen2.5 3B (qwen2.5:3b)`")
st.sidebar.markdown("- **Retriever**: `TF-IDF + Cosine Vectorizer`")
st.sidebar.markdown("- **Context Budget**: `4096 Tokens (Strict)`")
st.sidebar.markdown("- **Groundability Guard**: `Active`")

discovered_dirs = discover_artifact_directories()
if discovered_dirs:
    st.sidebar.caption(f"Detected Artifact Directories ({len(discovered_dirs)}):")
    for d in discovered_dirs:
        st.sidebar.code(str(d.relative_to(WORKSPACE_ROOT)), language="text")
else:
    st.sidebar.caption("No artifact directories detected.")


# ==============================================================================
# Tab 1: Agent Factory (No Emojis)
# ==============================================================================
if nav_choice == "Agent Factory":
    st.markdown('<div class="main-header">ForgeAI Meta-Agent Factory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">Provide a requester brief and domain documents to synthesize, ground, and benchmark a specialized agent.</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        st.subheader("1. Requester Brief")
        brief_source = st.radio(
            "Brief Source",
            ["Type / Paste Custom Brief", "Load Demo Brief", "Upload Text File"],
            horizontal=True,
        )

        brief_text = ""
        if brief_source == "Type / Paste Custom Brief":
            brief_text = st.text_area(
                "Requester Brief Details",
                height=220,
                placeholder="Specify the agent goals, target audience, preferred tone, and required capabilities...",
            )
        elif brief_source == "Load Demo Brief":
            demo_choices = {
                "HR Policy Assistant": DEMOS_DIR / "hr_policy" / "brief.txt",
                "Ship Procurement & Spares": DEMOS_DIR / "ship_procurement" / "brief.txt",
            }
            selected_demo = st.selectbox("Select Demo Brief", list(demo_choices.keys()))
            demo_path = demo_choices[selected_demo]
            if demo_path.exists():
                brief_text = demo_path.read_text(encoding="utf-8")
                st.text_area("Loaded Demo Brief", value=brief_text, height=220, disabled=True)
            else:
                st.warning(f"Demo brief not found at {demo_path}")
        else:
            uploaded_brief = st.file_uploader("Upload Brief (.txt)", type=["txt"])
            if uploaded_brief:
                brief_text = uploaded_brief.read().decode("utf-8")
                st.text_area("Uploaded Brief Preview", value=brief_text, height=180, disabled=True)

    with col_right:
        st.subheader("2. Grounding Documents")
        doc_source = st.radio(
            "Document Source",
            ["Upload Documents (PDF / CSV / TXT)", "Use Existing Demo Folder"],
            horizontal=True,
        )

        uploaded_files = []
        selected_doc_folder = None

        if doc_source == "Upload Documents (PDF / CSV / TXT)":
            uploaded_files = st.file_uploader(
                "Upload Domain Knowledge Files",
                type=["pdf", "csv", "txt"],
                accept_multiple_files=True,
            )
            if uploaded_files:
                st.info(f"{len(uploaded_files)} file(s) selected for ingestion.")
        else:
            demo_doc_folders = {
                "HR Policy Documents": DEMOS_DIR / "hr_policy" / "documents",
                "Ship Procurement Documents": DEMOS_DIR / "ship_procurement" / "documents",
            }
            chosen_folder_name = st.selectbox("Select Existing Document Folder", list(demo_doc_folders.keys()))
            selected_doc_folder = demo_doc_folders[chosen_folder_name]
            if selected_doc_folder.exists():
                doc_list = list(selected_doc_folder.iterdir())
                st.info(f"Folder contains {len(doc_list)} document(s): {', '.join([f.name for f in doc_list])}")
            else:
                st.warning(f"Folder does not exist: {selected_doc_folder}")

        st.subheader("3. Pipeline Settings")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            agent_id = st.text_input("Output Agent Identifier", value="custom_agent")
        with col_s2:
            max_iters = st.slider("Max Optimization Iterations", min_value=1, max_value=3, value=3)

    st.divider()

    build_btn = st.button("Build and Evaluate Agent", type="primary", use_container_width=True)

    if build_btn:
        if not brief_text.strip():
            st.error("Please provide a valid Requester Brief before starting the factory.")
        elif doc_source == "Upload Documents (PDF / CSV / TXT)" and not uploaded_files:
            st.error("Please upload at least one PDF, CSV, or TXT document.")
        elif doc_source == "Use Existing Demo Folder" and (not selected_doc_folder or not selected_doc_folder.exists()):
            st.error("Selected document folder does not exist.")
        else:
            target_out_dir = ARTIFACTS_DIR / agent_id.strip().replace(" ", "_")
            target_out_dir.mkdir(parents=True, exist_ok=True)

            brief_file = target_out_dir / "brief.txt"
            brief_file.write_text(brief_text, encoding="utf-8")

            if doc_source == "Upload Documents (PDF / CSV / TXT)":
                target_docs_dir = target_out_dir / "documents"
                target_docs_dir.mkdir(parents=True, exist_ok=True)
                for uf in uploaded_files:
                    out_file = target_docs_dir / uf.name
                    out_file.write_bytes(uf.getbuffer())
                run_docs_path = target_docs_dir
            else:
                run_docs_path = selected_doc_folder

            st.info(f"Target Output Directory: `{target_out_dir}`")

            progress_status = st.status("Executing ForgeAI Meta-Agent Factory Pipeline...", expanded=True)
            cmd = [
                sys.executable,
                "main.py",
                "--brief",
                str(brief_file),
                "--docs",
                str(run_docs_path),
                "--output",
                str(target_out_dir),
                "--max-iterations",
                str(max_iters),
            ]

            progress_status.write(f"Executing: `{' '.join(cmd)}`")

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(WORKSPACE_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                log_area = st.empty()
                full_logs = []
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        full_logs.append(line)
                        log_area.code("".join(full_logs[-25:]), language="text")

                ret_code = proc.wait()
                if ret_code == 0:
                    progress_status.update(label="Factory Pipeline Completed Successfully", state="complete", expanded=False)
                    st.success("Agent generated, verified, and audited. Switch to Evaluation Dashboard or Chat Playground to inspect.")
                else:
                    progress_status.update(label=f"Pipeline exited with code {ret_code}", state="error")
                    st.error("Factory execution encountered an issue. See log details above.")
            except Exception as exc:
                progress_status.update(label=f"Execution error: {exc}", state="error")
                st.exception(exc)


# ==============================================================================
# Tab 2: Evaluation Dashboard (Robust Scorecard Rendering & No Emojis)
# ==============================================================================
elif nav_choice == "Evaluation Dashboard":
    st.markdown('<div class="main-header">ForgeAI Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">Examine empirical groundability audits, pass rates, failure classifications, and honest iteration scorecards.</div>',
        unsafe_allow_html=True,
    )

    discovered_dirs = discover_artifact_directories()

    col_dir_select, col_dir_custom = st.columns([2, 1])
    with col_dir_select:
        options = [str(d.relative_to(WORKSPACE_ROOT)) for d in discovered_dirs]
        if options:
            selected_rel = st.selectbox("Select Agent Artifact Directory", options, index=0)
            selected_dir = WORKSPACE_ROOT / selected_rel
        else:
            st.warning("No artifact directories automatically detected.")
            selected_dir = ARTIFACTS_DIR / "hr_policy"

    with col_dir_custom:
        custom_input = st.text_input("Or Custom Directory Path", placeholder="artifacts/my_agent")
        if custom_input.strip():
            selected_dir = Path(custom_input.strip())
            if not selected_dir.is_absolute():
                selected_dir = WORKSPACE_ROOT / selected_dir

    st.markdown(f"**Inspecting Directory**: `{selected_dir}`")

    # Locate individual artifacts
    report_file = selected_dir / "final_report.md"
    req_file = selected_dir / "requirements.json"
    agent_file = selected_dir / "agent_definition.json"
    eval_set_file = selected_dir / "eval_set.json"
    scorecards_file = selected_dir / "scorecards.json"
    iter_logs_file = selected_dir / "iteration_logs.json"
    eval_results_file = selected_dir / "eval_results.json"

    # Status pills for artifacts (No Emojis)
    col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)
    col_p1.caption(f"Report: {'[Present]' if report_file.exists() else '[Missing]'}")
    col_p2.caption(f"Requirements: {'[Present]' if req_file.exists() else '[Missing]'}")
    col_p3.caption(f"Agent Def: {'[Present]' if agent_file.exists() else '[Missing]'}")
    col_p4.caption(f"Scorecards: {'[Present]' if scorecards_file.exists() else '[Missing]'}")
    col_p5.caption(f"Eval Set: {'[Present]' if eval_set_file.exists() else '[Missing]'}")

    st.divider()

    # Load artifacts safely
    report_md = load_text_safe(report_file)
    requirements = load_json_safe(req_file)
    agent_def = load_json_safe(agent_file)
    eval_set = load_json_safe(eval_set_file)
    scorecards_raw = load_json_safe(scorecards_file)
    iter_logs = load_json_safe(iter_logs_file)
    eval_results = load_json_safe(eval_results_file)

    if not report_md and not scorecards_raw and not agent_def:
        st.warning("No evaluation artifacts found in this directory. Run the Agent Factory to generate them.")
    else:
        # Extract baseline, best, and iteration scorecards safely
        baseline_card, best_card, iterations_list = extract_scorecard_data(scorecards_raw)

        # 1. Executive Summary Metrics
        st.subheader("1. Executive Summary & Benchmark Scores")

        baseline_score = None
        final_score = None
        status_text = "UNKNOWN"
        total_cases = 0

        if baseline_card:
            raw_b = baseline_card.get("pass_rate")
            if raw_b is not None:
                baseline_score = raw_b * 100.0 if raw_b <= 1.0 else raw_b
            total_cases = baseline_card.get("total_cases", 0)
            status_text = "Completed"

        if best_card:
            raw_f = best_card.get("pass_rate")
            if raw_f is not None:
                final_score = raw_f * 100.0 if raw_f <= 1.0 else raw_f
            if not total_cases:
                total_cases = best_card.get("total_cases", 0)
            status_text = "Completed"

        # Fallback to report_md parsing if JSON fields were missing
        if report_md and (baseline_score is None or final_score is None):
            parsed_m = parse_final_report_metrics(report_md)
            if baseline_score is None:
                baseline_score = parsed_m["baseline_pass_rate"]
            if final_score is None:
                final_score = parsed_m["final_pass_rate"]
            if not total_cases:
                total_cases = parsed_m["total_test_cases"]
            if parsed_m["pipeline_status"] != "UNKNOWN":
                status_text = parsed_m["pipeline_status"]

        delta = (final_score - baseline_score) if (final_score is not None and baseline_score is not None) else None

        # Clean compact status text to prevent truncation
        clean_status = re.sub(r"[^\w\s\-_/]", "", status_text).strip()
        if "COMPLETED" in clean_status.upper():
            display_status = "Completed"
        elif "INCOMPLETE" in clean_status.upper() or "FAILED" in clean_status.upper():
            display_status = "Incomplete"
        else:
            display_status = clean_status[:14]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Baseline Pass Rate", f"{baseline_score:.1f}%" if baseline_score is not None else "N/A")
        m2.metric(
            "Final Best Pass Rate",
            f"{final_score:.1f}%" if final_score is not None else "N/A",
            delta=f"{delta:+.1f}%" if delta is not None else None,
        )
        m3.metric("Evaluated Cases", total_cases if total_cases > 0 else (len(eval_set.get("cases", eval_set)) if isinstance(eval_set, (dict, list)) and eval_set else "N/A"))
        m4.metric("Pipeline Status", display_status)

        # 2. Scorecard Progression & Category Breakdown
        st.subheader("2. Category-Wise Performance & Iteration Progression")
        tab_cat, tab_iter = st.tabs(["Category Performance", "Scorecard Progression"])

        with tab_cat:
            cat_data = []
            active_card = best_card or baseline_card

            if active_card and isinstance(active_card, dict):
                cat_results = active_card.get("category_scores") or active_card.get("category_results") or {}
                for cat_name, c_info in cat_results.items():
                    raw_rate = c_info.get("pass_rate", 0.0)
                    rate_pct = raw_rate * 100.0 if raw_rate <= 1.0 else raw_rate
                    cat_data.append({
                        "Category": cat_name,
                        "Passed": c_info.get("passed", 0),
                        "Total": c_info.get("total", 0),
                        "Pass Rate": f"{rate_pct:.1f}%",
                    })
            elif eval_results and isinstance(eval_results, list):
                totals = Counter([r.get("category", "unknown") for r in eval_results])
                passed = Counter([r.get("category", "unknown") for r in eval_results if r.get("passed", False)])
                for cat_name in totals:
                    tot = totals[cat_name]
                    pas = passed[cat_name]
                    rate = (pas / tot * 100.0) if tot > 0 else 0.0
                    cat_data.append({
                        "Category": cat_name,
                        "Passed": pas,
                        "Total": tot,
                        "Pass Rate": f"{rate:.1f}%",
                    })

            if cat_data:
                st.table(cat_data)
            else:
                st.info("Category breakdown details not available in parsed JSON format. Inspect full report below.")

        with tab_iter:
            iter_rows = []
            if baseline_card and isinstance(baseline_card, dict):
                raw_b = baseline_card.get("pass_rate", 0.0)
                b_pct = raw_b * 100.0 if raw_b <= 1.0 else raw_b
                iter_rows.append({
                    "Iteration": "Baseline (0)",
                    "Pass Rate (%)": f"{b_pct:.1f}%",
                    "Passed Cases": baseline_card.get("passed_cases", 0),
                    "Failed Cases": baseline_card.get("failed_cases", 0),
                    "Total Cases": baseline_card.get("total_cases", 0),
                })

            if iterations_list and isinstance(iterations_list, list):
                for idx, it in enumerate(iterations_list, start=1):
                    if isinstance(it, dict):
                        it_num = it.get("iteration", idx)
                        raw_p = it.get("pass_rate", 0.0)
                        p_pct = raw_p * 100.0 if raw_p <= 1.0 else raw_p
                        iter_rows.append({
                            "Iteration": f"Iteration {it_num}",
                            "Pass Rate (%)": f"{p_pct:.1f}%",
                            "Passed Cases": it.get("passed_cases", 0),
                            "Failed Cases": it.get("failed_cases", 0),
                            "Total Cases": it.get("total_cases", 0),
                        })

            if iter_rows:
                st.dataframe(iter_rows, use_container_width=True)
            elif report_md and "## 6. Critic–Optimizer Loop Performance" in report_md:
                st.markdown("*(Extracted from Final Report markdown)*")
                sec6 = report_md.split("## 6. Critic–Optimizer Loop Performance")[-1].split("## 7.")[0]
                st.markdown(sec6)
            else:
                st.info("No multi-iteration history recorded.")

        # 3. Iteration Diagnosis & Action Summary
        if iter_logs and isinstance(iter_logs, list):
            st.subheader("3. Critic Diagnoses and Targeted Optimization Actions")
            log_table = []
            for item in iter_logs:
                if isinstance(item, dict):
                    fc = item.get("failure_categories", {})
                    fc_str = ", ".join([f"{k}: {v}" for k, v in fc.items()]) if isinstance(fc, dict) else str(fc)
                    log_table.append({
                        "Iteration": item.get("iteration", 1),
                        "Action Type": item.get("action_type", ""),
                        "Diagnosed Failures": fc_str or "None",
                        "Change Applied": item.get("change_summary", ""),
                        "Regressions": len(item.get("regressions", [])) if isinstance(item.get("regressions"), list) else 0,
                    })
            if log_table:
                st.dataframe(log_table, use_container_width=True)

        # 4. Knowledge Gaps & Groundability Guard
        st.subheader("4. Knowledge Gaps and Groundability Guard Decisions")
        col_must_ans, col_must_ref = st.columns(2)

        with col_must_ans:
            st.markdown("##### Permitted Must-Answer Topics")
            must_answer = []
            if requirements and "permitted_topics" in requirements:
                must_answer = requirements["permitted_topics"]
            elif agent_def and "must_answer_topics" in agent_def:
                must_answer = agent_def["must_answer_topics"]

            if must_answer:
                for topic in must_answer:
                    st.markdown(f"- {topic}")
            else:
                st.caption("No explicit permitted topics list found.")

        with col_must_ref:
            st.markdown("##### Must-Refuse Topics (Knowledge Gaps)")
            must_refuse = []
            if requirements and "unsupported_topics" in requirements:
                must_refuse = requirements["unsupported_topics"]
            elif agent_def and "must_refuse_topics" in agent_def:
                must_refuse = agent_def["must_refuse_topics"]

            if must_refuse:
                for topic in must_refuse:
                    st.markdown(f"- {topic}")
            else:
                st.caption("No explicit unsupported topics recorded.")

        # 5. Downloads & Raw Report Preview
        st.subheader("5. Artifact Downloads and Full Audit Report")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)

        if report_md:
            col_d1.download_button(
                "Download final_report.md",
                data=report_md,
                file_name="final_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        if agent_def:
            col_d2.download_button(
                "Download agent_definition.json",
                data=json.dumps(agent_def, indent=2),
                file_name="agent_definition.json",
                mime="application/json",
                use_container_width=True,
            )
        if eval_set:
            col_d3.download_button(
                "Download eval_set.json",
                data=json.dumps(eval_set, indent=2),
                file_name="eval_set.json",
                mime="application/json",
                use_container_width=True,
            )
        if scorecards_raw:
            col_d4.download_button(
                "Download scorecards.json",
                data=json.dumps(scorecards_raw, indent=2),
                file_name="scorecards.json",
                mime="application/json",
                use_container_width=True,
            )

        if report_md:
            with st.expander("View Full Markdown Audit Report", expanded=False):
                st.markdown(report_md)


# ==============================================================================
# Tab 3: Chat Playground (No Emojis)
# ==============================================================================
elif nav_choice == "Chat Playground":
    st.markdown('<div class="main-header">ForgeAI Agent Chat Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="header-sub">Engage in real conversational sessions with generated agents, verified citations, and automated refusal policies.</div>',
        unsafe_allow_html=True,
    )

    discovered_dirs = discover_artifact_directories()

    col_cfg1, col_cfg2 = st.columns([1, 1])

    with col_cfg1:
        agent_options = {}
        for d in discovered_dirs:
            ag_path = d / "agent_definition.json"
            if ag_path.exists():
                agent_options[str(d.relative_to(WORKSPACE_ROOT))] = ag_path

        if agent_options:
            selected_agent_key = st.selectbox("Select Generated Agent", list(agent_options.keys()))
            selected_def_path = agent_options[selected_agent_key]
        else:
            selected_def_path = ARTIFACTS_DIR / "hr_policy" / "agent_definition.json"
            st.warning("No agent definition automatically found. Specifying fallback path.")

        custom_def = st.text_input("Or Custom Definition Path", value=str(selected_def_path))
        def_path_final = Path(custom_def.strip())

    with col_cfg2:
        guessed_doc_folder = WORKSPACE_ROOT / "demos" / "hr_policy" / "documents"
        if "procurement" in str(def_path_final).lower():
            guessed_doc_folder = WORKSPACE_ROOT / "demos" / "ship_procurement" / "documents"

        doc_folder_input = st.text_input(
            "Document Knowledge Folder Path",
            value=str(guessed_doc_folder),
            help="Directory containing source PDFs, CSVs, or text documents",
        )
        doc_folder_final = Path(doc_folder_input.strip())

    # Agent Metadata Card
    agent_data = load_json_safe(def_path_final)
    if agent_data and isinstance(agent_data, dict):
        tool_count = len(agent_data.get("tools", []))
        tool_badge_text = f"{tool_count} Authorized Tools" if tool_count > 0 else "Zero-Tool Assistant"
        st.markdown(
            f"""
            <div class="agent-summary-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.15rem; color: #60A5FA;">{agent_data.get('agent_name', 'Unnamed Agent')}</span>
                    <span class="forge-badge badge-tool">{tool_badge_text}</span>
                </div>
                <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 4px;">{agent_data.get('description', '')}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 5px;">
                    <b>Permitted Scope:</b> {agent_data.get('scope', 'Document-based Q&A')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("View Agent System Prompt and Retrieval Config"):
            st.code(agent_data.get("system_prompt", ""), language="text")
            st.json(agent_data.get("retrieval_config", {}))

    # Initialize Chat History in Session State
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Chat controls
    col_chat_title, col_chat_clear = st.columns([4, 1])
    with col_chat_title:
        st.subheader("Conversation")
    with col_chat_clear:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()

    # Display Existing Messages
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                status = msg.get("status", "answered")
                if status == "refused":
                    st.markdown('<span class="forge-badge badge-refused">Status: Refused</span>', unsafe_allow_html=True)
                elif status == "clarification_needed":
                    st.markdown('<span class="forge-badge badge-clarification">Status: Clarification Needed</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="forge-badge badge-answered">Status: Answered</span>', unsafe_allow_html=True)

                st.markdown(msg["content"])

                # Citations Box
                citations = msg.get("citations", [])
                if citations:
                    with st.expander(f"Verified Citations ({len(citations)})", expanded=False):
                        for idx, cit in enumerate(citations):
                            src = cit.get("source", "unknown")
                            cid = cit.get("chunk_id", "")
                            excerpt = cit.get("excerpt", "")
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <b>[{idx + 1}] Source:</b> <code>{src}</code> (Chunk: <code>{cid}</code>)<br/>
                                    <em>"{excerpt}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # Tool executions
                tools = msg.get("tool_executions", [])
                if tools:
                    with st.expander(f"Tool Executions ({len(tools)})", expanded=False):
                        st.json(tools)

                # Execution errors (genuine errors only)
                errs = msg.get("execution_errors", [])
                if errs:
                    st.error("Execution Errors: " + "; ".join(errs))
            else:
                st.markdown(msg["content"])

    # Chat Input
    user_query = st.chat_input("Ask a question based on the agent's verified documents...")

    if user_query:
        if not def_path_final.exists():
            st.error(f"Agent definition file not found at `{def_path_final}`")
        elif not doc_folder_final.exists() or not doc_folder_final.is_dir():
            st.error(f"Document knowledge folder not found at `{doc_folder_final}`")
        else:
            st.session_state.chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving document evidence and executing agent..."):
                    try:
                        from forge.runner import AgentRunner

                        runner = AgentRunner(
                            agent_definition=def_path_final,
                            document_folder=doc_folder_final,
                        )
                        result = runner.answer(user_query)

                        status = result.status
                        if status == "refused":
                            st.markdown('<span class="forge-badge badge-refused">Status: Refused</span>', unsafe_allow_html=True)
                        elif status == "clarification_needed":
                            st.markdown('<span class="forge-badge badge-clarification">Status: Clarification Needed</span>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="forge-badge badge-answered">Status: Answered</span>', unsafe_allow_html=True)

                        st.markdown(result.answer)

                        if result.citations:
                            with st.expander(f"Verified Citations ({len(result.citations)})", expanded=True):
                                for idx, cit in enumerate(result.citations):
                                    src = cit.get("source", "unknown")
                                    cid = cit.get("chunk_id", "")
                                    excerpt = cit.get("excerpt", "")
                                    st.markdown(
                                        f"""
                                        <div class="citation-box">
                                            <b>[{idx + 1}] Source:</b> <code>{src}</code> (Chunk: <code>{cid}</code>)<br/>
                                            <em>"{excerpt}"</em>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                        if result.tool_executions:
                            with st.expander(f"Tool Executions ({len(result.tool_executions)})", expanded=False):
                                st.json(result.tool_executions)

                        if result.execution_errors:
                            st.error("Execution Errors: " + "; ".join(result.execution_errors))

                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": result.answer,
                            "status": result.status,
                            "citations": result.citations,
                            "tool_executions": result.tool_executions,
                            "execution_errors": result.execution_errors,
                        })

                    except Exception as exc:
                        st.error(f"Error executing agent runner: {exc}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")
