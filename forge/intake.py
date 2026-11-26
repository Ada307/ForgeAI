"""Intake Interviewer module for ForgeAI.

Analyzes brief and ingested documents, generates at most 5 clarifying questions,
collects answers or explicit assumptions, and synthesizes validated requirements.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from forge.documents import Document
from forge.llm import OllamaClient


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "intake.txt"


def load_intake_prompts(prompt_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Load question and requirements prompt templates from prompts/intake.txt."""
    path = Path(prompt_path) if prompt_path else DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Intake prompt file not found: {path}")

    content = path.read_text(encoding="utf-8")
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    buffer: List[str] = []

    for line in content.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("[") and line_stripped.endswith("]"):
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
                buffer = []
            current_key = line_stripped[1:-1].strip()
        else:
            buffer.append(line)

    if current_key:
        sections[current_key] = "\n".join(buffer).strip()

    return sections


def build_document_preview(
    documents: Sequence[Document],
    max_words_per_doc: int = 120,
    total_word_budget: int = 1200,
) -> Tuple[str, List[str], List[str]]:
    """Build a bounded, transparent document context preview respecting context limits.

    Returns:
        (preview_text, inspected_doc_ids, omitted_doc_ids)
    """
    inspected_docs: List[str] = []
    omitted_docs: List[str] = []
    preview_parts: List[str] = []
    consumed_words = 0

    for doc in documents:
        doc_words = doc.text.split()
        if consumed_words + min(len(doc_words), max_words_per_doc) > total_word_budget:
            omitted_docs.append(doc.doc_id)
            continue

        sample = " ".join(doc_words[:max_words_per_doc])
        inspected_docs.append(doc.doc_id)
        consumed_words += len(sample.split())

        source_info = doc.metadata.get("source", doc.doc_id)
        page_info = f", Page {doc.metadata['page']}" if "page" in doc.metadata else ""
        row_info = f", Row {doc.metadata['row']}" if "row" in doc.metadata else ""

        preview_parts.append(
            f"--- Source: {source_info}{page_info}{row_info} (ID: {doc.doc_id}) ---\n{sample}"
        )

    preview_text = "\n\n".join(preview_parts)
    if omitted_docs:
        preview_text += (
            f"\n\n[NOTE: Inspected {len(inspected_docs)} of {len(documents)} documents "
            f"due to context constraints. Omitted documents: {', '.join(omitted_docs[:5])}...]"
        )

    return preview_text, inspected_docs, omitted_docs


def make_conservative_assumption(question: str, default_assumption: str) -> str:
    """Ensure assumptions in unattended mode are strictly conservative and never expand scope."""
    clean = default_assumption.strip()
    lower = clean.lower()
    if lower.startswith("yes") or "include" in lower or "add" in lower:
        q_lower = question.lower()
        if any(w in q_lower for w in ("should", "is it necessary", "would you like", "do you want", "include", "provide")):
            return (
                f"No; do not include unrequested features for '{question}'. "
                "Restrict scope strictly to the provided brief and verified documentation."
            )
    return clean or "No unrequested capabilities; restrict strictly to confirmed brief and document evidence."


def validate_requirements(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate structure and data types of requirements dictionary."""
    if not isinstance(data, dict):
        raise ValueError(f"Requirements must be a dictionary, got {type(data).__name__}")

    for field_name in ("scope", "audience", "tone"):
        val = data.get(field_name)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"Missing or invalid required string field: '{field_name}'")

    for list_field in ("must_answer_topics", "must_refuse_topics"):
        val = data.get(list_field)
        if not isinstance(val, list):
            raise ValueError(f"Field '{list_field}' must be a list, got {type(val).__name__}")
        for item in val:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"Items in '{list_field}' must be non-empty strings, got: {item!r}")

    if not data["must_answer_topics"]:
        raise ValueError("Field 'must_answer_topics' cannot be empty.")

    # Ensure optional/tracking fields exist
    data.setdefault("clarifying_questions", [])
    data.setdefault("answers", {})
    data.setdefault("assumptions", {})

    return data


class IntakeInterviewer:
    """Interviewer agent that inspects brief and documents to formulate requirements."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        prompt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.client = client or OllamaClient()
        self.prompts = load_intake_prompts(prompt_path)

    def generate_clarifying_questions(
        self,
        brief: str,
        documents: Sequence[Document],
    ) -> List[Dict[str, str]]:
        """Generate at most 5 relevant clarifying questions with default assumptions."""
        doc_preview, _, _ = build_document_preview(documents)

        prompt_template = self.prompts.get(
            "QUESTION_PROMPT",
            "Generate at most 5 clarifying questions for the brief and documents.\nBrief: {brief}\nDocuments: {document_context}",
        )
        prompt = prompt_template.replace("{brief}", brief).replace("{document_context}", doc_preview)
        system_prompt = (
            "You are an expert AI requirements analyst. Return only JSON with up to 5 clarifying questions."
        )

        response = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)

        raw_questions = response.get("questions", [])
        if not isinstance(raw_questions, list):
            raw_questions = []

        validated_questions: List[Dict[str, str]] = []
        for item in raw_questions:
            if isinstance(item, dict) and "question" in item:
                q_text = str(item["question"]).strip()
                assumption = str(item.get("default_assumption", "Assumed reasonable standard defaults.")).strip()
            elif isinstance(item, str):
                q_text = item.strip()
                assumption = "Assumed reasonable standard defaults."
            else:
                continue

            if q_text:
                validated_questions.append({
                    "question": q_text,
                    "default_assumption": assumption,
                })

        # Enforce strict upper bound of at most 5 questions
        return validated_questions[:5]

    def collect_answers(
        self,
        questions: List[Dict[str, str]],
        interactive: bool = False,
        answers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Collect answers for questions or record explicit assumptions if unattended.

        Never fabricates requester answers.
        """
        provided_answers = answers or {}
        recorded_answers: Dict[str, str] = {}
        recorded_assumptions: Dict[str, str] = {}

        for item in questions:
            q = item["question"]
            default_assump = item.get("default_assumption", "Assumed reasonable standard defaults.")

            if q in provided_answers and provided_answers[q].strip():
                recorded_answers[q] = provided_answers[q].strip()
            elif interactive:
                try:
                    user_input = input(f"\n[Clarifying Question]: {q}\nYour answer (leave blank to assume default): ").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = ""

                if user_input:
                    recorded_answers[q] = user_input
                else:
                    recorded_assumptions[q] = default_assump
            else:
                # Unattended mode: record conservative assumption explicitly.
                # Never introduce unrequested capabilities through unanswered questions.
                recorded_assumptions[q] = make_conservative_assumption(q, default_assump)

        return recorded_answers, recorded_assumptions

    def generate_requirements(
        self,
        brief: str,
        documents: Sequence[Document],
        questions: List[str],
        answers: Dict[str, str],
        assumptions: Dict[str, str],
    ) -> Dict[str, Any]:
        """Synthesize structured, validated requirements JSON from interview context."""
        doc_preview, inspected_docs, omitted_docs = build_document_preview(documents)

        # Build Q&A context block
        qa_lines: List[str] = []
        for q in questions:
            if q in answers:
                qa_lines.append(f"Q: {q}\nA (Requester Answered): {answers[q]}")
            else:
                assump = assumptions.get(q, "Assumed standard defaults.")
                qa_lines.append(f"Q: {q}\nA (Unanswered, Explicit Assumption): {assump}")

        qa_context = "\n".join(qa_lines) if qa_lines else "No clarifying questions were raised."

        prompt_template = self.prompts.get(
            "REQUIREMENTS_PROMPT",
            "Synthesize requirements for: {brief}\nDocuments: {document_context}\nQ&A: {qa_context}",
        )
        prompt = (
            prompt_template.replace("{brief}", brief)
            .replace("{document_context}", doc_preview)
            .replace("{qa_context}", qa_context)
        )
        system_prompt = (
            "You are an expert AI system architect. Output valid JSON containing "
            "scope, audience, tone, must_answer_topics, and must_refuse_topics."
        )

        response = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)

        # Attach metadata and interview traces
        response["clarifying_questions"] = questions
        response["answers"] = answers
        response["assumptions"] = assumptions
        response["inspected_documents"] = inspected_docs
        response["omitted_documents"] = omitted_docs

        return validate_requirements(response)

    def conduct_intake(
        self,
        brief: str,
        documents: Sequence[Document],
        interactive: bool = False,
        answers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Conduct complete intake interview workflow and return validated requirements."""
        question_items = self.generate_clarifying_questions(brief=brief, documents=documents)
        recorded_answers, recorded_assumptions = self.collect_answers(
            questions=question_items,
            interactive=interactive,
            answers=answers,
        )

        questions = [item["question"] for item in question_items]
        return self.generate_requirements(
            brief=brief,
            documents=documents,
            questions=questions,
            answers=recorded_answers,
            assumptions=recorded_assumptions,
        )
