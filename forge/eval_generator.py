"""Automatic Evaluation Generator for ForgeAI.

Generates validated, benchmark-ready evaluation test cases directly from actual
ingested documents and verified requirements using OllamaClient.
"""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from forge.documents import Document
from forge.llm import OllamaClient


DEFAULT_EVAL_GENERATOR_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "eval_generator.txt"
)

DEFAULT_TARGET_COUNTS: Dict[str, int] = {
    "answerable": 10,
    "out_of_scope": 4,
    "ambiguous": 3,
    "cross_document": 3,
}

VALID_CATEGORIES = {"answerable", "out_of_scope", "ambiguous", "cross_document"}
CATEGORY_EXPECTED_STATUS = {
    "answerable": "answered",
    "out_of_scope": "refused",
    "ambiguous": "clarification_needed",
    "cross_document": "answered",
}


@dataclass
class EvaluationCase:
    """A single validated evaluation benchmark case."""

    id: str
    category: str
    question: str
    expected_status: str
    expected_key_facts: List[str] = field(default_factory=list)
    expected_sources: List[str] = field(default_factory=list)
    expected_evidence: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert case to a clean JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationCase":
        """Instantiate EvaluationCase from dictionary."""
        return cls(
            id=str(data.get("id", "")),
            category=str(data.get("category", "")),
            question=str(data.get("question", "")),
            expected_status=str(data.get("expected_status", "")),
            expected_key_facts=list(data.get("expected_key_facts", [])),
            expected_sources=list(data.get("expected_sources", [])),
            expected_evidence=list(data.get("expected_evidence", [])),
        )


def load_eval_generator_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    """Load evaluation generator prompt template from file."""
    path = Path(prompt_path) if prompt_path else DEFAULT_EVAL_GENERATOR_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Evaluation generator prompt file not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for line in content.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def normalize_text_for_matching(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and normalizing whitespace."""
    t = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()


def normalize_question_key(question: str) -> str:
    """Normalize question string for duplicate detection."""
    return re.sub(r"[^a-z0-9]", "", question.lower())


def build_documents_lookup(documents: Sequence[Document]) -> Dict[str, Document]:
    """Build a lookup mapping both full source_path and basename to Document."""
    lookup: Dict[str, Document] = {}
    for doc in documents:
        src = str(doc.metadata.get("source", doc.doc_id)).replace("\\", "/")
        lookup[src] = doc
        lookup[Path(src).name] = doc
        if doc.metadata.get("filename"):
            lookup[str(doc.metadata["filename"])] = doc
        lookup[doc.doc_id] = doc
    return lookup


def validate_excerpt_in_document(excerpt: str, doc_text: str) -> bool:
    """Verify if excerpt genuinely appears in document text."""
    clean_excerpt = normalize_text_for_matching(excerpt)
    if not clean_excerpt:
        return False

    clean_doc = normalize_text_for_matching(doc_text)
    if clean_excerpt in clean_doc:
        return True

    words = clean_excerpt.split()
    if len(words) >= 4:
        for i in range(len(words) - 3):
            quad = " ".join(words[i : i + 4])
            if quad in clean_doc:
                return True

    return False


def validate_evaluation_case(
    case: EvaluationCase,
    doc_lookup: Dict[str, Document],
    seen_question_keys: Set[str],
) -> Tuple[bool, Optional[str]]:
    """Validate a single evaluation case against actual documents and rules.

    Returns (is_valid, rejection_reason).
    """
    if case.category not in VALID_CATEGORIES:
        return False, f"Invalid category: '{case.category}'. Allowed: {sorted(VALID_CATEGORIES)}"

    q_text = case.question.strip()
    if not q_text:
        return False, "Question is empty."

    q_key = normalize_question_key(q_text)
    if not q_key:
        return False, "Question contains no alphanumeric characters."
    if q_key in seen_question_keys:
        return False, f"Duplicate question detected: '{q_text}'"

    expected_status = CATEGORY_EXPECTED_STATUS[case.category]
    if case.expected_status != expected_status:
        return (
            False,
            f"Expected status for category '{case.category}' must be '{expected_status}', got '{case.expected_status}'",
        )

    if not case.expected_key_facts or not any(str(f).strip() for f in case.expected_key_facts):
        return False, "expected_key_facts must be a non-empty list of factual strings."

    # Evidence validation for answerable and cross_document
    if case.category in ("answerable", "cross_document"):
        if not case.expected_evidence:
            return False, f"Category '{case.category}' requires at least one item in expected_evidence."

        distinct_valid_sources: Set[str] = set()

        for idx, ev in enumerate(case.expected_evidence):
            source = str(ev.get("source", "")).strip().replace("\\", "/")
            excerpt = str(ev.get("excerpt", "")).strip()

            if not source or not excerpt:
                return False, f"Evidence item {idx} missing source or excerpt."

            doc = doc_lookup.get(source) or doc_lookup.get(Path(source).name)
            if not doc or not validate_excerpt_in_document(excerpt, doc.text):
                # Search if excerpt genuinely belongs to another ingested document
                found_doc = None
                for candidate in doc_lookup.values():
                    if validate_excerpt_in_document(excerpt, candidate.text):
                        found_doc = candidate
                        break
                if found_doc:
                    doc = found_doc
                    source = str(doc.metadata.get("source", doc.doc_id))
                    ev["source"] = source
                else:
                    return (
                        False,
                        f"Fabricated excerpt in evidence: '{excerpt[:60]}...' not found in document '{source}'.",
                    )

            # Record standard source identifier
            distinct_valid_sources.add(str(doc.metadata.get("source", doc.doc_id)))

        if case.category == "cross_document":
            if len(distinct_valid_sources) < 2:
                return (
                    False,
                    f"Cross-document case requires evidence from at least 2 distinct files, got {len(distinct_valid_sources)}.",
                )

    elif case.category == "out_of_scope":
        if case.expected_status != "refused":
            return False, "Out-of-scope case expected_status must be 'refused'."

    elif case.category == "ambiguous":
        if case.expected_status != "clarification_needed":
            return False, "Ambiguous case expected_status must be 'clarification_needed'."

    return True, None


def format_documents_for_prompt(
    documents: Sequence[Document],
    category: str,
    max_total_chars: int = 2400,
) -> str:
    """Format documents into context bounds for prompting the model."""
    if not documents:
        return "No documents available."

    lines: List[str] = []
    # If cross_document, ensure at least 2 distinct documents get fair space
    per_doc_limit = max_total_chars // max(len(documents), 1)
    if category == "cross_document" and len(documents) >= 2:
        per_doc_limit = max_total_chars // min(len(documents), 3)

    for doc in documents:
        src = str(doc.metadata.get("source", doc.doc_id)).replace("\\", "/")
        text = doc.text.strip()
        if len(text) > per_doc_limit:
            text = text[:per_doc_limit].rsplit(" ", 1)[0] + " ... [truncated]"
        lines.append(f"--- Document: {src} ---\n{text}")

    return "\n\n".join(lines)


class EvaluationSetGenerator:
    """Generates and validates comprehensive evaluation benchmarks for ForgeAI."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        prompt_path: Optional[Union[str, Path]] = None,
        batch_size: int = 5,
    ) -> None:
        self.client = client or OllamaClient()
        self.prompt_template = load_eval_generator_prompt(prompt_path)
        self.batch_size = batch_size

    def generate_eval_set(
        self,
        documents: Sequence[Document],
        requirements: Dict[str, Any],
        target_counts: Optional[Dict[str, int]] = None,
        max_retries: int = 3,
    ) -> List[EvaluationCase]:
        """Generate at least 20 validated evaluation benchmark cases in small batches."""
        if not documents:
            raise ValueError("Cannot generate evaluation set without ingested documents.")

        targets = dict(DEFAULT_TARGET_COUNTS)
        if target_counts:
            targets.update(target_counts)

        doc_lookup = build_documents_lookup(documents)
        distinct_source_files = {str(doc.metadata.get("source", doc.doc_id)) for doc in documents}

        # Check cross-document feasibility honestly
        if targets.get("cross_document", 0) > 0 and len(distinct_source_files) < 2:
            raise RuntimeError(
                f"Insufficient documents for cross-document evaluation: "
                f"only {len(distinct_source_files)} distinct document available. "
                "Cross-document cases genuinely require at least 2 distinct source files."
            )

        scope = str(requirements.get("scope", ""))
        must_answer = json.dumps(requirements.get("must_answer_topics", []), indent=2)
        must_refuse = json.dumps(requirements.get("must_refuse_topics", []), indent=2)

        collected_cases: List[EvaluationCase] = []
        seen_question_keys: Set[str] = set()

        # Generate category by category in small bounded batches
        for category, target_count in targets.items():
            if target_count <= 0:
                continue

            category_cases: List[EvaluationCase] = []

            if category == "answerable" and len(documents) >= 2:
                docs_list = list(documents)
                per_doc = (target_count + len(docs_list) - 1) // len(docs_list)
                for doc in docs_list:
                    if len(category_cases) >= target_count:
                        break
                    sub_target = min(per_doc, target_count - len(category_cases))
                    doc_context = format_documents_for_prompt([doc], category=category)
                    self._generate_category_batch(
                        category=category,
                        target_count=len(category_cases) + sub_target,
                        category_cases=category_cases,
                        doc_context=doc_context,
                        scope=scope,
                        must_answer=must_answer,
                        must_refuse=must_refuse,
                        doc_lookup=doc_lookup,
                        seen_question_keys=seen_question_keys,
                        collected_cases=collected_cases,
                        max_retries=max_retries,
                    )
            else:
                doc_context = format_documents_for_prompt(documents, category=category)
                self._generate_category_batch(
                    category=category,
                    target_count=target_count,
                    category_cases=category_cases,
                    doc_context=doc_context,
                    scope=scope,
                    must_answer=must_answer,
                    must_refuse=must_refuse,
                    doc_lookup=doc_lookup,
                    seen_question_keys=seen_question_keys,
                    collected_cases=collected_cases,
                    max_retries=max_retries,
                )

            if len(category_cases) < target_count:
                raise RuntimeError(
                    f"Failed to generate valid complete set for category '{category}': "
                    f"generated {len(category_cases)}/{target_count} after {max_retries} retries."
                )

            collected_cases.extend(category_cases)

        # Renumber IDs sequentially to ensure clean 1-based order
        for idx, case in enumerate(collected_cases, start=1):
            case.id = f"eval_{idx:03d}"

        return collected_cases

    def _generate_category_batch(
        self,
        category: str,
        target_count: int,
        category_cases: List[EvaluationCase],
        doc_context: str,
        scope: str,
        must_answer: str,
        must_refuse: str,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
        collected_cases: List[EvaluationCase],
        max_retries: int,
    ) -> None:
        retries = 0
        while len(category_cases) < target_count and retries < max_retries:
            needed = target_count - len(category_cases)
            current_batch_target = min(needed, self.batch_size)

            if category_cases:
                prev_q_str = "\n".join(f"- {c.question}" for c in category_cases)
            else:
                prev_q_str = "None yet."

            prompt = (
                self.prompt_template.replace("{scope}", scope)
                .replace("{must_answer_topics}", must_answer)
                .replace("{must_refuse_topics}", must_refuse)
                .replace("{document_content}", doc_context)
                .replace("{category}", category)
                .replace("{target_count}", str(current_batch_target))
                .replace("{already_generated_questions}", prev_q_str)
            )

            system_prompt = (
                "You are a strict, objective AI evaluation benchmark designer. "
                "Output ONLY valid JSON adhering strictly to the required schema. "
                "Never invent or hallucinate document citations or text."
            )

            try:
                res = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
            except Exception:
                retries += 1
                continue

            raw_cases = res.get("cases", [])
            if not isinstance(raw_cases, list):
                retries += 1
                continue

            valid_added_in_turn = 0
            for c_dict in raw_cases:
                if not isinstance(c_dict, dict):
                    continue
                if len(category_cases) >= target_count:
                    break

                raw_facts = c_dict.get("expected_key_facts", [])
                if isinstance(raw_facts, str):
                    facts = [raw_facts] if raw_facts.strip() else []
                elif isinstance(raw_facts, list):
                    facts = [str(f) for f in raw_facts if str(f).strip()]
                else:
                    facts = []

                evidence = c_dict.get("expected_evidence", [])
                if not isinstance(evidence, list):
                    evidence = []

                sources = c_dict.get("expected_sources", [])
                if not isinstance(sources, list):
                    sources = []
                if not sources and evidence:
                    sources = sorted(
                        {
                            str(ev.get("source", ""))
                            for ev in evidence
                            if isinstance(ev, dict) and ev.get("source")
                        }
                    )

                temp_case = EvaluationCase(
                    id=f"eval_{len(collected_cases) + len(category_cases) + 1:03d}",
                    category=category,
                    question=str(c_dict.get("question", "")).strip(),
                    expected_status=str(
                        c_dict.get("expected_status", CATEGORY_EXPECTED_STATUS[category])
                    ).strip(),
                    expected_key_facts=facts,
                    expected_sources=sources,
                    expected_evidence=evidence,
                )

                is_valid, _reason = validate_evaluation_case(
                    temp_case, doc_lookup, seen_question_keys
                )
                if is_valid:
                    q_key = normalize_question_key(temp_case.question)
                    seen_question_keys.add(q_key)
                    category_cases.append(temp_case)
                    valid_added_in_turn += 1

            if valid_added_in_turn == 0:
                retries += 1

    def save_eval_set(
        self,
        cases: Sequence[EvaluationCase],
        output_path: Union[str, Path] = "eval_set.json",
    ) -> Path:
        """Save evaluation set to JSON file with metadata."""
        out = Path(output_path)
        counts: Dict[str, int] = {}
        for c in cases:
            counts[c.category] = counts.get(c.category, 0) + 1

        payload = {
            "total_cases": len(cases),
            "category_distribution": counts,
            "cases": [c.to_dict() for c in cases],
        }

        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out
