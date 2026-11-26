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
    composition_metadata: Optional[Dict[str, Any]] = None

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
            composition_metadata=data.get("composition_metadata"),
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

            if not excerpt:
                return False, f"Evidence item {idx} missing excerpt."

            doc = (doc_lookup.get(source) or doc_lookup.get(Path(source).name)) if source else None
            if not doc or not validate_excerpt_in_document(excerpt, doc.text):
                # Search if excerpt genuinely belongs to any ingested document
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
                    target_doc_str = f"document '{source}'" if source else "any ingested document"
                    return (
                        False,
                        f"Fabricated excerpt in evidence: '{excerpt[:60]}...' not found in {target_doc_str}.",
                    )

            # Record standard source identifier
            distinct_valid_sources.add(str(doc.metadata.get("source", doc.doc_id)))

        # Assign verified source list programmatically
        case.expected_sources = sorted(distinct_valid_sources)

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
        batch_size: int = 3,
    ) -> None:
        self.client = client or OllamaClient()
        self.prompt_template = load_eval_generator_prompt(prompt_path)
        self.batch_size = batch_size

    def _save_checkpoint(self, path: Path, cases: Sequence[EvaluationCase]) -> None:
        """Persist accepted cases incrementally to support resuming."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "total_cases": len(cases),
                "cases": [c.to_dict() for c in cases],
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_checkpoint(
        self,
        path: Path,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
        requirements: Optional[Dict[str, Any]] = None,
    ) -> List[EvaluationCase]:
        """Load and validate previously accepted cases from checkpoint against documents and requirements."""
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            raw_cases = data.get("cases", [])
            valid_resumed: List[EvaluationCase] = []
            for c_dict in raw_cases:
                if isinstance(c_dict, dict):
                    case = EvaluationCase.from_dict(c_dict)
                    is_valid, reason = validate_evaluation_case(case, doc_lookup, seen_question_keys)
                    if is_valid:
                        q_key = normalize_question_key(case.question)
                        seen_question_keys.add(q_key)
                        valid_resumed.append(case)
                    else:
                        print(f"  [checkpoint] Skipped unverified checkpoint case '{case.id}': {reason}")
            if valid_resumed:
                print(f"  [resume] Resumed {len(valid_resumed)} previously accepted cases from '{path}'.")
            return valid_resumed
        except Exception:
            return []

    def generate_eval_set(
        self,
        documents: Sequence[Document],
        requirements: Dict[str, Any],
        target_counts: Optional[Dict[str, int]] = None,
        max_retries: int = 4,
        checkpoint_path: Optional[Union[str, Path]] = "eval_set_checkpoint.json",
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

        checkpoint_file = Path(checkpoint_path) if checkpoint_path else None
        seen_question_keys: Set[str] = set()
        collected_cases: List[EvaluationCase] = []

        if checkpoint_file:
            collected_cases = self._load_checkpoint(
                checkpoint_file, doc_lookup, seen_question_keys, requirements=requirements
            )

        # Generate category by category in small bounded batches of 2-3
        for category, target_count in targets.items():
            if target_count <= 0:
                continue

            category_cases: List[EvaluationCase] = [
                c for c in collected_cases if c.category == category
            ]

            if len(category_cases) >= target_count:
                print(f"  [{category}] Target already satisfied ({len(category_cases)}/{target_count}).")
                continue

            # -------------------------------------------------------------
            # Category 1: answerable (one document chunk at a time)
            # -------------------------------------------------------------
            if category == "answerable":
                doc_idx = 0
                retries_without_progress = 0
                max_doc_cycles = max_retries * max(len(documents), 1)

                while len(category_cases) < target_count and retries_without_progress < max_doc_cycles:
                    doc = documents[doc_idx % len(documents)]
                    doc_idx += 1
                    needed = target_count - len(category_cases)
                    batch_target = min(needed, self.batch_size)
                    doc_src = str(doc.metadata.get("source", doc.doc_id)).replace("\\", "/")

                    added = self._generate_single_doc_answerable_batch(
                        doc=doc,
                        doc_src=doc_src,
                        batch_target=batch_target,
                        category_cases=category_cases,
                        scope=scope,
                        must_answer=must_answer,
                        must_refuse=must_refuse,
                        doc_lookup=doc_lookup,
                        seen_question_keys=seen_question_keys,
                        collected_cases=collected_cases,
                        target_count=target_count,
                    )

                    if added > 0:
                        retries_without_progress = 0
                        if checkpoint_file:
                            all_current = [c for c in collected_cases if c.category != category] + category_cases
                            self._save_checkpoint(checkpoint_file, all_current)
                    else:
                        retries_without_progress += 1

            # -------------------------------------------------------------
            # Category 2: cross_document (two explicitly selected sources)
            # -------------------------------------------------------------
            elif category == "cross_document":
                distinct_src_list = sorted(list(distinct_source_files))
                pair_idx = 0
                retries_without_progress = 0
                max_pair_cycles = max_retries * max(len(distinct_src_list), 1)

                while len(category_cases) < target_count and retries_without_progress < max_pair_cycles:
                    src_a = distinct_src_list[pair_idx % len(distinct_src_list)]
                    src_b = distinct_src_list[(pair_idx + 1) % len(distinct_src_list)]
                    pair_idx += 1

                    docs_a = [d for d in documents if str(d.metadata.get("source", d.doc_id)) == src_a]
                    docs_b = [d for d in documents if str(d.metadata.get("source", d.doc_id)) == src_b]

                    needed = target_count - len(category_cases)
                    batch_target = min(needed, self.batch_size)

                    added = self._generate_cross_doc_pair_batch(
                        src_a=src_a,
                        docs_a=docs_a,
                        src_b=src_b,
                        docs_b=docs_b,
                        batch_target=batch_target,
                        category_cases=category_cases,
                        scope=scope,
                        must_answer=must_answer,
                        must_refuse=must_refuse,
                        doc_lookup=doc_lookup,
                        seen_question_keys=seen_question_keys,
                        collected_cases=collected_cases,
                        target_count=target_count,
                    )

                    if added > 0:
                        retries_without_progress = 0
                        if checkpoint_file:
                            all_current = [c for c in collected_cases if c.category != category] + category_cases
                            self._save_checkpoint(checkpoint_file, all_current)
                    else:
                        retries_without_progress += 1
                        # Generic deterministic fallback before expensive repeated cross-document retries
                        candidate_ans = [c for c in collected_cases if c.category == "answerable"]
                        composed = self._compose_cross_document_fallback(
                            category_cases=category_cases,
                            candidate_answerable_cases=candidate_ans,
                            target_count=target_count,
                            doc_lookup=doc_lookup,
                            seen_question_keys=seen_question_keys,
                        )
                        if composed > 0:
                            retries_without_progress = 0
                            if checkpoint_file:
                                all_current = [c for c in collected_cases if c.category != category] + category_cases
                                self._save_checkpoint(checkpoint_file, all_current)

                # Ensure deterministic fallback runs if target is still unmet before concluding category
                if len(category_cases) < target_count:
                    candidate_ans = [c for c in collected_cases if c.category == "answerable"]
                    composed = self._compose_cross_document_fallback(
                        category_cases=category_cases,
                        candidate_answerable_cases=candidate_ans,
                        target_count=target_count,
                        doc_lookup=doc_lookup,
                        seen_question_keys=seen_question_keys,
                    )
                    if composed > 0 and checkpoint_file:
                        all_current = [c for c in collected_cases if c.category != category] + category_cases
                        self._save_checkpoint(checkpoint_file, all_current)

            # -------------------------------------------------------------
            # Categories 3 & 4: out_of_scope and ambiguous
            # -------------------------------------------------------------
            else:
                retries = 0
                doc_context = format_documents_for_prompt(documents, category=category)

                while len(category_cases) < target_count and retries < max_retries:
                    needed = target_count - len(category_cases)
                    batch_target = min(needed, self.batch_size)

                    added = self._generate_general_category_batch(
                        category=category,
                        batch_target=batch_target,
                        category_cases=category_cases,
                        doc_context=doc_context,
                        scope=scope,
                        must_answer=must_answer,
                        must_refuse=must_refuse,
                        doc_lookup=doc_lookup,
                        seen_question_keys=seen_question_keys,
                        collected_cases=collected_cases,
                        target_count=target_count,
                    )

                    if added > 0:
                        retries = 0
                        if checkpoint_file:
                            all_current = [c for c in collected_cases if c.category != category] + category_cases
                            self._save_checkpoint(checkpoint_file, all_current)
                    else:
                        retries += 1

            if len(category_cases) < target_count:
                raise RuntimeError(
                    f"Failed to generate valid complete set for category '{category}': "
                    f"generated {len(category_cases)}/{target_count} after retries."
                )

            # Update collected cases for this category
            collected_cases = [c for c in collected_cases if c.category != category] + category_cases

        # Renumber IDs sequentially to ensure clean 1-based order
        for idx, case in enumerate(collected_cases, start=1):
            case.id = f"eval_{idx:03d}"

        if checkpoint_file:
            self._save_checkpoint(checkpoint_file, collected_cases)

        return collected_cases

    def _compose_cross_document_fallback(
        self,
        category_cases: List[EvaluationCase],
        candidate_answerable_cases: Sequence[EvaluationCase],
        target_count: int,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
    ) -> int:
        """Deterministically combine compatible validated answerable cases from distinct documents into cross-document cases."""
        if len(category_cases) >= target_count:
            return 0

        # Group answerable cases by primary source
        cases_by_source: Dict[str, List[EvaluationCase]] = {}
        for case in candidate_answerable_cases:
            if case.category == "answerable" and case.expected_sources and case.expected_evidence:
                primary_src = str(case.expected_sources[0])
                cases_by_source.setdefault(primary_src, []).append(case)

        distinct_sources = sorted(cases_by_source.keys())
        if len(distinct_sources) < 2:
            return 0

        added_count = 0
        wh_words = {
            "what", "how", "where", "when", "which", "who", "whom", "whose", "why",
            "does", "do", "is", "are", "can", "could", "should", "will", "would"
        }

        # Iterate over distinct source pairs
        for i in range(len(distinct_sources)):
            src_a = distinct_sources[i]
            for j in range(i + 1, len(distinct_sources)):
                src_b = distinct_sources[j]

                list_a = cases_by_source[src_a]
                list_b = cases_by_source[src_b]

                for case_a in list_a:
                    for case_b in list_b:
                        if len(category_cases) >= target_count:
                            return added_count

                        q_a = case_a.question.strip().rstrip("?. ")
                        q_b = case_b.question.strip().rstrip("?. ")

                        if not q_a or not q_b:
                            continue

                        # Avoid repeating the exact same pair of parent cases
                        already_used = False
                        for existing in category_cases:
                            meta = getattr(existing, "composition_metadata", None)
                            if meta and isinstance(meta, dict):
                                parents = set(meta.get("parent_case_ids", []))
                                if case_a.id in parents and case_b.id in parents:
                                    already_used = True
                                    break
                        if already_used:
                            continue

                        words_b = q_b.split()
                        if words_b and words_b[0].lower() in wh_words and not words_b[0].isupper():
                            q_b_formatted = words_b[0].lower() + q_b[len(words_b[0]):]
                        else:
                            q_b_formatted = q_b

                        combined_q = f"{q_a}, and {q_b_formatted}?"

                        # Combine facts preserving order and deduplicating
                        combined_facts: List[str] = []
                        for f in (case_a.expected_key_facts + case_b.expected_key_facts):
                            f_str = str(f).strip()
                            if f_str and f_str not in combined_facts:
                                combined_facts.append(f_str)

                        # Combine evidence preserving order and deduplicating
                        combined_evidence: List[Dict[str, str]] = []
                        seen_excerpts: Set[str] = set()
                        for ev in (case_a.expected_evidence + case_b.expected_evidence):
                            exc = str(ev.get("excerpt", "")).strip()
                            src = str(ev.get("source", "")).strip()
                            if exc and exc not in seen_excerpts:
                                seen_excerpts.add(exc)
                                combined_evidence.append({"source": src, "excerpt": exc})

                        combined_sources = sorted(list(set(case_a.expected_sources + case_b.expected_sources)))
                        if len(combined_sources) < 2:
                            continue

                        candidate_case = EvaluationCase(
                            id=f"eval_comp_{len(category_cases) + 1}",
                            category="cross_document",
                            question=combined_q,
                            expected_status="answered",
                            expected_key_facts=combined_facts,
                            expected_sources=combined_sources,
                            expected_evidence=combined_evidence,
                            composition_metadata={
                                "method": "deterministic_pair_composition",
                                "parent_case_ids": [case_a.id, case_b.id],
                                "parent_sources": combined_sources,
                            },
                        )

                        is_valid, reason = validate_evaluation_case(candidate_case, doc_lookup, seen_question_keys)
                        if is_valid:
                            q_key = normalize_question_key(candidate_case.question)
                            seen_question_keys.add(q_key)
                            category_cases.append(candidate_case)
                            added_count += 1
                            print(f"  [cross_document] Fallback composed ({len(category_cases)}/{target_count}): {candidate_case.question[:65]}...")
                        else:
                            continue

        return added_count

    def _generate_single_doc_answerable_batch(
        self,
        doc: Document,
        doc_src: str,
        batch_target: int,
        category_cases: List[EvaluationCase],
        scope: str,
        must_answer: str,
        must_refuse: str,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
        collected_cases: List[EvaluationCase],
        target_count: int,
    ) -> int:
        """Generate answerable cases from one source document chunk at a time."""
        if category_cases:
            prev_q_str = "\n".join(f"- {c.question}" for c in category_cases)
        else:
            prev_q_str = "None yet."

        doc_context = f"--- Document Source: {doc_src} ---\n{doc.text.strip()}"

        prompt = (
            self.prompt_template.replace("{scope}", scope)
            .replace("{must_answer_topics}", must_answer)
            .replace("{must_refuse_topics}", must_refuse)
            .replace("{document_content}", doc_context)
            .replace("{category}", "answerable")
            .replace("{target_count}", str(batch_target))
            .replace("{already_generated_questions}", prev_q_str)
        )

        system_prompt = (
            "You are a strict, objective AI evaluation benchmark designer. "
            "Output ONLY valid JSON adhering strictly to the schema. "
            "Never invent or hallucinate document citations or text."
        )

        try:
            res = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as e:
            print(f"  [answerable] LLM generation error: {e}")
            return 0

        raw_cases = res.get("cases", [])
        if not isinstance(raw_cases, list):
            return 0

        added_in_turn = 0
        for c_dict in raw_cases:
            if not isinstance(c_dict, dict) or len(category_cases) >= target_count:
                break

            q_text = str(c_dict.get("question", "")).strip()
            raw_facts = c_dict.get("expected_key_facts", [])
            facts = [str(f) for f in raw_facts if str(f).strip()] if isinstance(raw_facts, list) else ([str(raw_facts).strip()] if str(raw_facts).strip() else [])

            evidence = c_dict.get("expected_evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            # Programmatically assign source from the supplied single document
            for ev in evidence:
                if isinstance(ev, dict):
                    if not ev.get("source"):
                        ev["source"] = doc_src
                    elif str(ev.get("source")).strip() not in doc_lookup:
                        ev["source"] = doc_src

            temp_case = EvaluationCase(
                id=f"eval_{len(collected_cases) + len(category_cases) + 1:03d}",
                category="answerable",
                question=q_text,
                expected_status="answered",
                expected_key_facts=facts,
                expected_sources=[doc_src],
                expected_evidence=evidence,
            )

            is_valid, reason = validate_evaluation_case(temp_case, doc_lookup, seen_question_keys)
            if is_valid:
                q_key = normalize_question_key(temp_case.question)
                seen_question_keys.add(q_key)
                category_cases.append(temp_case)
                added_in_turn += 1
                print(f"  [answerable] Accepted ({len(category_cases)}/{target_count}): {temp_case.question[:65]}")
            else:
                print(f"  [answerable] Rejected: {reason}")

        return added_in_turn

    def _generate_cross_doc_pair_batch(
        self,
        src_a: str,
        docs_a: List[Document],
        src_b: str,
        docs_b: List[Document],
        batch_target: int,
        category_cases: List[EvaluationCase],
        scope: str,
        must_answer: str,
        must_refuse: str,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
        collected_cases: List[EvaluationCase],
        target_count: int,
    ) -> int:
        """Generate cross-document cases using two explicitly selected distinct sources."""
        if category_cases:
            prev_q_str = "\n".join(f"- {c.question}" for c in category_cases)
        else:
            prev_q_str = "None yet."

        text_a = "\n".join(d.text.strip()[:1000] for d in docs_a)
        text_b = "\n".join(d.text.strip()[:1000] for d in docs_b)

        doc_context = (
            f"--- Source File A: {src_a} ---\n{text_a}\n\n"
            f"--- Source File B: {src_b} ---\n{text_b}"
        )

        prompt = (
            self.prompt_template.replace("{scope}", scope)
            .replace("{must_answer_topics}", must_answer)
            .replace("{must_refuse_topics}", must_refuse)
            .replace("{document_content}", doc_context)
            .replace("{category}", "cross_document")
            .replace("{target_count}", str(batch_target))
            .replace("{already_generated_questions}", prev_q_str)
        )

        system_prompt = (
            "You are a strict, objective AI evaluation benchmark designer. "
            "Generate questions requiring facts from BOTH Source File A and Source File B. "
            "Output ONLY valid JSON adhering strictly to the schema."
        )

        try:
            res = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as e:
            print(f"  [cross_document] LLM generation error: {e}")
            return 0

        raw_cases = res.get("cases", [])
        if not isinstance(raw_cases, list):
            return 0

        combined_text_a = " ".join(d.text for d in docs_a)
        combined_text_b = " ".join(d.text for d in docs_b)

        added_in_turn = 0
        for c_dict in raw_cases:
            if not isinstance(c_dict, dict) or len(category_cases) >= target_count:
                break

            q_text = str(c_dict.get("question", "")).strip()
            raw_facts = c_dict.get("expected_key_facts", [])
            facts = [str(f) for f in raw_facts if str(f).strip()] if isinstance(raw_facts, list) else ([str(raw_facts).strip()] if str(raw_facts).strip() else [])

            evidence = c_dict.get("expected_evidence", [])
            if not isinstance(evidence, list):
                evidence = []

            # Programmatically assign sources by checking which of the two documents contains the excerpt
            for ev in evidence:
                if isinstance(ev, dict):
                    excerpt = str(ev.get("excerpt", "")).strip()
                    if validate_excerpt_in_document(excerpt, combined_text_a):
                        ev["source"] = src_a
                    elif validate_excerpt_in_document(excerpt, combined_text_b):
                        ev["source"] = src_b

            temp_case = EvaluationCase(
                id=f"eval_{len(collected_cases) + len(category_cases) + 1:03d}",
                category="cross_document",
                question=q_text,
                expected_status="answered",
                expected_key_facts=facts,
                expected_sources=[src_a, src_b],
                expected_evidence=evidence,
            )

            is_valid, reason = validate_evaluation_case(temp_case, doc_lookup, seen_question_keys)
            if is_valid:
                q_key = normalize_question_key(temp_case.question)
                seen_question_keys.add(q_key)
                category_cases.append(temp_case)
                added_in_turn += 1
                print(f"  [cross_document] Accepted ({len(category_cases)}/{target_count}): {temp_case.question[:65]}")
            else:
                print(f"  [cross_document] Rejected: {reason}")

        return added_in_turn

    def _generate_general_category_batch(
        self,
        category: str,
        batch_target: int,
        category_cases: List[EvaluationCase],
        doc_context: str,
        scope: str,
        must_answer: str,
        must_refuse: str,
        doc_lookup: Dict[str, Document],
        seen_question_keys: Set[str],
        collected_cases: List[EvaluationCase],
        target_count: int,
    ) -> int:
        """Generate out_of_scope and ambiguous evaluation cases in small batches."""
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
            .replace("{target_count}", str(batch_target))
            .replace("{already_generated_questions}", prev_q_str)
        )

        system_prompt = (
            "You are a strict, objective AI evaluation benchmark designer. "
            "Output ONLY valid JSON adhering strictly to the required schema."
        )

        try:
            res = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as e:
            print(f"  [{category}] LLM generation error: {e}")
            return 0

        raw_cases = res.get("cases", [])
        if not isinstance(raw_cases, list):
            return 0

        added_in_turn = 0
        for c_dict in raw_cases:
            if not isinstance(c_dict, dict) or len(category_cases) >= target_count:
                break

            q_text = str(c_dict.get("question", "")).strip()
            raw_facts = c_dict.get("expected_key_facts", [])
            facts = [str(f) for f in raw_facts if str(f).strip()] if isinstance(raw_facts, list) else ([str(raw_facts).strip()] if str(raw_facts).strip() else [])

            temp_case = EvaluationCase(
                id=f"eval_{len(collected_cases) + len(category_cases) + 1:03d}",
                category=category,
                question=q_text,
                expected_status=str(c_dict.get("expected_status", CATEGORY_EXPECTED_STATUS[category])).strip(),
                expected_key_facts=facts,
                expected_sources=[],
                expected_evidence=[],
            )

            is_valid, reason = validate_evaluation_case(temp_case, doc_lookup, seen_question_keys)
            if is_valid:
                q_key = normalize_question_key(temp_case.question)
                seen_question_keys.add(q_key)
                category_cases.append(temp_case)
                added_in_turn += 1
                print(f"  [{category}] Accepted ({len(category_cases)}/{target_count}): {temp_case.question[:65]}")
            else:
                print(f"  [{category}] Rejected: {reason}")

        return added_in_turn

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
