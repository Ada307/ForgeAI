"""Groundability Guard module for ForgeAI.

Verifies must-answer topics against retrieved document evidence, validates citations,
records knowledge gaps, and routes unsupported topics to must_refuse_topics.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Union

from forge.llm import OllamaClient
from forge.retrieval import RetrievalResult, TFIDFRetriever


DEFAULT_GROUNDABILITY_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "groundability.txt"
)


def load_groundability_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    """Load verification prompt template from prompts/groundability.txt."""
    path = Path(prompt_path) if prompt_path else DEFAULT_GROUNDABILITY_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Groundability prompt file not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines: List[str] = []
    for line in content.splitlines():
        if line.strip().startswith("[") and line.strip().endswith("]"):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def normalize_text_for_matching(text: str) -> str:
    """Normalize text for excerpt substring matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def validate_citation_excerpt(excerpt: str, chunks: Sequence[RetrievalResult]) -> bool:
    """Check if the cited excerpt genuinely appears in any retrieved chunk text."""
    clean_excerpt = normalize_text_for_matching(excerpt)
    if not clean_excerpt:
        return False

    for chunk in chunks:
        clean_chunk = normalize_text_for_matching(chunk.text)
        if clean_excerpt in clean_chunk:
            return True

    # If excerpt has multiple words, allow prefix/suffix partial match
    excerpt_words = clean_excerpt.split()
    if len(excerpt_words) >= 4:
        sub_phrase = " ".join(excerpt_words[:4])
        for chunk in chunks:
            if sub_phrase in normalize_text_for_matching(chunk.text):
                return True

    return False


def validate_citation(citation: Dict[str, str], chunks: Sequence[RetrievalResult]) -> bool:
    """Validate that citation excerpt appears in its specific claimed source and chunk ID."""
    claimed_source = str(citation.get("source", "")).strip()
    claimed_chunk_id = str(citation.get("chunk_id", "")).strip()
    excerpt = str(citation.get("excerpt", "")).strip()

    if not excerpt:
        return False

    clean_excerpt = normalize_text_for_matching(excerpt)

    # Locate chunk matching BOTH claimed source and chunk_id
    target_chunk: Optional[RetrievalResult] = None
    for chunk in chunks:
        chunk_src = str(chunk.metadata.get("source", "")).strip()
        id_match = (claimed_chunk_id == chunk.chunk_id) if claimed_chunk_id else True
        src_match = (
            (claimed_source == chunk_src)
            or (claimed_source and claimed_source in chunk_src)
            or (chunk_src and chunk_src in claimed_source)
        ) if claimed_source else True

        if id_match and src_match and (claimed_chunk_id or claimed_source):
            target_chunk = chunk
            break

    if target_chunk is None:
        return False

    clean_chunk = normalize_text_for_matching(target_chunk.text)
    if clean_excerpt in clean_chunk:
        return True

    excerpt_words = clean_excerpt.split()
    if len(excerpt_words) >= 4:
        sub_phrase = " ".join(excerpt_words[:4])
        if sub_phrase in clean_chunk:
            return True

    return False


@dataclass
class TopicVerification:
    """Verification outcome for a single topic."""
    topic: str
    status: str  # "supported", "partially_supported", "unsupported"
    reason: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    evidence_validated: bool = True
    supported_portion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "status": self.status,
            "supported_portion": self.supported_portion,
            "reason": self.reason,
            "citations": self.citations,
            "knowledge_gaps": self.knowledge_gaps,
            "evidence_validated": self.evidence_validated,
        }


class GroundabilityGuard:
    """Audits requirement topics against retrieved document evidence."""

    def __init__(
        self,
        retriever: TFIDFRetriever,
        client: Optional[OllamaClient] = None,
        prompt_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.retriever = retriever
        self.client = client or OllamaClient()
        self.prompt_template = load_groundability_prompt(prompt_path)

    def verify_topic(self, topic: str, top_k: int = 4) -> TopicVerification:
        """Verify evidence for a topic using retrieval and LLM evaluation."""
        retrieved_chunks = self.retriever.retrieve(
            query=topic,
            top_k=top_k,
            minimum_similarity_score=0.0,
        )

        # Never treat zero chunks or absence of evidence as supported
        if not retrieved_chunks:
            return TopicVerification(
                topic=topic,
                status="unsupported",
                reason="No matching document evidence found in corpus for this topic.",
                citations=[],
                knowledge_gaps=[f"Corpus lacks any documentation regarding '{topic}'."],
                evidence_validated=True,
            )

        evidence_text = "\n\n".join(
            f"[Chunk ID: {c.chunk_id} | Source: {c.metadata.get('source', 'unknown')} | Score: {c.score:.4f}]\n{c.text}"
            for c in retrieved_chunks
        )

        prompt = (
            self.prompt_template.replace("{topic}", topic)
            .replace("{evidence_chunks}", evidence_text)
        )
        system_prompt = (
            "You are a strict Groundability Guard. Verify factual evidence and output valid JSON."
        )

        try:
            response = self.client.generate_json(prompt=prompt, system_prompt=system_prompt)
        except Exception as exc:
            # Report uncertainty rather than false support
            return TopicVerification(
                topic=topic,
                status="unsupported",
                reason=f"Model evaluation failed or returned invalid JSON ({exc}); reporting uncertainty.",
                citations=[],
                knowledge_gaps=[f"Uncertainty: verification could not be completed for '{topic}'."],
                evidence_validated=False,
            )

        raw_status = str(response.get("status", "unsupported")).strip().lower()
        if raw_status not in {"supported", "partially_supported", "unsupported"}:
            raw_status = "unsupported"

        supported_portion = str(response.get("supported_portion", "")).strip()
        reason = str(response.get("reason", "No reason provided by verification model."))
        raw_citations = response.get("citations", [])
        raw_gaps = response.get("knowledge_gaps", [])

        citations: List[Dict[str, str]] = []
        if isinstance(raw_citations, list):
            for c in raw_citations:
                if isinstance(c, dict):
                    citations.append({
                        "source": str(c.get("source", "unknown")),
                        "chunk_id": str(c.get("chunk_id", "unknown")),
                        "excerpt": str(c.get("excerpt", "")),
                    })

        knowledge_gaps: List[str] = []
        if isinstance(raw_gaps, list):
            for g in raw_gaps:
                if isinstance(g, str) and g.strip():
                    knowledge_gaps.append(g.strip())

        # Validate cited excerpts against actual retrieved chunks
        evidence_validated = True
        if raw_status in {"supported", "partially_supported"}:
            if not citations:
                # Claimed support without citing any excerpt
                evidence_validated = False
                raw_status = "unsupported"
                reason = "Claimed support without providing verifiable citations or excerpts."
                knowledge_gaps.append("Missing verifiable source excerpts.")
            else:
                for cit in citations:
                    if not validate_citation(cit, retrieved_chunks):
                        evidence_validated = False
                        raw_status = "unsupported"
                        reason = (
                            f"Citation validation failed: excerpt {cit.get('excerpt', '')!r} "
                            f"not found in specific claimed source '{cit.get('source')}' "
                            f"and chunk ID '{cit.get('chunk_id')}'."
                        )
                        knowledge_gaps.append(f"Unverifiable citation for source '{cit.get('source')}' / '{cit.get('chunk_id')}'.")
                        break

        if raw_status == "unsupported" and not knowledge_gaps:
            knowledge_gaps.append(f"No factual support found for topic: '{topic}'.")

        return TopicVerification(
            topic=topic,
            status=raw_status,
            reason=reason,
            citations=citations,
            knowledge_gaps=knowledge_gaps,
            evidence_validated=evidence_validated,
            supported_portion=supported_portion,
        )

    def check_requirements(
        self,
        requirements: Dict[str, Any],
        top_k: int = 4,
    ) -> Dict[str, Any]:
        """Audit all must-answer topics, reclassifying unsupported topics to must_refuse."""
        enriched = dict(requirements)
        original_scope = str(requirements.get("scope", "")).strip()
        original_topics = list(requirements.get("must_answer_topics", []))
        existing_refuse = list(requirements.get("must_refuse_topics", []))

        verifications: List[TopicVerification] = []
        permitted_topics: List[str] = []
        updated_refuse: List[str] = list(existing_refuse)
        all_knowledge_gaps: List[str] = []

        for topic in original_topics:
            v = self.verify_topic(topic=topic, top_k=top_k)
            verifications.append(v)

            if v.status == "supported":
                permitted_topics.append(topic)
            elif v.status == "partially_supported":
                # For partially supported topics, permit only the explicitly supported portions
                supported_label = v.supported_portion if v.supported_portion else f"Verified aspects of {topic}"
                permitted_topics.append(supported_label)

                # Record unsupported portions as knowledge gaps and route to refusal
                gap_summary = "; ".join(v.knowledge_gaps) if v.knowledge_gaps else f"unsupported aspects of {topic}"
                updated_refuse.append(f"{topic} [Unsupported aspects: {gap_summary}]")
                all_knowledge_gaps.extend(v.knowledge_gaps or [f"Partial lack of evidence for {topic}"])
            else:
                # Unsupported topic is excluded from must_answer and routed to refusal
                updated_refuse.append(f"{topic} [Refusal: Unsupported by available documents]")
                all_knowledge_gaps.extend(v.knowledge_gaps)

        # Ensure any factual capability in final permitted scope is groundability-checked
        if permitted_topics:
            permitted_scope = (
                f"Assistance strictly constrained to verified capabilities: {'; '.join(permitted_topics)}. "
                "All unrequested, speculative, or unverified capabilities are excluded."
            )
        else:
            permitted_scope = (
                "No capabilities permitted: no requested topics were supported by verified document evidence."
            )

        enriched["original_scope"] = original_scope
        enriched["permitted_scope"] = permitted_scope
        enriched["scope"] = permitted_scope
        enriched["original_must_answer_topics"] = original_topics
        enriched["must_answer_topics"] = permitted_topics
        enriched["must_refuse_topics"] = updated_refuse
        enriched["groundability_report"] = [v.to_dict() for v in verifications]
        enriched["knowledge_gaps"] = all_knowledge_gaps

        return enriched


def save_requirements(requirements: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save enriched requirements dictionary as a JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(requirements, f, indent=2)
