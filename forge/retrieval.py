"""Lightweight, self-contained TF-IDF retrieval system for ForgeAI.

Splits documents into overlapping word-based chunks, builds a reusable TF-IDF index,
and computes cosine similarity for ranked retrieval using Python standard library only.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence
from forge.documents import Document


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class RetrieverConfig:
    """Serializable configuration for chunking and retrieval parameters."""
    chunk_size: int = 180
    chunk_overlap: int = 30
    top_k: int = 4
    minimum_similarity_score: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration constraints."""
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got: {self.chunk_size}")
        if not isinstance(self.chunk_overlap, int) or self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be a non-negative integer, got: {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got: {self.top_k}")
        if not isinstance(self.minimum_similarity_score, (int, float)) or self.minimum_similarity_score < 0.0:
            raise ValueError(
                f"minimum_similarity_score must be a non-negative float, got: {self.minimum_similarity_score}"
            )
        if self.minimum_similarity_score > 1.0:
            raise ValueError(
                f"minimum_similarity_score cannot exceed 1.0, got: {self.minimum_similarity_score}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "minimum_similarity_score": float(self.minimum_similarity_score),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrieverConfig":
        """Instantiate RetrieverConfig from a dictionary."""
        return cls(
            chunk_size=data.get("chunk_size", 180),
            chunk_overlap=data.get("chunk_overlap", 30),
            top_k=data.get("top_k", 4),
            minimum_similarity_score=data.get("minimum_similarity_score", 0.0),
        )


@dataclass
class Chunk:
    """A text chunk with associated document metadata and stable identifier."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Ranked retrieval result item."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    score: float

    @property
    def similarity_score(self) -> float:
        return self.score


def split_text_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping word-based chunks.

    Words are preserved and chunk boundaries step forward by (chunk_size - chunk_overlap).
    """
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be a non-negative integer, got: {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    words = text.split()
    if not words:
        return []

    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(
    documents: Sequence[Document],
    chunk_size: int = 180,
    chunk_overlap: int = 30,
) -> List[Chunk]:
    """Chunk documents while preserving document boundaries, metadata, and column names."""
    all_chunks: List[Chunk] = []

    for doc in documents:
        text_chunks = split_text_into_chunks(doc.text, chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc.doc_id}:c{idx}"
            chunk_metadata = dict(doc.metadata)
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["doc_id"] = doc.doc_id

            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    return all_chunks


class TFIDFRetriever:
    """TF-IDF retrieval system using cosine similarity and precomputed vector index."""

    def __init__(
        self,
        config: Optional[RetrieverConfig] = None,
        chunk_size: int = 180,
        chunk_overlap: int = 30,
        top_k: int = 4,
        minimum_similarity_score: float = 0.0,
        documents: Optional[Sequence[Document]] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = RetrieverConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k,
                minimum_similarity_score=minimum_similarity_score,
            )

        self.chunks: List[Chunk] = []
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.chunk_vectors: List[Dict[str, float]] = []
        self.chunk_norms: List[float] = []
        self._indexed_documents: List[Document] = []

        if documents:
            self.index_documents(documents)

    def get_config(self) -> Dict[str, Any]:
        """Return serializable retriever configuration."""
        return self.config.to_dict()

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters with validation.

        Automatically rebuilds index when chunk_size or chunk_overlap changes.
        Does not rebuild for top_k or similarity threshold changes.
        """
        old_chunk_size = self.config.chunk_size
        old_chunk_overlap = self.config.chunk_overlap

        current = self.config.to_dict()
        current.update(kwargs)
        new_config = RetrieverConfig.from_dict(current)
        new_config.validate()
        self.config = new_config

        if (
            new_config.chunk_size != old_chunk_size
            or new_config.chunk_overlap != old_chunk_overlap
        ):
            if self._indexed_documents:
                self.index_documents(self._indexed_documents)

    def index_documents(self, documents: Sequence[Document]) -> None:
        """Build the TF-IDF index once over the provided documents."""
        self._indexed_documents = list(documents)
        self.chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

        n_chunks = len(self.chunks)
        if n_chunks == 0:
            self.vocabulary = {}
            self.idf = {}
            self.chunk_vectors = []
            self.chunk_norms = []
            return

        # 1. Document frequency (DF) per term across chunks
        doc_freq: Counter[str] = Counter()
        chunk_token_counts: List[Counter[str]] = []
        chunk_total_tokens: List[int] = []

        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            chunk_token_counts.append(counts)
            chunk_total_tokens.append(len(tokens))
            for term in counts:
                doc_freq[term] += 1

        # 2. Vocabulary and Smoothed IDF
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(doc_freq.keys()))}
        self.idf = {}
        for term, df in doc_freq.items():
            # Standard smoothed IDF: ln((N + 1) / (DF + 1)) + 1.0
            self.idf[term] = math.log((n_chunks + 1.0) / (df + 1.0)) + 1.0

        # 3. Precompute chunk TF-IDF vectors and L2 norms
        self.chunk_vectors = []
        self.chunk_norms = []

        for i, counts in enumerate(chunk_token_counts):
            total_tokens = chunk_total_tokens[i]
            vector: Dict[str, float] = {}
            sum_sq = 0.0

            if total_tokens > 0:
                for term, count in counts.items():
                    tf = count / total_tokens
                    weight = tf * self.idf[term]
                    vector[term] = weight
                    sum_sq += weight * weight

            norm = math.sqrt(sum_sq)
            self.chunk_vectors.append(vector)
            self.chunk_norms.append(norm)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        minimum_similarity_score: Optional[float] = None,
    ) -> List[RetrievalResult]:
        """Rank and return relevant chunks for a query using cosine similarity.

        Returns empty list for queries with no matching terms or zero similarity.
        Ties are broken deterministically by chunk_id.
        """
        k = top_k if top_k is not None else self.config.top_k
        min_score = (
            minimum_similarity_score
            if minimum_similarity_score is not None
            else self.config.minimum_similarity_score
        )

        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        # Build query TF-IDF vector
        query_counts = Counter(query_tokens)
        query_total = len(query_tokens)
        query_vector: Dict[str, float] = {}
        query_sum_sq = 0.0

        for term, count in query_counts.items():
            if term in self.idf:
                tf = count / query_total
                weight = tf * self.idf[term]
                query_vector[term] = weight
                query_sum_sq += weight * weight

        query_norm = math.sqrt(query_sum_sq)
        if query_norm == 0.0:
            return []

        # Compute cosine similarity for each chunk
        results: List[RetrievalResult] = []

        for idx, chunk in enumerate(self.chunks):
            chunk_norm = self.chunk_norms[idx]
            if chunk_norm == 0.0:
                continue

            chunk_vec = self.chunk_vectors[idx]

            # Dot product over intersection of terms
            dot_product = 0.0
            for term, q_weight in query_vector.items():
                if term in chunk_vec:
                    dot_product += q_weight * chunk_vec[term]

            if dot_product <= 0.0:
                continue

            sim = dot_product / (query_norm * chunk_norm)

            # Strict non-zero and threshold check
            if sim > 0.0 and sim >= min_score:
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=sim,
                    )
                )

        # Deterministic sorting: highest score first, ties broken by chunk_id
        results.sort(key=lambda r: (-r.score, r.chunk_id))

        return results[:k]
