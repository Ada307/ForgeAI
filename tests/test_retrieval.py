"""Tests for lightweight TF-IDF retrieval system (forge/retrieval.py)."""

import unittest
from forge.documents import Document
from forge.retrieval import (
    RetrieverConfig,
    TFIDFRetriever,
    chunk_documents,
    split_text_into_chunks,
)


class TestRetrievalSystem(unittest.TestCase):
    """Test suite for chunking and TFIDFRetriever."""

    def test_chunk_sizes_and_overlap(self) -> None:
        """Test splitting text into overlapping word chunks."""
        words = [f"word{i}" for i in range(100)]
        text = " ".join(words)

        chunks = split_text_into_chunks(text, chunk_size=40, chunk_overlap=10)
        # step = 40 - 10 = 30
        # Chunk 0: 0..40 (40 words)
        # Chunk 1: 30..70 (40 words)
        # Chunk 2: 60..100 (40 words)
        self.assertEqual(len(chunks), 3)

        chunk0_words = chunks[0].split()
        chunk1_words = chunks[1].split()
        chunk2_words = chunks[2].split()

        self.assertEqual(len(chunk0_words), 40)
        self.assertEqual(len(chunk1_words), 40)
        self.assertEqual(len(chunk2_words), 40)

        # Verify exact overlap between chunk0 and chunk1
        self.assertEqual(chunk0_words[-10:], chunk1_words[:10])
        # Verify exact overlap between chunk1 and chunk2
        self.assertEqual(chunk1_words[-10:], chunk2_words[:10])

    def test_stable_chunk_ids_and_metadata(self) -> None:
        """Test stable chunk identifiers and document metadata preservation."""
        doc = Document(
            doc_id="equipment/pump_manual.txt",
            text="First sentence about pumps. Second sentence about maintenance.",
            metadata={"source": "equipment/pump_manual.txt", "equipment_id": "P-401"},
        )
        chunks = chunk_documents([doc], chunk_size=4, chunk_overlap=1)
        self.assertGreater(len(chunks), 1)

        for idx, chunk in enumerate(chunks):
            self.assertEqual(chunk.chunk_id, f"equipment/pump_manual.txt:c{idx}")
            self.assertEqual(chunk.doc_id, "equipment/pump_manual.txt")
            self.assertEqual(chunk.metadata["source"], "equipment/pump_manual.txt")
            self.assertEqual(chunk.metadata["equipment_id"], "P-401")
            self.assertEqual(chunk.metadata["chunk_index"], idx)

    def test_relevant_query_ranking(self) -> None:
        """Test that TF-IDF cosine ranking places the most relevant document first."""
        docs = [
            Document(
                doc_id="turbine.txt",
                text="The steam turbine converts high pressure steam into rotational mechanical energy.",
                metadata={"source": "turbine.txt"},
            ),
            Document(
                doc_id="pump.txt",
                text="The centrifugal pump moves cooling fluid through the reactor heat exchanger.",
                metadata={"source": "pump.txt"},
            ),
            Document(
                doc_id="conveyor.txt",
                text="The belt conveyor transports bulk raw materials across the assembly facility.",
                metadata={"source": "conveyor.txt"},
            ),
        ]

        retriever = TFIDFRetriever(documents=docs)
        results = retriever.retrieve("centrifugal pump cooling fluid")

        self.assertGreaterEqual(len(results), 1)
        top_result = results[0]
        self.assertEqual(top_result.metadata["source"], "pump.txt")
        self.assertGreater(top_result.score, 0.0)
        self.assertIn("centrifugal pump", top_result.text)

    def test_configurable_top_k_and_similarity_threshold(self) -> None:
        """Test top_k limiting and minimum similarity score thresholding."""
        docs = [
            Document(
                doc_id=f"doc_{i}.txt",
                text=f"Hydraulic pressure line inspection procedure number {i}.",
                metadata={"source": f"doc_{i}.txt"},
            )
            for i in range(10)
        ]

        retriever = TFIDFRetriever(documents=docs, top_k=3, minimum_similarity_score=0.0)
        results = retriever.retrieve("hydraulic pressure inspection")
        self.assertEqual(len(results), 3)

        # Higher threshold filters out low-scoring chunks
        high_threshold_results = retriever.retrieve(
            "hydraulic pressure inspection", minimum_similarity_score=0.99
        )
        # Scores are lower than 0.99 because documents have extra words
        self.assertEqual(len(high_threshold_results), 0)

    def test_retrieval_across_multiple_documents(self) -> None:
        """Test evidence retrieval capturing relevant chunks from different source documents."""
        docs = [
            Document(
                doc_id="furnace_spec.txt",
                text="Blast furnace operates at extreme temperatures reaching 1600 degrees Celsius.",
                metadata={"source": "furnace_spec.txt"},
            ),
            Document(
                doc_id="safety_protocol.txt",
                text="High temperature protective equipment is required near the blast furnace.",
                metadata={"source": "safety_protocol.txt"},
            ),
            Document(
                doc_id="office_guidelines.txt",
                text="Employees should submit weekly time sheets every Friday afternoon.",
                metadata={"source": "office_guidelines.txt"},
            ),
        ]

        retriever = TFIDFRetriever(documents=docs, top_k=5)
        results = retriever.retrieve("blast furnace temperature")

        sources = [r.metadata["source"] for r in results]
        self.assertIn("furnace_spec.txt", sources)
        self.assertIn("safety_protocol.txt", sources)
        self.assertNotIn("office_guidelines.txt", sources)

    def test_empty_and_unrelated_queries(self) -> None:
        """Test that empty queries and queries without matching vocabulary return empty lists."""
        docs = [
            Document(
                doc_id="doc.txt",
                text="Standard operational protocol for chemical refining plant.",
                metadata={"source": "doc.txt"},
            )
        ]
        retriever = TFIDFRetriever(documents=docs)

        self.assertEqual(retriever.retrieve(""), [])
        self.assertEqual(retriever.retrieve("   "), [])
        self.assertEqual(retriever.retrieve("!@#$%^&*()"), [])
        self.assertEqual(retriever.retrieve("extraterrestrial spacecraft orbit"), [])

    def test_zero_similarity_not_returned(self) -> None:
        """Ensure zero-similarity chunks are excluded even when minimum_similarity_score is 0.0."""
        docs = [
            Document(
                doc_id="relevant.txt",
                text="Bearing lubrication oil viscosity check.",
                metadata={"source": "relevant.txt"},
            ),
            Document(
                doc_id="irrelevant.txt",
                text="Cafeteria lunch menu schedule for next week.",
                metadata={"source": "irrelevant.txt"},
            ),
        ]
        retriever = TFIDFRetriever(documents=docs, minimum_similarity_score=0.0)
        results = retriever.retrieve("lubrication oil")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["source"], "relevant.txt")

    def test_invalid_configuration_raises_errors(self) -> None:
        """Test parameter validation for RetrieverConfig."""
        with self.assertRaises(ValueError):
            RetrieverConfig(chunk_size=0)

        with self.assertRaises(ValueError):
            RetrieverConfig(chunk_overlap=-1)

        with self.assertRaises(ValueError):
            RetrieverConfig(chunk_size=100, chunk_overlap=100)

        with self.assertRaises(ValueError):
            RetrieverConfig(top_k=0)

        with self.assertRaises(ValueError):
            RetrieverConfig(minimum_similarity_score=-0.1)

        with self.assertRaises(ValueError):
            RetrieverConfig(minimum_similarity_score=1.5)

    def test_configuration_serialization(self) -> None:
        """Test that RetrieverConfig serializes to dict and restores accurately."""
        cfg = RetrieverConfig(
            chunk_size=200,
            chunk_overlap=40,
            top_k=6,
            minimum_similarity_score=0.15,
        )
        data = cfg.to_dict()
        restored = RetrieverConfig.from_dict(data)

        self.assertEqual(restored.chunk_size, 200)
        self.assertEqual(restored.chunk_overlap, 40)
        self.assertEqual(restored.top_k, 6)
        self.assertAlmostEqual(restored.minimum_similarity_score, 0.15)

    def test_deterministic_ordering_on_tied_scores(self) -> None:
        """Test that chunks with tied similarity scores are deterministically ordered by chunk_id."""
        docs = [
            Document(doc_id="doc_b.txt", text="identical keyword match", metadata={"source": "doc_b.txt"}),
            Document(doc_id="doc_a.txt", text="identical keyword match", metadata={"source": "doc_a.txt"}),
        ]
        retriever = TFIDFRetriever(documents=docs, top_k=2)
        results = retriever.retrieve("identical match")

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0].score, results[1].score)
        # Should be ordered by chunk_id alphabetically ("doc_a.txt:c0" before "doc_b.txt:c0")
        self.assertEqual(results[0].chunk_id, "doc_a.txt:c0")
        self.assertEqual(results[1].chunk_id, "doc_b.txt:c0")

    def test_split_text_into_chunks_validation(self) -> None:
        """Test parameter validation inside split_text_into_chunks to prevent infinite loops."""
        with self.assertRaises(ValueError):
            split_text_into_chunks("some text", chunk_size=0, chunk_overlap=0)

        with self.assertRaises(ValueError):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=-1)

        with self.assertRaises(ValueError):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=10)

        with self.assertRaises(ValueError):
            split_text_into_chunks("some text", chunk_size=10, chunk_overlap=15)

    def test_update_config_rebuilds_index_on_chunking_changes_only(self) -> None:
        """Test that update_config automatically rebuilds index only when chunking params change."""
        doc = Document(
            doc_id="doc1.txt",
            text=" ".join(f"word{i}" for i in range(100)),
            metadata={"source": "doc1.txt"},
        )
        retriever = TFIDFRetriever(documents=[doc], chunk_size=50, chunk_overlap=10)
        initial_chunk_count = len(retriever.chunks)
        self.assertEqual(initial_chunk_count, 3)

        # Changing top_k or similarity threshold should NOT rebuild
        retriever.update_config(top_k=10, minimum_similarity_score=0.2)
        self.assertEqual(len(retriever.chunks), initial_chunk_count)

        # Changing chunk_size should rebuild index automatically
        retriever.update_config(chunk_size=20, chunk_overlap=5)
        self.assertGreater(len(retriever.chunks), initial_chunk_count)


if __name__ == "__main__":
    unittest.main()
