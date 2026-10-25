import unittest

from research_assistant.models import Chunk
from research_assistant.retrieval import (
    BM25Retriever,
    HybridRetriever,
    TfidfRetriever,
    build_retriever,
    tokenize,
)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            Chunk("c1", "The transformer model uses attention for language understanding.", (1,)),
            Chunk("c2", "Experiments used the SQuAD question answering dataset and accuracy metrics.", (2,)),
            Chunk("c3", "The soup recipe requires carrots, potatoes, and fresh basil.", (3,)),
        ]

    def test_bm25_finds_relevant_dataset_passage(self):
        results = BM25Retriever(self.chunks).retrieve("Which dataset was used?", 2)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_tfidf_finds_semantic_query_terms(self):
        results = TfidfRetriever(self.chunks).retrieve("attention transformer", 1)
        self.assertEqual(results[0].chunk.chunk_id, "c1")

    def test_plural_normalization_supports_lexical_matching(self):
        self.assertEqual(
            tokenize("datasets studies classes class"),
            ["dataset", "study", "class", "class"],
        )
        results = BM25Retriever(self.chunks).retrieve("datasets", 1)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_zero_match_returns_no_evidence(self):
        self.assertEqual(BM25Retriever(self.chunks).retrieve("unmatched zebra", 3), [])

    def test_hybrid_fuses_rankings(self):
        hybrid = HybridRetriever([BM25Retriever(self.chunks), TfidfRetriever(self.chunks)])
        results = hybrid.retrieve("SQuAD dataset", 2)
        self.assertEqual(results[0].chunk.chunk_id, "c2")

    def test_unknown_method_is_rejected(self):
        with self.assertRaises(ValueError):
            build_retriever(self.chunks, "unknown")


if __name__ == "__main__":
    unittest.main()
