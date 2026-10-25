import unittest

from research_assistant.models import Chunk
from research_assistant.rag import INSUFFICIENT_ANSWER, answer_question
from research_assistant.retrieval import BM25Retriever


class RagTests(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever([
            Chunk("c1", "The experiments used the SQuAD dataset for question answering evaluation.", (4,), "Experiments"),
            Chunk("c2", "The proposed model achieved an accuracy of eighty percent on the test set.", (7,), "Results"),
        ])

    def test_answer_contains_source_and_page_citation(self):
        answer = answer_question("What dataset was used?", self.retriever)
        self.assertFalse(answer.insufficient_evidence)
        self.assertIn("SQuAD", answer.text)
        self.assertIn("[S1]", answer.text)
        self.assertEqual(answer.citations[0].chunk.page_numbers, (4,))

    def test_unsupported_question_acknowledges_insufficient_evidence(self):
        answer = answer_question("Who won the football match?", self.retriever)
        self.assertTrue(answer.insufficient_evidence)
        self.assertEqual(answer.text, INSUFFICIENT_ANSWER)

    def test_named_term_absent_from_evidence_causes_abstention(self):
        answer = answer_question(
            "What ImageNet classification accuracy did the model achieve?", self.retriever
        )
        self.assertTrue(answer.insufficient_evidence)
        self.assertEqual(answer.text, INSUFFICIENT_ANSWER)
        self.assertEqual(answer.citations, [])

    def test_empty_question_is_rejected(self):
        with self.assertRaises(ValueError):
            answer_question("", self.retriever)


if __name__ == "__main__":
    unittest.main()
