import unittest

from research_assistant.evaluation import precision_recall_at_k, rouge_l, rouge_n, rouge_scores


class EvaluationTests(unittest.TestCase):
    def test_identical_text_has_perfect_rouge(self):
        text = "The system retrieves relevant scientific evidence."
        self.assertEqual(rouge_scores(text, text), {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0})

    def test_empty_candidate_has_zero_rouge(self):
        self.assertEqual(rouge_n("", "A reference summary"), 0.0)
        self.assertEqual(rouge_l("", "A reference summary"), 0.0)

    def test_precision_and_recall_at_k(self):
        scores = precision_recall_at_k(["c1", "c2", "c3"], {"c1", "c4"}, 3)
        self.assertAlmostEqual(scores["precision_at_k"], 1 / 3)
        self.assertEqual(scores["recall_at_k"], 0.5)


if __name__ == "__main__":
    unittest.main()
