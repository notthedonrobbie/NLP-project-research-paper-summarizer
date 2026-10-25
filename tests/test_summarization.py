import unittest

from research_assistant.models import Document, PageText
from research_assistant.summarization import SUMMARY_FIELDS, lead_summary, summarize_document, textrank_summary


class SummarizationTests(unittest.TestCase):
    def test_textrank_returns_original_sentences(self):
        text = (
            "Scientific papers contain dense technical information for specialist readers. "
            "This study proposes a retrieval method for grounded question answering systems. "
            "The evaluation uses twenty scientific papers from an open access collection. "
            "The proposed system improves evidence retrieval across the evaluation questions."
        )
        summary = textrank_summary(text, max_sentences=2)
        for sentence in summary.split(". "):
            self.assertIn(sentence.rstrip("."), text)

    def test_lead_baseline_selects_opening_sentences(self):
        text = (
            "The opening sentence introduces the central research problem clearly. "
            "A second sentence explains the proposed scientific method in detail. "
            "A final sentence presents experimental findings and their significance."
        )
        self.assertEqual(
            lead_summary(text, max_sentences=2),
            "The opening sentence introduces the central research problem clearly. "
            "A second sentence explains the proposed scientific method in detail.",
        )

    def test_structured_summary_contains_required_fields(self):
        document = Document("Paper", [PageText(1, (
            "Abstract\nThis paper addresses slow scientific literature reviews for researchers. "
            "We propose a retrieval augmented generation method for document question answering. "
            "The experiments use twenty papers and measure retrieval accuracy. "
            "The results show that the proposed method improves evidence identification. "
            "The limitations include extraction errors in complex scientific documents."
        ))])
        summary = summarize_document(document, max_sentences=3)
        self.assertEqual(set(summary.sections), set(SUMMARY_FIELDS))
        self.assertTrue(summary.overview)
        self.assertEqual(summary.method, "textrank")


if __name__ == "__main__":
    unittest.main()
