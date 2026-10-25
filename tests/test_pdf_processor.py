import unittest

from research_assistant.pdf_processor import clean_text, extract_pdf
from tests.helpers import make_pdf


class PdfProcessorTests(unittest.TestCase):
    def test_clean_text_preserves_words_and_removes_artifacts(self):
        self.assertEqual(clean_text("multi-\ncolumn\x00  text"), "multicolumn text")

    def test_extracts_pages_title_and_repeated_headers(self):
        data = make_pdf([
            ["Conference Header", "Abstract", "This study addresses document retrieval for scientific papers."],
            ["Conference Header", "Methods", "The proposed method uses lexical retrieval and ranked evidence passages."],
            ["Conference Header", "Results", "The system achieved eighty percent accuracy on the evaluation dataset."],
        ])
        document = extract_pdf(data, title="Test paper")
        self.assertEqual(document.title, "Test paper")
        self.assertEqual(document.page_count, 3)
        self.assertEqual(document.pages[1].page_number, 2)
        self.assertIn("lexical retrieval", document.text)
        self.assertNotIn("Conference Header", document.text)


if __name__ == "__main__":
    unittest.main()
