import unittest

from research_assistant.chunking import chunk_document
from research_assistant.models import Document, PageText


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.document = Document("Paper", [
            PageText(1, "Abstract\n" + "Scientific document retrieval requires reliable evidence. " * 30),
            PageText(2, "Methods\nWe use a ranking model to identify relevant passages. " * 20),
        ])

    def test_chunks_have_bounded_size_and_page_sources(self):
        chunks = chunk_document(self.document, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk.text) <= 301 for chunk in chunks))
        self.assertTrue(all(len(chunk.page_numbers) == 1 for chunk in chunks))
        self.assertTrue(all(chunk.page_numbers == (1,) for chunk in chunks[:2]))

    def test_section_headings_are_preserved(self):
        chunks = chunk_document(self.document)
        self.assertEqual(chunks[0].section, "Abstract")
        self.assertEqual(chunks[-1].section, "Methods")

    def test_invalid_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_document(self.document, chunk_size=300, overlap=150)


if __name__ == "__main__":
    unittest.main()
