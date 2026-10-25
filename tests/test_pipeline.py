import unittest

from research_assistant import ResearchAssistant
from tests.helpers import make_pdf


class PipelineTests(unittest.TestCase):
    def test_end_to_end_offline_pdf_workflow(self):
        data = make_pdf([[
            "Abstract",
            "This paper addresses unsupported answers in scientific document assistants.",
            "We propose a retrieval method that grounds responses in source passages.",
            "The evaluation uses the SQuAD dataset and twenty scientific papers.",
            "The proposed system achieved eighty percent accuracy in the experiments.",
        ]])
        assistant = ResearchAssistant.from_pdf(data)
        summary = assistant.summarize(max_sentences=3)
        answer = assistant.ask("Which dataset was used?")
        self.assertTrue(summary.overview)
        self.assertIn("SQuAD", answer.text)
        self.assertEqual(answer.citations[0].chunk.page_numbers, (1,))


if __name__ == "__main__":
    unittest.main()
