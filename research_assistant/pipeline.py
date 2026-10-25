"""High-level API used by the Streamlit app, notebooks, and tests."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from .chunking import chunk_document
from .models import Answer, Document, Summary
from .pdf_processor import extract_pdf
from .rag import answer_question
from .retrieval import Retriever, build_retriever
from .summarization import summarize_document


class ResearchAssistant:
    def __init__(
        self,
        document: Document,
        retrieval_method: str = "tfidf",
        chunk_size: int = 1000,
        overlap: int = 150,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.document = document
        self.chunks = chunk_document(document, chunk_size, overlap)
        self.retriever: Retriever = build_retriever(
            self.chunks, retrieval_method, embedding_model=embedding_model
        )
        self.retrieval_method = retrieval_method

    @classmethod
    def from_pdf(
        cls,
        source: str | Path | bytes | BinaryIO,
        title: str | None = None,
        **kwargs,
    ) -> "ResearchAssistant":
        return cls(extract_pdf(source, title=title), **kwargs)

    def summarize(self, method: str = "textrank", max_sentences: int = 6, **kwargs) -> Summary:
        return summarize_document(self.document, method, max_sentences, **kwargs)

    def ask(self, question: str, top_k: int = 5, generator=None) -> Answer:
        return answer_question(question, self.retriever, top_k, generator)
