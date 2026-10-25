"""Shared data structures for the research assistant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PageText:
    """Text extracted from one PDF page; page_number is one-based."""

    page_number: int
    text: str


@dataclass
class Document:
    """A processed research paper."""

    title: str
    pages: list[PageText]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class Chunk:
    """A retrieval unit with traceable source pages."""

    chunk_id: str
    text: str
    page_numbers: tuple[int, ...]
    section: str = "Unknown section"


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk and its retrieval score."""

    chunk: Chunk
    score: float


@dataclass
class Summary:
    """An overview and structured research information."""

    overview: str
    sections: dict[str, str]
    method: str


@dataclass
class Answer:
    """A generated answer and the passages that support it."""

    text: str
    citations: list[RetrievedChunk]
    evidence: list[RetrievedChunk]
    insufficient_evidence: bool
    generation_mode: str
