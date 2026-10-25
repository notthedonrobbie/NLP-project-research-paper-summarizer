"""Source-grounded scientific document summarization and question answering."""

from .models import Answer, Chunk, Document, PageText, RetrievedChunk, Summary
from .pipeline import ResearchAssistant

__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "PageText",
    "ResearchAssistant",
    "RetrievedChunk",
    "Summary",
]

__version__ = "0.1.0"
