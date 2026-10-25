"""TextRank summaries and optional transformer-based abstractive summaries."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import is_section_heading
from .models import Document, Summary

SUMMARY_FIELDS: dict[str, tuple[str, ...]] = {
    "Research problem": ("problem", "challenge", "aim", "purpose", "introduction"),
    "Research objectives": ("objective", "aim", "goal", "we propose", "this paper"),
    "Proposed methodology": ("method", "methodology", "approach", "model", "algorithm"),
    "Dataset and experimental setup": ("dataset", "corpus", "experiment", "setup", "baseline"),
    "Main findings": ("result", "finding", "accuracy", "performance", "outperform"),
    "Limitations and future work": ("limitation", "future", "however", "constraint"),
}

_SECTION_KEYWORDS = {
    "Research problem": ("introduction", "abstract", "background"),
    "Research objectives": ("introduction", "abstract"),
    "Proposed methodology": ("method", "methodology", "approach"),
    "Dataset and experimental setup": ("experiment", "setup", "dataset", "material"),
    "Main findings": ("result", "discussion", "conclusion"),
    "Limitations and future work": ("limitation", "future", "discussion", "conclusion"),
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def lead_summary(text: str, max_sentences: int = 8) -> str:
    """Select the opening sentences as a simple extractive comparison baseline."""
    return " ".join(split_sentences(text)[:max_sentences])


def textrank_summary(text: str, max_sentences: int = 8) -> str:
    """Rank sentences by TF-IDF similarity and iterative TextRank centrality."""
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:max_sentences])

    similarity = cosine_similarity(matrix)
    np.fill_diagonal(similarity, 0)
    row_sums = similarity.sum(axis=1, keepdims=True)
    weights = np.divide(similarity, row_sums, out=np.zeros_like(similarity), where=row_sums != 0)
    scores = np.ones(len(sentences), dtype=float) / len(sentences)
    for _ in range(40):
        updated = 0.15 / len(sentences) + 0.85 * weights.T @ scores
        if np.linalg.norm(updated - scores, ord=1) < 1e-6:
            scores = updated
            break
        scores = updated
    selected = sorted(np.argsort(-scores)[:max_sentences])
    return " ".join(sentences[index] for index in selected)


class TransformerSummarizer:
    """Lazy wrapper around a Hugging Face summarization pipeline."""

    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Abstractive summarization requires the optional ML dependencies: "
                "pip install -r requirements-ml.txt"
            ) from exc
        self.model_name = model_name
        self.pipeline = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_sentences: int = 8) -> str:
        # BART-family models have limited context; summarize bounded excerpts hierarchically.
        words = text.split()
        excerpts = [" ".join(words[index : index + 750]) for index in range(0, len(words), 750)]
        excerpts = excerpts[:8]
        partial = []
        for excerpt in excerpts:
            if len(excerpt.split()) < 40:
                partial.append(excerpt)
                continue
            result = self.pipeline(
                excerpt, max_length=min(180, max(60, max_sentences * 25)), min_length=30, do_sample=False
            )
            partial.append(result[0]["summary_text"])
        combined = " ".join(partial)
        if len(partial) > 1 and len(combined.split()) >= 40:
            result = self.pipeline(combined, max_length=220, min_length=50, do_sample=False)
            combined = result[0]["summary_text"]
        return combined


def _document_sections(document: Document) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = "untitled"
    for page in document.pages:
        for line in page.text.splitlines():
            stripped = line.strip()
            if is_section_heading(stripped):
                current = stripped.rstrip(":").lower()
                sections.setdefault(current, [])
            else:
                sections.setdefault(current, []).append(stripped)
    return {name: " ".join(lines) for name, lines in sections.items()}


def _section_for_field(sections: dict[str, str], field: str) -> str:
    keywords = _SECTION_KEYWORDS[field]
    matches = [text for name, text in sections.items() if any(keyword in name for keyword in keywords)]
    return "\n".join(matches)


def _focused_summary(text: str, field: str, max_sentences: int, summarizer) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return "Not explicitly identified in the extracted text."
    keywords = SUMMARY_FIELDS[field]
    scored = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        focus = sum(keyword in lowered for keyword in keywords)
        scored.append((focus, -index, sentence))
    focused = [item for item in scored if item[0] > 0][: max_sentences * 3]
    source = " ".join(item[2] for item in focused) if focused else text
    return summarizer(source, max_sentences)


def summarize_document(
    document: Document,
    method: str = "textrank",
    max_sentences: int = 6,
    model_name: str = "facebook/bart-large-cnn",
) -> Summary:
    """Create an overview plus the six research fields required by the project."""
    if method == "lead":
        summarizer = lead_summary
    elif method == "textrank":
        summarizer = textrank_summary
    elif method == "abstractive":
        transformer = TransformerSummarizer(model_name)
        summarizer = transformer.summarize
    else:
        raise ValueError(f"Unknown summarization method: {method}")

    sections = _document_sections(document)
    abstract = next((text for name, text in sections.items() if "abstract" in name), "")
    overview = summarizer(abstract or document.text, max_sentences)
    structured = {}
    for field in SUMMARY_FIELDS:
        source = _section_for_field(sections, field) or document.text
        structured[field] = _focused_summary(source, field, max(2, max_sentences // 2), summarizer)
    return Summary(overview, structured, method)
