"""PDF extraction and conservative cleanup for scientific papers."""

from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader

from .models import Document, PageText


def _normalized_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def _remove_repeated_lines(pages: list[str]) -> list[str]:
    """Remove headers/footers repeated on at least half of the pages."""
    if len(pages) < 3:
        return pages

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [_normalized_line(line) for line in page.splitlines()]
        counts.update(set(line for line in lines if line and len(line) <= 120))
    repeated = {line for line, count in counts.items() if count >= (len(pages) + 1) // 2}

    cleaned = []
    for page in pages:
        kept = [line for line in page.splitlines() if _normalized_line(line) not in repeated]
        cleaned.append("\n".join(kept))
    return cleaned


def clean_text(text: str) -> str:
    """Fix common PDF artifacts without deleting technical content."""
    text = text.replace("\x00", "")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(source: str | Path | bytes | BinaryIO, title: str | None = None) -> Document:
    """Extract text and page numbers from a digital (non-scanned) PDF."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        reader = PdfReader(str(path))
        default_title = path.stem
    else:
        data = source if isinstance(source, bytes) else source.read()
        reader = PdfReader(io.BytesIO(data))
        default_title = "Uploaded paper"

    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    pages = _remove_repeated_lines(pages)
    if not any(page.strip() for page in pages):
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; run OCR before uploading it."
        )

    metadata = dict(reader.metadata or {})
    pdf_title = metadata.get("/Title")
    document_title = title or (str(pdf_title).strip() if pdf_title else "") or default_title
    numbered_pages = [PageText(index + 1, text) for index, text in enumerate(pages)]
    return Document(document_title, numbered_pages, metadata)
