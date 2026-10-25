"""Page-aware chunking with section labels and bounded overlap."""

from __future__ import annotations

import re

from .models import Chunk, Document

_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|background|related work|methodology|methods?|"
    r"experimental setup|experiments?|results?|discussion|limitations?|"
    r"conclusion|future work|references)\s*:?$",
    re.IGNORECASE,
)


def is_section_heading(line: str) -> bool:
    return bool(_HEADING.match(line.strip()))


def _words_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and current_size + extra > chunk_size:
            chunks.append(" ".join(current))
            retained: list[str] = []
            retained_size = 0
            for previous in reversed(current):
                size = len(previous) + (1 if retained else 0)
                if retained_size + size > overlap:
                    break
                retained.insert(0, previous)
                retained_size += size
            current = retained
            current_size = retained_size
        current.append(word)
        current_size += len(word) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(document: Document, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Chunk each page independently so every chunk has an exact page reference."""
    if chunk_size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if overlap < 0 or overlap >= chunk_size // 2:
        raise ValueError("overlap must be non-negative and less than half of chunk_size")

    chunks: list[Chunk] = []
    section = "Unknown section"
    for page in document.pages:
        units: list[tuple[str, str]] = []
        for block in re.split(r"\n\s*\n|\n(?=\d+(?:\.\d+)*\.?\s+[A-Z])", page.text):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            if len(lines) == 1 and is_section_heading(lines[0]):
                section = lines[0].strip().rstrip(":")
                continue
            if lines and is_section_heading(lines[0]):
                section = lines.pop(0).strip().rstrip(":")
                block = "\n".join(lines).strip()
                if not block:
                    continue
            units.append((section, block))

        buffer = ""
        buffer_section = section
        for unit_section, block in units:
            if buffer and (len(buffer) + len(block) + 2 > chunk_size or unit_section != buffer_section):
                for piece in _words_with_overlap(buffer, chunk_size, overlap):
                    chunks.append(Chunk(f"chunk-{len(chunks) + 1:04d}", piece, (page.page_number,), buffer_section))
                buffer = ""
            if not buffer:
                buffer_section = unit_section
            buffer = f"{buffer}\n\n{block}".strip()
        if buffer:
            for piece in _words_with_overlap(buffer, chunk_size, overlap):
                chunks.append(Chunk(f"chunk-{len(chunks) + 1:04d}", piece, (page.page_number,), buffer_section))

    if not chunks and document.text.strip():
        chunks.append(Chunk("chunk-0001", document.text.strip(), (1,), section))
    return chunks
