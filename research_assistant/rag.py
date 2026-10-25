"""Document-grounded answer generation with explicit source identifiers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .models import Answer, RetrievedChunk
from .retrieval import Retriever, tokenize
from .summarization import split_sentences

INSUFFICIENT_ANSWER = "The uploaded paper does not provide enough evidence to answer this question."
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}


@dataclass
class GenerationResult:
    text: str
    source_ids: list[str]
    insufficient_evidence: bool
    mode: str


class ExtractiveAnswerGenerator:
    """Offline fallback: quote relevant source sentences instead of inventing prose."""

    def generate(self, question: str, evidence: list[RetrievedChunk]) -> GenerationResult:
        query_terms = {token for token in tokenize(question) if token not in _STOPWORDS}
        evidence_terms = {token for item in evidence for token in tokenize(item.chunk.text)}
        raw_terms = re.findall(r"[A-Za-z][A-Za-z0-9]*", question)
        named_terms = {
            tokenize(token)[0]
            for token in raw_terms[1:]
            if any(character.isupper() for character in token)
        }
        if named_terms - evidence_terms:
            return GenerationResult(INSUFFICIENT_ANSWER, [], True, "Extractive evidence (offline)")

        candidates = []
        for source_id, retrieved in enumerate(evidence, start=1):
            for sentence in split_sentences(retrieved.chunk.text):
                if "@" in sentence or "proper attribution" in sentence.lower():
                    continue
                terms = set(tokenize(sentence))
                overlap = len(query_terms & terms) / max(len(query_terms), 1)
                if overlap >= 0.25:
                    score = overlap + 0.03 * (len(evidence) - source_id)
                    candidates.append((score, source_id, sentence))
        if not candidates:
            return GenerationResult(INSUFFICIENT_ANSWER, [], True, "Extractive evidence (offline)")
        selected = sorted(candidates, key=lambda item: item[0], reverse=True)[:3]
        answer = " ".join(f"{sentence} [S{source_id}]" for _, source_id, sentence in selected)
        source_ids = sorted({f"S{source_id}" for _, source_id, _ in selected}, key=lambda value: int(value[1:]))
        return GenerationResult(answer, source_ids, False, "Extractive evidence (offline)")


class OpenAIAnswerGenerator:
    """Optional LLM generator using retrieved evidence as its only source."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("OpenAI generation requires: pip install openai") from exc
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, question: str, evidence: list[RetrievedChunk]) -> GenerationResult:
        context = "\n\n".join(
            f"[S{index}] Pages {', '.join(map(str, item.chunk.page_numbers))}\n{item.chunk.text}"
            for index, item in enumerate(evidence, start=1)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions using only the supplied scientific-paper excerpts. "
                        "Never use outside knowledge. Cite every factual claim with source IDs such as [S1]. "
                        f"If the excerpts are insufficient, respond exactly: {INSUFFICIENT_ANSWER}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nEvidence:\n{context}",
                },
            ],
        )
        text = response.choices[0].message.content or INSUFFICIENT_ANSWER
        source_ids = sorted(set(re.findall(r"S\d+", text)), key=lambda value: int(value[1:]))
        insufficient = INSUFFICIENT_ANSWER.lower() in text.lower()
        return GenerationResult(text, source_ids, insufficient, f"OpenAI RAG ({self.model})")


def answer_question(
    question: str,
    retriever: Retriever,
    top_k: int = 5,
    generator=None,
) -> Answer:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    evidence = retriever.retrieve(question, top_k)
    if not evidence:
        return Answer(INSUFFICIENT_ANSWER, [], [], True, "No retrieval")
    generator = generator or ExtractiveAnswerGenerator()
    generated = generator.generate(question, evidence)
    citations = []
    for source_id in generated.source_ids:
        index = int(source_id[1:]) - 1
        if 0 <= index < len(evidence):
            citations.append(evidence[index])
    return Answer(
        generated.text,
        citations,
        evidence,
        generated.insufficient_evidence,
        generated.mode,
    )
