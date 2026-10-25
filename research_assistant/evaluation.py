"""Dependency-light ROUGE and retrieval evaluation metrics."""

from __future__ import annotations

from collections import Counter

from .retrieval import tokenize


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def rouge_n(candidate: str, reference: str, n: int = 1) -> float:
    """Return ROUGE-N F1 in the range [0, 1]."""
    candidate_counts = _ngrams(tokenize(candidate), n)
    reference_counts = _ngrams(tokenize(reference), n)
    if not candidate_counts or not reference_counts:
        return 0.0
    overlap = sum((candidate_counts & reference_counts).values())
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """Return longest-common-subsequence ROUGE-L F1."""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    overlap = _lcs_length(candidate_tokens, reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def rouge_scores(candidate: str, reference: str) -> dict[str, float]:
    return {
        "rouge1": rouge_n(candidate, reference, 1),
        "rouge2": rouge_n(candidate, reference, 2),
        "rougeL": rouge_l(candidate, reference),
    }


def precision_recall_at_k(
    retrieved_ids: list[str], relevant_ids: set[str], k: int
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be positive")
    retrieved_at_k = retrieved_ids[:k]
    hits = sum(chunk_id in relevant_ids for chunk_id in retrieved_at_k)
    return {
        "precision_at_k": hits / k,
        "recall_at_k": hits / len(relevant_ids) if relevant_ids else 0.0,
    }
