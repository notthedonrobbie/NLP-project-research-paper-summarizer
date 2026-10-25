"""Profile the supplied 600-paper CSV and evaluate title-to-abstract retrieval.

The automatic task is known-item retrieval: a paper title is the query and only
its own abstract is relevant. It is not natural-language question answering.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant.models import Chunk
from research_assistant.retrieval import build_retriever

INPUT = ROOT / "data" / "ai_research_papers.csv"
OUTPUT_DIR = ROOT / "results" / "ai_research_papers"
WORD = re.compile(r"\b\w+['’-]?\w*\b")


def clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    frame = pd.read_csv(INPUT)
    required = ["Title", "Authors", "Published", "Summary", "Link"]
    if list(frame.columns) != required:
        raise ValueError(f"Expected columns {required}, found {list(frame.columns)}")
    if frame[required].isna().any().any():
        raise ValueError("The supplied evaluation expects complete metadata in every row")
    if frame["Link"].nunique() != len(frame):
        raise ValueError("Paper links must be unique for known-item evaluation")

    published = pd.to_datetime(frame["Published"], utc=True, errors="raise")
    abstract_words = frame["Summary"].map(lambda text: len(WORD.findall(str(text))))
    title_words = frame["Title"].map(lambda text: len(WORD.findall(str(text))))
    author_counts = frame["Authors"].map(lambda text: len(str(text).split(",")))
    normalized_titles = frame["Title"].map(lambda text: clean(text).casefold())

    profile = {
        "dataset": INPUT.name,
        "sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
        "columns": required,
        "missing_cells": int(frame[required].isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_normalized_titles": int(normalized_titles.duplicated().sum()),
        "unique_links": int(frame["Link"].nunique()),
        "arxiv_abstract_links": int(frame["Link"].str.contains(r"arxiv\.org/abs/", regex=True).sum()),
        "date_range_utc": [published.min().isoformat(), published.max().isoformat()],
        "papers_by_year": {
            str(int(year)): int(count)
            for year, count in published.dt.year.value_counts().sort_index().items()
        },
        "abstract_words": {
            "total": int(abstract_words.sum()),
            "minimum": int(abstract_words.min()),
            "median": float(abstract_words.median()),
            "mean": float(abstract_words.mean()),
            "p90": float(abstract_words.quantile(0.9)),
            "maximum": int(abstract_words.max()),
        },
        "title_words": {
            "median": float(title_words.median()),
            "mean": float(title_words.mean()),
            "maximum": int(title_words.max()),
        },
        "authors_per_paper_comma_separated_estimate": {
            "median": float(author_counts.median()),
            "mean": float(author_counts.mean()),
            "maximum": int(author_counts.max()),
        },
        "scope_note": "Metadata and author abstracts only; no full text, relevance judgments, reference-generated summaries, or QA annotations.",
    }

    chunks = [
        Chunk(f"paper-{index:04d}", clean(row.Summary), (), "Abstract")
        for index, row in enumerate(frame.itertuples(index=False), start=1)
    ]
    methods = ["bm25", "tfidf"]
    per_query: list[dict] = [
        {
            "paper_id": chunk.chunk_id,
            "title": clean(row.Title),
            "link": row.Link,
            "year": int(year.year),
        }
        for chunk, row, year in zip(chunks, frame.itertuples(index=False), published)
    ]
    ranks: dict[str, np.ndarray] = {}

    for method in methods:
        retriever = build_retriever(chunks, method)
        method_ranks = []
        for index, row in enumerate(frame.itertuples(index=False)):
            target = chunks[index].chunk_id
            retrieved = retriever.retrieve(clean(row.Title), 10)
            ids = [item.chunk.chunk_id for item in retrieved]
            rank = ids.index(target) + 1 if target in ids else None
            method_ranks.append(rank if rank is not None else 0)
            per_query[index][f"{method}_rank"] = rank
        ranks[method] = np.asarray(method_ranks, dtype=int)

    rng = np.random.default_rng(2026)
    resamples = rng.integers(0, len(frame), size=(5000, len(frame)))
    evaluation = {
        "task": "Known-item title-to-abstract retrieval; titles are queries and titles are excluded from indexed documents.",
        "papers": int(len(frame)),
        "cutoffs": [1, 3, 5, 10],
        "bootstrap": {"resamples": 5000, "seed": 2026, "unit": "paper"},
        "methods": {},
        "per_query": per_query,
    }
    mrr_arrays = {}
    hit5_arrays = {}
    for method in methods:
        rank = ranks[method]
        mrr = np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0)
        mrr_arrays[method] = mrr
        hit5_arrays[method] = ((rank > 0) & (rank <= 5)).astype(float)
        result = {
            "mrr": mean(mrr.tolist()),
            "mrr_95ci": np.quantile(mrr[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
            "hit_rate": {},
            "median_rank_when_found_in_top10": float(np.median(rank[rank > 0])),
            "not_found_in_top10": int((rank == 0).sum()),
        }
        for k in [1, 3, 5, 10]:
            hits = ((rank > 0) & (rank <= k)).astype(float)
            result["hit_rate"][f"k{k}"] = {
                "mean": mean(hits.tolist()),
                "95ci": np.quantile(hits[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
            }
        evaluation["methods"][method] = result

    difference = mrr_arrays["tfidf"] - mrr_arrays["bm25"]
    evaluation["paired_tfidf_minus_bm25_mrr"] = {
        "mean_difference": mean(difference.tolist()),
        "95ci": np.quantile(difference[resamples].mean(axis=1), [0.025, 0.975]).tolist(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_DIR / "known_item_retrieval.json").write_text(
        json.dumps(evaluation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"profile": profile, "retrieval": evaluation["methods"],
                      "paired_difference": evaluation["paired_tfidf_minus_bm25_mrr"]}, indent=2))
    print(f"Wrote {OUTPUT_DIR / 'dataset_profile.json'}")
    print(f"Wrote {OUTPUT_DIR / 'known_item_retrieval.json'}")


if __name__ == "__main__":
    main()
