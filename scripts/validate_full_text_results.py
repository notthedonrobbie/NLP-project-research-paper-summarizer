"""Independently verify full-text artifacts against their per-paper records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "full_text" / "evaluation.json"
PROFILE = ROOT / "results" / "full_text" / "corpus_profile.json"
REPORT = ROOT / "report" / "technical_report.md"


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12), (actual, expected)


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert result["papers"] == profile["extracted_full_papers"] == 595
    assert profile["requested_records"] == 600
    assert len(profile["excluded_error_records"]) == 5
    assert profile["pages"]["total"] == 12836
    assert profile["extracted_words"]["total"] == 6607965
    assert profile["abstract_token_coverage"]["below_0_80"] == 9

    queries = result["retrieval"]["per_query"]
    assert len(queries) == 595
    for method in ["bm25", "tfidf"]:
        ranks = np.array([row[f"{method}_rank"] or 0 for row in queries])
        saved = result["retrieval"]["methods"][method]
        close(float(np.where(ranks > 0, 1 / np.maximum(ranks, 1), 0).mean()), saved["mrr"])
        assert int((ranks == 0).sum()) == saved["not_found_in_top10"]
        for k in [1, 3, 5, 10]:
            close(float(((ranks > 0) & (ranks <= k)).mean()), saved["hit_rate"][f"k{k}"]["mean"])

    papers = result["summarization"]["per_paper"]
    assert len(papers) == 595
    for method in ["lead", "textrank"]:
        for metric in ["rouge1", "rouge2", "rougeL"]:
            values = np.array([row["methods"][method]["rouge"][metric] for row in papers])
            close(float(values.mean()), result["summarization"]["methods"][method]["macro_average"][metric])
            close(float(np.median(values)), result["summarization"]["methods"][method]["median"][metric])

    report = REPORT.read_text(encoding="utf-8")
    word_count = len(re.findall(r"\b[\w’'-]+\b", report))
    assert 3000 <= word_count <= 4000, word_count
    assert "BM25 achieved 0.905" not in report and "TF-IDF achieved 0.903" not in report
    figures = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report)
    assert len(figures) == 3
    for figure in figures:
        path = REPORT.parent / figure
        assert path.is_file() and path.stat().st_size > 1000, path

    print(json.dumps({
        "papers": result["papers"], "pages": profile["pages"]["total"],
        "words": profile["extracted_words"]["total"], "report_words": word_count,
        "figures_verified": figures, "metric_recalculation": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
