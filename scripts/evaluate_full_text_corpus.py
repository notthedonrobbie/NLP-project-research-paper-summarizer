"""Evaluate retrieval and extractive summarization on extracted full papers.

Summarization uses the author abstract as reference while conservatively removing
it from the input. The first page is retained only when an Abstract-to-
Introduction boundary is detected; otherwise that page is excluded.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant.evaluation import rouge_scores
from research_assistant.models import Chunk
from research_assistant.retrieval import build_retriever
from research_assistant.summarization import lead_summary, textrank_summary

PROGRESS = ROOT / "results" / "full_text" / "progress.json"
TEXT_DIR = ROOT / "data" / "full_text" / "text"
CSV_PATH = ROOT / "data" / "ai_research_papers.csv"
OUTPUT_DIR = ROOT / "results" / "full_text"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def body_without_abstract_or_references(pages: list[dict]) -> tuple[str, str]:
    first = pages[0]["text"] if pages else ""
    abstract = re.search(r"(?im)^\s*abstract\b[:\s]*", first)
    first_body = ""
    boundary = "first_page_dropped"
    if abstract:
        tail = first[abstract.end():]
        introduction = re.search(r"(?im)^\s*(?:1[.\s]+)?introduction\b", tail)
        if introduction:
            first_body = tail[introduction.start():]
            boundary = "abstract_to_introduction_boundary"
    text = "\n\n".join([first_body, *[page["text"] for page in pages[1:]]])
    references = list(
        re.finditer(r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:references|bibliography)\s*$", text)
    )
    if references:
        text = text[: references[-1].start()]
    return text.strip(), boundary


def load_corpus() -> tuple[list[dict], dict]:
    progress = json.loads(PROGRESS.read_text(encoding="utf-8"))["papers"]
    frame = pd.read_csv(CSV_PATH)
    papers = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        paper_id = f"paper-{index:04d}"
        record = progress.get(paper_id, {})
        text_path = TEXT_DIR / f"{paper_id}.json"
        if record.get("status") != "extracted" or not text_path.is_file():
            continue
        extracted = json.loads(text_path.read_text(encoding="utf-8"))
        body, boundary = body_without_abstract_or_references(extracted["pages"])
        papers.append({
            "paper_id": paper_id,
            "title": clean(row.Title),
            "abstract": clean(row.Summary),
            "link": row.Link,
            "year": int(pd.to_datetime(row.Published, utc=True).year),
            "full_text": "\n\n".join(page["text"] for page in extracted["pages"]),
            "body": body,
            "body_boundary": boundary,
            "pages": extracted["page_count"],
            "words": extracted["word_count"],
            "coverage": extracted["abstract_token_coverage"],
        })
    return papers, progress


def profile_corpus(papers: list[dict], progress: dict) -> dict:
    pages = np.array([paper["pages"] for paper in papers])
    words = np.array([paper["words"] for paper in papers])
    coverage = np.array([paper["coverage"] for paper in papers])
    errors = [
        {"paper_id": paper_id, "arxiv_id": record.get("arxiv_id"),
         "stage": record.get("stage"), "error": record.get("error")}
        for paper_id, record in progress.items() if record.get("status") == "error"
    ]
    return {
        "requested_records": 600,
        "extracted_full_papers": len(papers),
        "excluded_error_records": errors,
        "pages": {"total": int(pages.sum()), "mean": float(pages.mean()),
                  "median": float(np.median(pages)), "maximum": int(pages.max())},
        "extracted_words": {"total": int(words.sum()), "mean": float(words.mean()),
                            "median": float(np.median(words)), "maximum": int(words.max())},
        "abstract_token_coverage": {"mean": float(coverage.mean()),
                                    "median": float(np.median(coverage)),
                                    "minimum": float(coverage.min()),
                                    "below_0_80": int((coverage < 0.8).sum())},
        "summarization_body_boundaries": {
            method: sum(paper["body_boundary"] == method for paper in papers)
            for method in sorted({paper["body_boundary"] for paper in papers})
        },
        "note": "595 records accepted by the user after four 404 downloads and one parser-limit failure. Full text is local and excluded from Git.",
    }


def evaluate_retrieval(papers: list[dict]) -> tuple[dict, dict[str, np.ndarray]]:
    chunks = [Chunk(paper["paper_id"], paper["full_text"], (), "Full paper") for paper in papers]
    per_query = [{"paper_id": paper["paper_id"], "title": paper["title"],
                  "link": paper["link"], "year": paper["year"]} for paper in papers]
    rank_arrays = {}
    for method in ["bm25", "tfidf"]:
        retriever = build_retriever(chunks, method)
        ranks = []
        for index, paper in enumerate(papers):
            ids = [item.chunk.chunk_id for item in retriever.retrieve(paper["title"], 10)]
            rank = ids.index(paper["paper_id"]) + 1 if paper["paper_id"] in ids else None
            ranks.append(rank or 0)
            per_query[index][f"{method}_rank"] = rank
        rank_arrays[method] = np.asarray(ranks, dtype=int)
    return {"per_query": per_query}, rank_arrays


def evaluate_summaries(papers: list[dict]) -> list[dict]:
    def one(paper: dict) -> dict:
        output = {"paper_id": paper["paper_id"], "title": paper["title"], "methods": {}}
        for method, function in [("lead", lead_summary), ("textrank", textrank_summary)]:
            generated = function(paper["body"], max_sentences=6)
            output["methods"][method] = {
                "generated_summary": generated,
                "rouge": rouge_scores(generated, paper["abstract"]),
            }
        return output

    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(one, papers))


def bootstrap(values: np.ndarray, resamples: np.ndarray) -> list[float]:
    return np.quantile(values[resamples].mean(axis=1), [0.025, 0.975]).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-summarization", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "evaluation.json")
    args = parser.parse_args()

    papers, progress = load_corpus()
    if len(papers) != 595:
        raise RuntimeError(f"Expected 595 accepted full papers, found {len(papers)}")
    profile = profile_corpus(papers, progress)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "corpus_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.limit:
        papers = papers[: args.limit]

    retrieval, ranks = evaluate_retrieval(papers)
    rng = np.random.default_rng(2026)
    resamples = rng.integers(0, len(papers), size=(5000, len(papers)))
    result = {
        "papers": len(papers),
        "retrieval_task": "Known-item title-to-full-text retrieval; title fields are excluded from indexed documents.",
        "summarization_task": "Six-sentence extractive summaries of body text with abstract and references conservatively excluded; author abstracts are references.",
        "retrieval": {"methods": {}, **retrieval},
        "bootstrap": {"resamples": 5000, "seed": 2026, "unit": "paper"},
    }
    mrr_values = {}
    for method, rank in ranks.items():
        mrr = np.where(rank > 0, 1.0 / np.maximum(rank, 1), 0.0)
        mrr_values[method] = mrr
        method_result = {
            "mrr": float(mrr.mean()), "mrr_95ci": bootstrap(mrr, resamples),
            "not_found_in_top10": int((rank == 0).sum()), "hit_rate": {},
        }
        for k in [1, 3, 5, 10]:
            hits = ((rank > 0) & (rank <= k)).astype(float)
            method_result["hit_rate"][f"k{k}"] = {
                "mean": float(hits.mean()), "95ci": bootstrap(hits, resamples)
            }
        result["retrieval"]["methods"][method] = method_result
    difference = mrr_values["tfidf"] - mrr_values["bm25"]
    result["retrieval"]["paired_tfidf_minus_bm25_mrr"] = {
        "mean_difference": float(difference.mean()), "95ci": bootstrap(difference, resamples)
    }

    if not args.skip_summarization:
        summaries = evaluate_summaries(papers)
        summary_metrics = {}
        for method in ["lead", "textrank"]:
            summary_metrics[method] = {"macro_average": {}, "median": {}}
            for metric in ["rouge1", "rouge2", "rougeL"]:
                values = np.array([paper["methods"][method]["rouge"][metric] for paper in summaries])
                summary_metrics[method]["macro_average"][metric] = float(values.mean())
                summary_metrics[method]["median"][metric] = float(np.median(values))
        result["summarization"] = {"methods": summary_metrics, "per_paper": summaries}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "profile": profile,
        "retrieval": result["retrieval"]["methods"],
        "paired_difference": result["retrieval"]["paired_tfidf_minus_bm25_mrr"],
        "summarization": result.get("summarization", {}).get("methods"),
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
