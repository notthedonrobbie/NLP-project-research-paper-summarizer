"""Aggregate per-paper pilot results, bootstrap intervals, and human Q&A ratings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant.evaluation import precision_recall_at_k
PAPERS = ["attention-is-all-you-need", "retrieval-augmented-generation", "textrank"]
METRICS = ["rouge1", "rouge2", "rougeL"]
DIMENSIONS = [
    "correctness_0_to_2",
    "relevance_0_to_2",
    "evidence_support_0_to_2",
    "citation_accuracy_0_to_2",
]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 2026) -> list[float]:
    """Question-level percentile bootstrap; intervals describe this tiny pilot only."""
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(5000, len(array)), replace=True).mean(axis=1)
    return [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))]


def aggregate(directory: Path, include_human: bool = False) -> dict:
    results = {paper: read_json(directory / f"{paper}.json") for paper in PAPERS}
    annotations = {paper: read_json(ROOT / "data" / "annotations" / f"{paper}.json") for paper in PAPERS}
    output = {"papers": {}, "summarization": {}, "retrieval": {}}

    for paper in PAPERS:
        output["papers"][paper] = {
            "pages": results[paper]["configuration"] | {},
            "answerable_questions": sum(q.get("answerable", True) for q in annotations[paper]["questions"]),
            "unanswerable_questions": sum(not q.get("answerable", True) for q in annotations[paper]["questions"]),
        }

    for method in ["lead", "textrank"]:
        rows = [results[paper]["summarization"][method] for paper in PAPERS]
        output["summarization"][method] = {
            "macro_average": {metric: mean([row["rouge"][metric] for row in rows]) for metric in METRICS},
            "per_paper": {paper: results[paper]["summarization"][method]["rouge"] for paper in PAPERS},
        }

    for method in ["bm25", "tfidf"]:
        per_k = {}
        question_rows = []
        for k in [1, 3, 5]:
            precision_values = []
            recall_values = []
            for paper in PAPERS:
                annotation_by_question = {q["question"]: q for q in annotations[paper]["questions"] if q.get("answerable", True)}
                for row in results[paper]["retrieval"][method]["questions"]:
                    relevant = set(annotation_by_question[row["question"]]["relevant_chunk_ids"])
                    score = precision_recall_at_k(row["retrieved_chunk_ids"], relevant, k)
                    precision_values.append(score["precision_at_k"])
                    recall_values.append(score["recall_at_k"])
                    if k == 5:
                        question_rows.append({"paper": paper, "question": row["question"], **score})
            per_k[f"k{k}"] = {
                "mean_precision": mean(precision_values),
                "mean_recall": mean(recall_values),
            }
        per_k["k5"]["precision_95ci"] = bootstrap_ci([row["precision_at_k"] for row in question_rows])
        per_k["k5"]["recall_95ci"] = bootstrap_ci([row["recall_at_k"] for row in question_rows])
        output["retrieval"][method] = {
            "question_level": per_k,
            "per_paper_k5": {
                paper: {
                    "precision": results[paper]["retrieval"][method]["mean_precision_at_k"],
                    "recall": results[paper]["retrieval"][method]["mean_recall_at_k"],
                }
                for paper in PAPERS
            },
        }

    predictions = [q for paper in PAPERS for q in results[paper]["qa_predictions"]]
    output["qa"] = {
        "total_questions": len(predictions),
        "correct_abstentions": sum(not q["answerable"] and q["insufficient_evidence"] for q in predictions),
        "unanswerable_questions": sum(not q["answerable"] for q in predictions),
        "unsupported_answers": sum(not q["answerable"] and not q["insufficient_evidence"] for q in predictions),
        "over_refusals": sum(q["answerable"] and q["insufficient_evidence"] for q in predictions),
        "answerable_questions": sum(q["answerable"] for q in predictions),
    }

    if include_human:
        human = read_json(ROOT / "data" / "human_scores.json")["scores"]
        reviewed = []
        for paper in PAPERS:
            for question, prediction, rating in zip(
                annotations[paper]["questions"], results[paper]["qa_predictions"], human[paper]
            ):
                assert question["question"] == prediction["question"]
                reviewed.append({"paper": paper, "question": question["question"], **rating})
        output["qa"]["human_review"] = {
            "annotators": 1,
            "mean_scores": {dimension: mean([row[dimension] for row in reviewed]) for dimension in DIMENSIONS},
            "ratings": reviewed,
        }
    return output


def main() -> None:
    final = aggregate(ROOT / "results" / "pilot", include_human=True)
    initial = aggregate(ROOT / "results" / "pilot" / "initial")
    final["initial_comparison"] = {
        "retrieval_k5": {
            method: initial["retrieval"][method]["question_level"]["k5"]
            for method in ["bm25", "tfidf"]
        },
        "qa": initial["qa"],
    }
    output = ROOT / "results" / "pilot" / "aggregate.json"
    output.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "summarization": {m: v["macro_average"] for m, v in final["summarization"].items()},
        "retrieval_k5": {m: v["question_level"]["k5"] for m, v in final["retrieval"].items()},
        "qa": {k: v for k, v in final["qa"].items() if k != "human_review"},
        "human_means": final["qa"]["human_review"]["mean_scores"],
    }, indent=2))
    print(f"Aggregate results written to {output}")


if __name__ == "__main__":
    main()
