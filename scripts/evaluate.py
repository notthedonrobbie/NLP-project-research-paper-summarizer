"""Run reproducible summarization and retrieval experiments on one paper."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant.chunking import chunk_document
from research_assistant.evaluation import precision_recall_at_k, rouge_scores
from research_assistant.pdf_processor import extract_pdf
from research_assistant.rag import ExtractiveAnswerGenerator, answer_question
from research_assistant.retrieval import build_retriever
from research_assistant.summarization import summarize_document


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("annotations", type=Path, help="Independent human-created JSON annotations")
    parser.add_argument("--methods", nargs="+", default=["bm25", "tfidf"], choices=["bm25", "tfidf", "dense", "hybrid"])
    parser.add_argument("--summarizers", nargs="+", default=["textrank"], choices=["lead", "textrank", "abstractive"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    document = extract_pdf(args.pdf)
    chunks = chunk_document(document, args.chunk_size, args.overlap)
    results = {
        "document": document.title,
        "configuration": {
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "k": args.k,
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        },
        "summarization": {},
        "retrieval": {},
        "qa_predictions": [],
    }

    reference_summary = annotations.get("reference_summary", "")
    for method in args.summarizers:
        summary = summarize_document(document, method)
        results["summarization"][method] = {
            "overview": summary.overview,
            "sections": summary.sections,
            "rouge": rouge_scores(summary.overview, reference_summary) if reference_summary else None,
        }

    for method in args.methods:
        retriever = build_retriever(chunks, method)
        per_question = []
        for item in annotations.get("questions", []):
            answerable = item.get("answerable", True)
            retrieved = retriever.retrieve(item["question"], args.k)
            ids = [result.chunk.chunk_id for result in retrieved]
            if answerable:
                metrics = precision_recall_at_k(ids, set(item.get("relevant_chunk_ids", [])), args.k)
                per_question.append({"question": item["question"], "retrieved_chunk_ids": ids, **metrics})
            if method == args.methods[0]:
                answer = answer_question(item["question"], retriever, args.k, ExtractiveAnswerGenerator())
                results["qa_predictions"].append(
                    {
                        "question": item["question"],
                        "answerable": answerable,
                        "reference_answer": item.get("reference_answer", ""),
                        "generated_answer": answer.text,
                        "cited_pages": sorted({page for citation in answer.citations for page in citation.chunk.page_numbers}),
                        "insufficient_evidence": answer.insufficient_evidence,
                        "human_scores": {
                            "correctness_0_to_2": None,
                            "relevance_0_to_2": None,
                            "evidence_support_0_to_2": None,
                            "citation_accuracy_0_to_2": None,
                        },
                    }
                )
        results["retrieval"][method] = {
            "mean_precision_at_k": average([row["precision_at_k"] for row in per_question]),
            "mean_recall_at_k": average([row["recall_at_k"] for row in per_question]),
            "questions": per_question,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation written to {args.output}")


if __name__ == "__main__":
    main()
