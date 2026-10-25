"""Run a concise, fully offline PaperTrail demonstration from the terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant import ResearchAssistant


def print_answer(label: str, question: str, answer) -> None:
    print(f"\n{label}\nQ: {question}\nA: {answer.text}")
    if answer.citations:
        pages = sorted(
            {page for citation in answer.citations for page in citation.chunk.page_numbers}
        )
        print(f"Cited PDF pages: {pages}")
    print(f"Insufficient evidence: {answer.insufficient_evidence}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        default=ROOT / "data" / "papers" / "textrank.pdf",
        help="Digitally generated English research PDF",
    )
    parser.add_argument(
        "--question",
        default="Which two extraction applications are investigated?",
    )
    parser.add_argument(
        "--unsupported-question",
        default="What F1 score did TextRank achieve on SQuAD?",
    )
    parser.add_argument(
        "--retrieval", choices=["tfidf", "bm25", "dense", "hybrid"], default="tfidf"
    )
    parser.add_argument(
        "--summarizer", choices=["lead", "textrank", "abstractive"], default="textrank"
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(
            f"PDF not found: {args.pdf}. Run 'python scripts/download_corpus.py' "
            "or supply another PDF path."
        )

    assistant = ResearchAssistant.from_pdf(args.pdf, retrieval_method=args.retrieval)
    summary = assistant.summarize(method=args.summarizer)
    print(f"Paper: {assistant.document.title}")
    print(f"Pages: {len(assistant.document.pages)} | Chunks: {len(assistant.chunks)}")
    print(f"Retriever: {args.retrieval} | Summarizer: {args.summarizer}")
    print(f"\nOVERVIEW\n{summary.overview}")
    print("\nSTRUCTURED RESEARCH FIELDS")
    for name, value in summary.sections.items():
        print(f"\n{name.replace('_', ' ').title()}\n{value}")

    print_answer(
        "ANSWERABLE QUESTION", args.question, assistant.ask(args.question, top_k=args.top_k)
    )
    print_answer(
        "UNSUPPORTED-QUESTION CHECK",
        args.unsupported_question,
        assistant.ask(args.unsupported_question, top_k=args.top_k),
    )
    print("\nVerify important claims against the original PDF; this is a research prototype.")


if __name__ == "__main__":
    main()
