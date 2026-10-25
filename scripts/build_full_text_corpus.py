"""Download and extract every version-matched arXiv PDF in the supplied CSV.

The process is sequential, rate-limited, and resumable. PDFs and extracted text
stay local; only progress metadata and evaluation summaries are intended for Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research_assistant.pdf_processor import extract_pdf
from research_assistant.retrieval import tokenize

CSV_PATH = ROOT / "data" / "ai_research_papers.csv"
PDF_DIR = ROOT / "data" / "full_text" / "pdfs"
TEXT_DIR = ROOT / "data" / "full_text" / "text"
PROGRESS_PATH = ROOT / "results" / "full_text" / "progress.json"
MAX_PDF_BYTES = 250 * 1024 * 1024


class RateLimitError(RuntimeError):
    pass


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_progress() -> dict:
    if not PROGRESS_PATH.is_file():
        return {}
    payload = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return payload.get("papers", {})


def save_progress(records: dict) -> None:
    atomic_json(
        PROGRESS_PATH,
        {
            "dataset": CSV_PATH.name,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "papers": records,
        },
    )


def arxiv_id(link: str) -> str:
    match = re.search(r"arxiv\.org/abs/([^/?#]+)", link)
    if not match:
        raise ValueError(f"Unsupported arXiv link: {link}")
    return match.group(1)


def valid_pdf(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 5 and path.read_bytes()[:5] == b"%PDF-"


def download_pdf(url: str, destination: Path) -> dict:
    if valid_pdf(destination):
        return {
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "bytes": destination.stat().st_size,
            "reused_download": True,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pdf.part")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "PaperTrailCoursework/1.0 (sequential academic corpus builder)",
            "Accept": "application/pdf",
        },
    )
    for attempt in range(3):
        try:
            digest = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(request, timeout=120) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > MAX_PDF_BYTES:
                    raise RuntimeError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                with temporary.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        size += len(block)
                        if size > MAX_PDF_BYTES:
                            raise RuntimeError(f"PDF exceeds {MAX_PDF_BYTES} bytes")
                        digest.update(block)
                        handle.write(block)
            if not valid_pdf(temporary):
                raise RuntimeError("Downloaded response is not a valid PDF")
            temporary.replace(destination)
            return {"sha256": digest.hexdigest(), "bytes": size, "reused_download": False}
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After", "not supplied")
                raise RateLimitError(
                    f"arXiv returned HTTP 429; Retry-After: {retry_after}. Stop and resume later."
                ) from exc
            if exc.code not in {500, 502, 503, 504} or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        temporary.unlink(missing_ok=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError("Download failed after three attempts")


def extract_document(pdf_path: Path, text_path: Path, row, paper_id: str, identifier: str) -> dict:
    document = extract_pdf(pdf_path, title=re.sub(r"\s+", " ", row.Title).strip())
    abstract_terms = set(tokenize(row.Summary))
    full_text_terms = set(tokenize(document.text))
    coverage = len(abstract_terms & full_text_terms) / max(len(abstract_terms), 1)
    pages = [{"page_number": page.page_number, "text": page.text} for page in document.pages]
    word_count = len(re.findall(r"\b\w+\b", document.text))
    payload = {
        "paper_id": paper_id,
        "arxiv_id": identifier,
        "title": document.title,
        "authors": row.Authors,
        "published": row.Published,
        "link": row.Link,
        "page_count": len(pages),
        "word_count": word_count,
        "abstract_token_coverage": coverage,
        "pages": pages,
    }
    atomic_json(text_path, payload)
    return {
        "pages": len(pages),
        "words": word_count,
        "abstract_token_coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=None, help="Limit new attempts in this run")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds after each network attempt")
    parser.add_argument("--retry-errors", action="store_true", help="Retry previously failed records")
    args = parser.parse_args()
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.delay < 0:
        parser.error("--delay cannot be negative")

    frame = pd.read_csv(CSV_PATH)
    records = load_progress()
    attempts = 0
    stopped_for_rate_limit = False

    for index, row in enumerate(frame.itertuples(index=False), start=1):
        paper_id = f"paper-{index:04d}"
        identifier = arxiv_id(row.Link)
        pdf_path = PDF_DIR / f"{paper_id}.pdf"
        text_path = TEXT_DIR / f"{paper_id}.json"
        previous = records.get(paper_id, {})
        if previous.get("status") == "extracted" and text_path.is_file():
            continue
        if previous.get("status") == "error" and not args.retry_errors:
            continue
        if args.max_attempts is not None and attempts >= args.max_attempts:
            break

        attempts += 1
        print(f"[{index}/600] {paper_id} {identifier}", flush=True)
        network_attempted = not valid_pdf(pdf_path)
        try:
            pdf_url = f"https://arxiv.org/pdf/{identifier}"
            download = download_pdf(pdf_url, pdf_path)
            extraction = extract_document(pdf_path, text_path, row, paper_id, identifier)
            records[paper_id] = {
                "status": "extracted",
                "arxiv_id": identifier,
                "link": row.Link,
                "pdf_url": pdf_url,
                "pdf_sha256": download["sha256"],
                "pdf_bytes": download["bytes"],
                **extraction,
                "processed_utc": datetime.now(timezone.utc).isoformat(),
            }
            print(
                f"  OK: {extraction['pages']} pages, {extraction['words']} words, "
                f"abstract coverage {extraction['abstract_token_coverage']:.3f}",
                flush=True,
            )
        except RateLimitError as exc:
            records[paper_id] = {
                **previous,
                "status": "error",
                "stage": "download",
                "error": str(exc),
                "arxiv_id": identifier,
                "link": row.Link,
            }
            save_progress(records)
            print(f"  RATE LIMITED: {exc}", flush=True)
            stopped_for_rate_limit = True
            break
        except Exception as exc:
            records[paper_id] = {
                **previous,
                "status": "error",
                "stage": "extract" if valid_pdf(pdf_path) else "download",
                "error": f"{type(exc).__name__}: {exc}",
                "arxiv_id": identifier,
                "link": row.Link,
            }
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
        save_progress(records)
        if network_attempted and args.delay:
            time.sleep(args.delay)

    extracted = sum(
        record.get("status") == "extracted" and (TEXT_DIR / f"{paper_id}.json").is_file()
        for paper_id, record in records.items()
    )
    errors = sum(record.get("status") == "error" for record in records.values())
    summary = {
        "attempts_this_run": attempts,
        "extracted_total": extracted,
        "error_records": errors,
        "remaining": len(frame) - extracted,
        "stopped_for_rate_limit": stopped_for_rate_limit,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if stopped_for_rate_limit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
