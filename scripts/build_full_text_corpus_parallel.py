"""Bounded parallel driver for the resumable full-text corpus builder."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

from build_full_text_corpus import (
    CSV_PATH,
    PDF_DIR,
    PROGRESS_PATH,
    TEXT_DIR,
    RateLimitError,
    arxiv_id,
    download_pdf,
    extract_document,
    load_progress,
    save_progress,
    valid_pdf,
)


class StartRateLimiter:
    """Space network starts globally even when several workers are active."""

    def __init__(self, interval: float):
        self.interval = interval
        self.lock = threading.Lock()
        self.last_start = 0.0

    def wait(self) -> None:
        with self.lock:
            remaining = self.interval - (time.monotonic() - self.last_start)
            if remaining > 0:
                time.sleep(remaining)
            self.last_start = time.monotonic()


def process(row, index: int, limiter: StartRateLimiter) -> tuple[str, dict, bool]:
    paper_id = f"paper-{index:04d}"
    identifier = arxiv_id(row.Link)
    pdf_path = PDF_DIR / f"{paper_id}.pdf"
    text_path = TEXT_DIR / f"{paper_id}.json"
    network_attempted = not valid_pdf(pdf_path)
    if network_attempted:
        limiter.wait()
    pdf_url = f"https://arxiv.org/pdf/{identifier}"
    download = download_pdf(pdf_url, pdf_path)
    extraction = extract_document(pdf_path, text_path, row, paper_id, identifier)
    record = {
        "status": "extracted",
        "arxiv_id": identifier,
        "link": row.Link,
        "pdf_url": pdf_url,
        "pdf_sha256": download["sha256"],
        "pdf_bytes": download["bytes"],
        **extraction,
        "processed_utc": datetime.now(timezone.utc).isoformat(),
    }
    return paper_id, record, network_attempted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start-interval", type=float, default=1.0)
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.start_interval < 0.5:
        parser.error("--start-interval must be at least 0.5 seconds")
    if args.max_attempts is not None and args.max_attempts < 1:
        parser.error("--max-attempts must be positive")

    frame = pd.read_csv(CSV_PATH)
    records = load_progress()
    tasks = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        paper_id = f"paper-{index:04d}"
        previous = records.get(paper_id, {})
        if previous.get("status") == "extracted" and (TEXT_DIR / f"{paper_id}.json").is_file():
            continue
        if previous.get("status") == "error" and not args.retry_errors:
            continue
        tasks.append((index, row))
        if args.max_attempts is not None and len(tasks) >= args.max_attempts:
            break

    limiter = StartRateLimiter(args.start_interval)
    stopped_for_rate_limit = False
    completed_attempts = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process, row, index, limiter): (index, row)
            for index, row in tasks
        }
        for future in as_completed(futures):
            if future.cancelled():
                continue
            index, row = futures[future]
            paper_id = f"paper-{index:04d}"
            previous = records.get(paper_id, {})
            completed_attempts += 1
            try:
                _, record, _ = future.result()
                records[paper_id] = record
                print(
                    f"[{index}/600] OK {record['pages']} pages, {record['words']} words, "
                    f"coverage {record['abstract_token_coverage']:.3f}",
                    flush=True,
                )
            except RateLimitError as exc:
                records[paper_id] = {
                    **previous, "status": "error", "stage": "download", "error": str(exc),
                    "arxiv_id": arxiv_id(row.Link), "link": row.Link,
                }
                print(f"[{index}/600] RATE LIMITED: {exc}", flush=True)
                stopped_for_rate_limit = True
                for pending in futures:
                    pending.cancel()
            except Exception as exc:
                records[paper_id] = {
                    **previous, "status": "error",
                    "stage": "extract" if valid_pdf(PDF_DIR / f"{paper_id}.pdf") else "download",
                    "error": f"{type(exc).__name__}: {exc}",
                    "arxiv_id": arxiv_id(row.Link), "link": row.Link,
                }
                print(f"[{index}/600] ERROR: {type(exc).__name__}: {exc}", flush=True)
            save_progress(records)

    extracted = sum(
        record.get("status") == "extracted" and (TEXT_DIR / f"{paper_id}.json").is_file()
        for paper_id, record in records.items()
    )
    errors = sum(record.get("status") == "error" for record in records.values())
    print(json.dumps({
        "attempts_this_run": completed_attempts,
        "extracted_total": extracted,
        "error_records": errors,
        "remaining": len(frame) - extracted,
        "stopped_for_rate_limit": stopped_for_rate_limit,
    }, indent=2), flush=True)
    if stopped_for_rate_limit:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
