"""Reproduce all measured pilot outputs with one command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from download_corpus import download_corpus

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest_path = ROOT / "data" / "corpus_manifest.json"
    download_corpus(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for paper in manifest["papers"]:
        output = ROOT / "results" / "pilot" / f"{paper['id']}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            str(ROOT / paper["local_pdf"]),
            str(ROOT / paper["annotations"]),
            "--methods",
            "bm25",
            "tfidf",
            "--summarizers",
            "lead",
            "textrank",
            "--k",
            "5",
            "--chunk-size",
            "1000",
            "--overlap",
            "150",
            "--output",
            str(output),
        ]
        print(f"\nEvaluating {paper['title']}...")
        subprocess.run(command, cwd=ROOT, check=True)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "aggregate_pilot.py")],
        cwd=ROOT,
        check=True,
    )
    print("\nPilot complete: results/pilot/aggregate.json")
    print("Optional figures: npm install && node scripts/render_charts.mjs")


if __name__ == "__main__":
    main()
