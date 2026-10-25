"""Download and hash-check the public papers listed in the pilot manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_corpus(manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for paper in manifest["papers"]:
        destination = ROOT / paper["local_pdf"]
        expected_hash = paper["sha256"]
        if destination.is_file():
            actual_hash = sha256(destination)
            if actual_hash == expected_hash:
                print(f"Verified existing: {destination.relative_to(ROOT)}")
                continue
            raise RuntimeError(
                f"Hash mismatch for existing file {destination}. "
                "Move it aside before downloading; it was not overwritten."
            )

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".pdf.part")
        request = urllib.request.Request(
            paper["pdf_url"], headers={"User-Agent": "PaperTrail-pilot/1.0"}
        )
        try:
            print(f"Downloading: {paper['landing_page']}")
            with urllib.request.urlopen(request, timeout=90) as response:
                temporary.write_bytes(response.read())
            actual_hash = sha256(temporary)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Downloaded hash mismatch for {paper['id']}: "
                    f"expected {expected_hash}, got {actual_hash}"
                )
            temporary.replace(destination)
            print(f"Saved and verified: {destination.relative_to(ROOT)}")
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "corpus_manifest.json",
        help="Corpus manifest containing source URLs and expected SHA-256 hashes",
    )
    args = parser.parse_args()
    download_corpus(args.manifest.resolve())


if __name__ == "__main__":
    main()
