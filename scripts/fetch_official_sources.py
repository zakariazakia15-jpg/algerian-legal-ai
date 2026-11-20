"""Download official Algerian legal PDFs listed in an explicit manifest.

The script never discovers or downloads URLs implicitly: every source must be
listed in scripts/official_source_manifest.json. Files are stored under
data/raw/ with a manifest that records the source URL, byte count, SHA-256 and
fetch time so a later extraction step can prove which bytes were processed.

Usage:
    python scripts/fetch_official_sources.py --only constitution-2026-ar
    python scripts/fetch_official_sources.py --max-priority 1 --zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "scripts" / "official_source_manifest.json"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
USER_AGENT = "dz-legal-ai/0.1 (local research corpus builder; contact: project maintainer)"


def _load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("Manifest must be an object with a 'sources' list")
    return data


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "files": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("files"), dict):
        raise ValueError("Raw manifest is invalid")
    return state


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, *, timeout: int, retries: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                content_type = (response.headers.get("Content-Type") or "").lower()
                if "pdf" not in content_type and "octet-stream" not in content_type:
                    raise ValueError(f"unexpected content type: {content_type!r}")
                payload = response.read()
            if not payload.startswith(b"%PDF"):
                raise ValueError("downloaded bytes are not a PDF")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def _zip_raw(raw_dir: Path, archive_path: Path) -> int:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for pdf in pdfs:
            archive.write(pdf, arcname=f"raw/{pdf.name}")
        state_path = raw_dir / "manifest.json"
        if state_path.exists():
            archive.write(state_path, arcname="raw/manifest.json")
    return len(pdfs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--only", action="append", default=[], help="source_id to fetch (repeatable)")
    parser.add_argument("--max-priority", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument("--zip", action="store_true", help="write data/raw_sources.zip")
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    base_url = str(manifest.get("base_url", ""))
    state_path = args.raw_dir / "manifest.json"
    state = _read_state(state_path)
    selected = []
    for source in manifest["sources"]:
        source_id = source["source_id"]
        if args.only and source_id not in args.only:
            continue
        priority = int(source.get("priority", 9))
        if args.max_priority is not None and priority > args.max_priority:
            continue
        selected.append(source)

    failures: list[str] = []
    fetched = 0
    reused = 0
    for source in selected:
        source_id = source["source_id"]
        url = urljoin(base_url, source["file"])
        destination = args.raw_dir / f"{source_id}.pdf"
        if destination.exists() and not args.force:
            existing = state["files"].get(source_id, {})
            if existing.get("sha256") == _sha256(destination):
                reused += 1
                print(f"[reuse] {source_id}: {destination.name}")
                continue
        print(f"[fetch] {source_id}: {url}")
        try:
            _download(url, destination, timeout=args.timeout, retries=args.retries)
        except Exception as exc:
            failures.append(f"{source_id}: {exc}")
            print(f"[error] {source_id}: {exc}", file=sys.stderr)
            continue
        fetched += 1
        state["files"][source_id] = {
            "source_id": source_id,
            "url": url,
            "file": f"data/raw/{destination.name}",
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "priority": int(source.get("priority", 9)),
            "title": source.get("title"),
            "legal_domain": source.get("legal_domain"),
            "source_language": source.get("source_language"),
            "legal_instrument_id": source.get("legal_instrument_id"),
            "source_locator": source.get("source_locator"),
            "legal_text_type": source.get("legal_text_type"),
            "issuer": source.get("issuer"),
            "reverse_article_digits": bool(source.get("reverse_article_digits", False)),
            "extraction_note": source.get("extraction_note"),
        }
        _write_state(state_path, state)
        print(f"[ok] {source_id}: {destination.stat().st_size} bytes")

    if args.zip:
        archive = ROOT / "data" / "raw_sources.zip"
        count = _zip_raw(args.raw_dir, archive)
        print(f"[zip] {archive.relative_to(ROOT)}: {count} PDF files")

    print(
        json.dumps(
            {
                "selected": len(selected),
                "fetched": fetched,
                "reused": reused,
                "failed": len(failures),
                "failures": failures,
                "raw_manifest": str(state_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
