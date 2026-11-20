"""Download every law published by the Constantine Bar association site.

Source: https://oraconstantine.com/laws/list_pages.php/
This is a SECONDARY, non-official publisher. Texts downloaded here are stored
with legal_status="unknown" and source_tier="secondary" and can only be cited
in informational mode with an explicit non-official-source warning.

For each law we save the raw HTML and a clean UTF-8 text (one source paragraph
per line) plus a SHA-256 of both, so the provenance is fully reproducible.

Usage:
    python scripts/fetch_oraconstantine.py
    python scripts/fetch_oraconstantine.py --max-laws 5
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
LIST_URL = "https://oraconstantine.com/laws/list_pages.php/"
LAWS_BASE = "https://oraconstantine.com/laws/"
DEFAULT_RAW_DIR = ROOT / "data" / "raw" / "oraconstantine"
DEFAULT_EXTRACTED_DIR = ROOT / "data" / "extracted" / "oraconstantine"
DEFAULT_MANIFEST = ROOT / "data" / "oraconstantine_manifest.json"
USER_AGENT = "dz-legal-ai/0.1 (local research corpus builder)"

# A few links on the site's index page are typos. The corrected pages exist
# and were verified manually (HTTP 200 with article headings).
MANUAL_URL_FIXES = {
    "شروط وقrules ممارسة الشعائر الدينية لغير المسلمين": "شروط وقواعد ممارسة الشعائر الدينية لغير المسلمين",
}
# These two index links return HTTP 404 even with corrected spellings; the
# publisher's own table of contents is defective. Kept for audit transparency.
PERMANENTLY_UNAVAILABLE = (
    "التأمين عن البطالة لف",
    "الوقایة الصحیة و الأمن و طب العمل",
)

_HREF = re.compile(r"href=[\"']([^\"']+)[\"']", re.I)
_PARAGRAPH = re.compile(r"<p\b[^>]*>(.*?)</p\s*>", re.I | re.S)
_TAG = re.compile(r"<[^>]+>")
_BOILERPLATE = (
    "الرجوع للصفحة الرئيسية",
    "تصميم و إخراج",
    "منظمة المحامين قسنطينة",
    "اخر تنسيق",
    "آخر تنسيق",
)


def _get_bytes(url: str, timeout: int, retries: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(attempt)
    raise RuntimeError(f"fetch failed after {retries} attempts: {last}")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _law_links(list_html: str) -> list[tuple[str, str]]:
    """Return (source_url, human title) for every distinct law page."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in _HREF.findall(list_html):
        parsed = urllib.parse.urlparse(raw)
        if "laws" not in parsed.path or not parsed.path.endswith(".php"):
            continue
        filename = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
        title = filename[:-4].strip()
        fixed = MANUAL_URL_FIXES.get(title)
        # Trailing spaces are significant in several server-side filenames,
        # so keep the original filename unless a manual fix is needed.
        url_filename = fixed + ".php" if fixed else filename
        title = fixed or title
        if not title:
            continue
        source_url = LAWS_BASE + urllib.parse.quote(url_filename, safe="")
        if source_url in seen:
            continue
        seen.add(source_url)
        links.append((source_url, title))
    return links


def _page_text(page_html: str) -> str:
    """One source paragraph per line; spans inside a <p> are joined verbatim.

    The site frequently splits a single Arabic word across several <span>
    tags (e.g. "قانون العقوبا" + "ت"), so spans must be concatenated without a
    separator before HTML entities are unescaped.
    """
    lines: list[str] = []
    for match in _PARAGRAPH.finditer(page_html):
        joined = "".join(_TAG.sub("", match.group(1)).split("\n"))
        line = html_module.unescape(joined).strip()
        line = unicodedata.normalize("NFKC", line)
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if not line or any(marker in line for marker in _BOILERPLATE):
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


def _slug(source_url: str, index: int) -> str:
    filename = urllib.parse.unquote(urllib.parse.urlparse(source_url).path.rsplit("/", 1)[-1])
    stem = filename[:-4] if filename.endswith(".php") else filename
    ascii_stem = re.sub(r"[^A-Za-z0-9]+", "-", stem).strip("-").lower()
    return f"{index:03d}-{ascii_stem or 'law'}"


def _fetch_one(
    index: int,
    source_url: str,
    title: str,
    raw_dir: Path,
    extracted_dir: Path,
    timeout: int,
    retries: int,
) -> dict[str, Any]:
    page_bytes = _get_bytes(source_url, timeout, retries)
    page_html = page_bytes.decode("utf-8", "replace")
    text = _page_text(page_html)
    text_bytes = text.encode("utf-8")
    slug = _slug(source_url, index)

    raw_path = raw_dir / f"{slug}.html"
    text_path = extracted_dir / f"{slug}.txt"
    raw_path.write_bytes(page_bytes)
    text_path.write_bytes(text_bytes)

    article_count = len(re.findall(r"(?m)^.{0,12}المادة", text))
    return {
        "document_id": f"orac-{slug}",
        "title": title,
        "source_url": source_url,
        "raw_file": str(raw_path.relative_to(ROOT)).replace("\\", "/"),
        "text_file": str(text_path.relative_to(ROOT)).replace("\\", "/"),
        "html_sha256": _sha256(page_bytes),
        "text_sha256": _sha256(text_bytes),
        "characters": len(text),
        "lines": text.count("\n"),
        "article_headings": article_count,
        "status": "text_layer_ok" if article_count > 0 else "no_article_headings",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-laws", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep already downloaded texts and retry only the missing pages",
    )
    args = parser.parse_args(argv)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.extracted_dir.mkdir(parents=True, exist_ok=True)

    documents: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    if args.resume and args.manifest.exists():
        previous = json.loads(args.manifest.read_text(encoding="utf-8"))
        documents.update(previous.get("documents", {}))
        failures.update(previous.get("failures", {}))

    list_html = _get_bytes(LIST_URL, args.timeout, args.retries).decode("utf-8", "replace")
    # Enumerate before filtering so slugs stay stable across resumed runs.
    indexed = list(enumerate(_law_links(list_html), start=1))
    if args.max_laws:
        indexed = indexed[: args.max_laws]
    unavailable = [
        {"source_url": url, "title": title, "reason": "publisher link returns HTTP 404"}
        for _, (url, title) in indexed
        if any(marker in urllib.parse.unquote(url) for marker in PERMANENTLY_UNAVAILABLE)
    ]
    indexed = [
        (index, url, title)
        for index, (url, title) in indexed
        if not any(marker in urllib.parse.unquote(url) for marker in PERMANENTLY_UNAVAILABLE)
    ]
    already = {item["source_url"] for item in documents.values()}
    links = [
        (index, url, title)
        for index, url, title in indexed
        if url not in already
    ]
    print(f"Found {len(links)} law pages to fetch on {LIST_URL} ({len(already)} already saved)")

    failures.clear()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _fetch_one,
                index,
                source_url,
                title,
                args.raw_dir,
                args.extracted_dir,
                args.timeout,
                args.retries,
            ): source_url
            for index, source_url, title in links
        }
        for future in as_completed(futures):
            source_url = futures[future]
            try:
                entry = future.result()
                documents[entry["document_id"]] = entry
            except Exception as exc:  # keep going, one bad page must not abort all
                failures[source_url] = str(exc)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "fetched_on": datetime.now(timezone.utc).isoformat(),
        "list_url": LIST_URL,
        "source_tier": "secondary",
        "note": (
            "Secondary non-official publisher (Constantine Bar association). "
            "Every text is legal_status=unknown and requires human verification "
            "against the official Gazette before it can be treated as current law."
        ),
        "documents": dict(sorted(documents.items())),
        "failures": dict(sorted(failures.items())),
        "unavailable_links": unavailable,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    ok = len(documents)
    with_articles = sum(1 for item in documents.values() if item["status"] == "text_layer_ok")
    total_chars = sum(item["characters"] for item in documents.values())
    print(f"Downloaded {ok}/{len(indexed)} pages; {with_articles} have article headings")
    print(
        f"Total characters: {total_chars:,}; failures: {len(failures)}; "
        f"publisher 404 links: {len(unavailable)}"
    )
    print(f"Manifest: {args.manifest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
