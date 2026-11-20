"""Download the Ministry of Justice HTML portal edition of the Algerian Penal Code.

The scanned PDF of the Penal Code has no usable text layer. The same official
portal publishes a complete HTML edition (windows-1256) in framed topics.
This script reconstructs a clean UTF-8 text from those topics and records the
source URL and a SHA-256 of the resulting text. No OCR is involved.

Usage:
    python scripts/fetch_html_portal.py
    python scripts/fetch_html_portal.py --max-pages 20
"""

from __future__ import annotations

import argparse
import hashlib
import html as html_module
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    "https://droit.mjustice.gov.dz/sites/default/files/portail/"
    "legisl_nouveau/code_penal_2010/ar/"
)
USER_AGENT = "dz-legal-ai/0.1 (local research corpus builder)"
_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script>", re.I | re.S)
_STYLE = re.compile(r"<style\b[^>]*>.*?</style>", re.I | re.S)
_BREADCRUMB = re.compile(r"<p[^>]*>\s*(?:<span[^>]*>)?\s*أنت هنا\s*:.*?</p>", re.I | re.S)
_NAV_LINK = re.compile(r"<p[^>]*>\s*<a\s+href=[^>]+>.*?</a>.*?</p>", re.I | re.S)
_BLOCK_END = re.compile(r"</(?:p|div|h[1-6]|li|tr|td)\s*>", re.I)
# A <br> is a soft line break inside an article. Turning it into a newline
# makes cross-references like "المادة 121 من قانون..." look like article
# headings; a space keeps them inside their own article text.
_LINE_BREAK = re.compile(r"<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")


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


def _get(url: str, timeout: int, retries: int) -> str:
    return _get_bytes(url, timeout, retries).decode("cp1256", errors="replace")


def _page_text(page_html: str) -> str:
    body = re.search(r"<body\b[^>]*>(.*?)</body>", page_html, re.I | re.S)
    content = body.group(1) if body else page_html
    content = html_module.unescape(content)
    content = _SCRIPT.sub(" ", content)
    content = _STYLE.sub(" ", content)
    content = _BREADCRUMB.sub(" ", content)
    content = _NAV_LINK.sub(" ", content)
    content = _LINE_BREAK.sub(" ", content)
    content = _BLOCK_END.sub("\n", content)
    content = _TAG.sub(" ", content)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in content.splitlines()]
    return "\n".join(line for line in lines if line and line != "highlight();").strip()


def _page_list(base: str, timeout: int, retries: int) -> tuple[list[str], str]:
    page_list_url = base + "scripts/HelpCommonData.js"
    raw = _get_bytes(page_list_url, timeout, retries)
    data = raw.decode("cp1256", errors="replace")
    match = re.search(r"var\s+ps\s*=\s*\[(.*?)\];", data, re.S)
    if match is None:
        raise ValueError("Could not read page list from HelpCommonData.js")
    pages = re.findall(r'"([^"]+)"', match.group(1))
    return (
        [page for page in pages if page.lower().endswith(".html")],
        hashlib.sha256(raw).hexdigest(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument(
        "--output", default=str(ROOT / "data" / "extracted" / "penal-code-ar-portal.txt")
    )
    parser.add_argument("--manifest", default=str(ROOT / "data" / "extracted" / "html_portal_manifest.json"))
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    pages, page_list_sha256 = _page_list(args.base, args.timeout, args.retries)
    if args.max_pages is not None:
        pages = pages[: args.max_pages]
    print(f"[info] {len(pages)} portal pages", file=sys.stderr)

    results: dict[int, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_get, args.base + "topics/" + page, args.timeout, args.retries): (
                index,
                page,
            )
            for index, page in enumerate(pages)
        }
        for future in as_completed(futures):
            index, page = futures[future]
            try:
                results[index] = _page_text(future.result())
            except Exception as exc:
                failures.append(f"{page}: {exc}")

    sections = [results[index] for index in sorted(results) if results[index]]
    text = "\n\n".join(sections).strip()
    if not text:
        raise SystemExit("No text extracted from the portal")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = {
        "schema_version": 1,
        "source": "Algerian Ministry of Justice HTML portal, Penal Code Arabic edition",
        "base_url": args.base,
        "page_list_url": args.base + "scripts/HelpCommonData.js",
        "page_list_sha256": page_list_sha256,
        "encoding": "windows-1256",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "extracted_page_count": len(sections),
        "failed_pages": failures,
        "text_file": str(output_path.relative_to(ROOT)),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "characters": len(text),
        "note": (
            "Official portal HTML reconstruction, not a scan or OCR. Current "
            "force and amendments still require qualified human review."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pages": len(pages),
                "extracted": len(sections),
                "failed": len(failures),
                "characters": len(text),
                "text_file": str(output_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
