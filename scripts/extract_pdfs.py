"""Extract searchable text from the downloaded official PDFs with pypdf.

The script records a per-page text-density report. Pages with little or no
text are flagged as requiring OCR; their PDFs are NOT turned into corpus
records because the project has no validated Arabic OCR pipeline.

Usage:
    python scripts/extract_pdfs.py
    python scripts/extract_pdfs.py --only civil-code-fr
    python scripts/extract_pdfs.py --include-low-text --min-chars 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_OUT_DIR = ROOT / "data" / "extracted"
_MIN_PAGE_CHARS = 50
_MIN_TOTAL_CHARS = 500
_MIN_PAGE_RATIO = 0.60
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_ARTICLE_NUMBER = re.compile(r"(المادة[ \t]*[:：]?[ \t]*)(\d+)", re.UNICODE)


def _reverse_article_numbers(text: str) -> str:
    """Fix RTL digit reversal in article headings of one known PDF source."""

    def replace(match: re.Match[str]) -> str:
        return match.group(1) + match.group(2)[::-1]

    return _ARTICLE_NUMBER.sub(replace, text)


def _clean_page(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ARABIC_DIGITS)
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_pages(pdf_path: Path, *, reverse_article_digits: bool = False) -> tuple[list[str], str]:
    """Prefer PyMuPDF: it preserves Arabic reading order better than pypdf."""
    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            pymupdf = None
    if pymupdf is not None:
        document = pymupdf.open(str(pdf_path))
        pages = [_clean_page(page.get_text("text")) for page in document]
        if reverse_article_digits:
            pages = [_reverse_article_numbers(page) for page in pages]
        method = "PyMuPDF get_text(text), NFKC normalized, Arabic digits to ASCII"
        document.close()
        return pages, method

    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(_clean_page(page.extract_text() or ""))
        except Exception as exc:  # a damaged page must not abort the whole PDF
            pages.append("")
            print(f"[warn] {pdf_path.name}: page {len(pages)} extraction failed: {exc}", file=sys.stderr)
    return pages, "pypdf PdfReader.extract_text, NFKC normalized, Arabic digits to ASCII"


def _extract_one(pdf_path: Path, *, reverse_article_digits: bool = False) -> dict[str, Any]:
    pages, method = _extract_pages(pdf_path, reverse_article_digits=reverse_article_digits)
    if reverse_article_digits:
        method += "; reversed article numbers corrected"
    total_pages = len(pages) or 1
    usable = [index + 1 for index, text in enumerate(pages) if len(text) >= _MIN_PAGE_CHARS]
    low_text = [index + 1 for index, text in enumerate(pages) if len(text) < _MIN_PAGE_CHARS]
    text = "\n\n".join(f"[صفحة {index + 1}]\n{pages[index]}" for index in range(len(pages)))
    return {
        "pages": len(pages),
        "pages_with_text": len(usable),
        "page_text_ratio": round(len(usable) / total_pages, 4),
        "low_text_pages": low_text[:80],
        "low_text_page_count": len(low_text),
        "characters": len(text),
        "text": text,
        "extraction_method": method + "; per-page text markers retained",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--include-low-text", action="store_true")
    parser.add_argument("--min-page-ratio", type=float, default=_MIN_PAGE_RATIO)
    parser.add_argument("--min-chars", type=int, default=_MIN_TOTAL_CHARS)
    args = parser.parse_args(argv)

    raw_manifest_path = args.raw_dir / "manifest.json"
    if not raw_manifest_path.exists():
        raise SystemExit(f"Run fetch_official_sources.py first: {raw_manifest_path}")
    state = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    files: dict[str, Any] = state.get("files", {})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "manifest.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "min_page_ratio": args.min_page_ratio,
            "min_chars": args.min_chars,
            "include_low_text": args.include_low_text,
        },
        "documents": {},
    }

    ready = 0
    needs_ocr = 0
    missing = 0
    for source_id, entry in sorted(files.items()):
        if args.only and source_id not in args.only:
            continue
        pdf_path = ROOT / str(entry["file"])
        if not pdf_path.exists():
            missing += 1
            print(f"[missing] {source_id}: {pdf_path}", file=sys.stderr)
            continue
        print(f"[extract] {source_id}: {pdf_path.name}")
        try:
            result = _extract_one(
                pdf_path,
                reverse_article_digits=bool(entry.get("reverse_article_digits", False)),
            )
        except Exception as exc:
            print(f"[error] {source_id}: {exc}", file=sys.stderr)
            report["documents"][source_id] = {"status": "error", "error": str(exc)}
            continue
        has_text = (
            result["page_text_ratio"] >= args.min_page_ratio
            and result["characters"] >= args.min_chars
        )
        status = "text_layer_ok" if has_text else "needs_ocr_or_review"
        if not has_text and not args.include_low_text:
            needs_ocr += 1
            print(
                f"[needs-ocr] {source_id}: {result['pages_with_text']}/{result['pages']} "
                f"pages with text, {result['characters']} chars"
            )
            report["documents"][source_id] = {
                "status": status,
                "pages": result["pages"],
                "pages_with_text": result["pages_with_text"],
                "page_text_ratio": result["page_text_ratio"],
                "characters": result["characters"],
                "low_text_pages": result["low_text_pages"],
                "reason": "No usable text layer; do not ingest without validated OCR and human proofreading.",
            }
            continue

        text_path = args.out_dir / f"{source_id}.txt"
        text_path.write_text(result["text"], encoding="utf-8")
        ready += 1
        report["documents"][source_id] = {
            "status": "text_layer_ok",
            "text_file": str(text_path.relative_to(ROOT)),
            "text_sha256": _sha256_text(result["text"]),
            "pages": result["pages"],
            "pages_with_text": result["pages_with_text"],
            "page_text_ratio": result["page_text_ratio"],
            "characters": result["characters"],
            "low_text_pages": result["low_text_pages"],
            "extraction_method": result["extraction_method"],
            "source_pdf_sha256": entry.get("sha256"),
            "source_url": entry.get("url"),
        }
        print(
            f"[ok] {source_id}: {result['characters']} chars, "
            f"{result['pages_with_text']}/{result['pages']} pages with text"
        )

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "ready_for_ingest": ready,
                "needs_ocr_or_review": needs_ocr,
                "missing": missing,
                "report": str(report_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
