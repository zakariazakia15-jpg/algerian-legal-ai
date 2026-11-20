"""Ingest extracted official texts into the local answer corpus.

Every document is stored with legal_status=unknown. It can be retrieved only
in informational mode (--allow-unverified) until a qualified reviewer confirms
the current legal status against the official Gazette.

Usage:
    python scripts/ingest_official_sources.py
    python scripts/ingest_official_sources.py --only civil-code-fr penal-code-fr
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTED = ROOT / "data" / "extracted"
DEFAULT_RAW_MANIFEST = ROOT / "data" / "raw" / "manifest.json"
DEFAULT_CORPUS = ROOT / "data" / "corpus.jsonl"
DEFAULT_INVENTORY = ROOT / "data" / "legal_source_inventory.json"
REUSE_BASIS = "DZ-ORD-03-05-ART-11"
REUSE_EVIDENCE_URL = "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args(argv)

    # Imported here so PYTHONPATH=src is not required when running only the
    # download/extract steps.
    sys.path.insert(0, str(ROOT / "src"))
    from dz_legal_ai.ingest import build_record, read_corpus

    extraction = _load_json(args.extracted_dir / "manifest.json")
    raw = _load_json(args.raw_manifest)
    raw_files: dict[str, Any] = raw.get("files", {})
    documents = extraction.get("documents", {})
    checked_on = date.today().isoformat()

    records = read_corpus(args.corpus)
    by_id = {record["document_id"]: record for record in records}
    added = 0
    skipped_existing = 0
    ingested: list[str] = []
    ingested_text_files: dict[str, str] = {}
    ingested_limitations: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "ingested_on": checked_on,
        "note": "All texts are legal_status=unknown and answer-eligible only in informational mode.",
        "documents": {},
    }

    for source_id in sorted(documents):
        if args.only and source_id not in args.only:
            continue
        entry = documents[source_id]
        if entry.get("status") != "text_layer_ok":
            continue
        raw_entry = raw_files.get(source_id)
        if raw_entry is None:
            print(f"[skip] {source_id}: missing raw provenance", file=sys.stderr)
            continue
        text_path = ROOT / entry["text_file"]
        text = text_path.read_text(encoding="utf-8")
        if source_id in by_id:
            existing = by_id[source_id]
            if existing.get("content_sha256") != entry.get("text_sha256"):
                raise ValueError(f"Corpus already has a different text for {source_id}")
            skipped_existing += 1
            continue

        metadata: dict[str, Any] = {
            "document_id": source_id,
            "title": raw_entry["title"],
            "issuer": raw_entry["issuer"],
            "jurisdiction": "DZ",
            "source_url": raw_entry["url"],
            "source_language": raw_entry["source_language"],
            "source_locator": raw_entry["source_locator"],
            "official_source_checked": True,
            "source_checked_on": checked_on,
            "verification_note": (
                "Downloaded from the official Algerian Ministry of Justice legal portal; "
                "SHA-256 recorded in data/raw/manifest.json. Text layer extracted with pypdf "
                f"and checked for per-page density ({entry['pages_with_text']}/{entry['pages']} pages). "
                "This is not a visual, article-by-article proofread."
            ),
            "reuse_basis": REUSE_BASIS,
            "legal_text_type": raw_entry.get("legal_text_type") or "law",
            "reuse_evidence_url": REUSE_EVIDENCE_URL,
            "rights_checked_on": checked_on,
            "rights_verification_note": (
                "Official Algerian legal text retrieved from a government portal. Reuse is "
                "documented under the statutory exception for official legal texts in Article 11 "
                "of Ordinance 03-05; this is a technical documentation basis, not a legal opinion."
            ),
            "legal_status": "unknown",
            "legal_status_checked_on": None,
            "legal_status_evidence_urls": [],
            "legal_status_verification_note": (
                "Download date, consolidations and possible later amendments have not been "
                "verified against the official Gazette by a qualified reviewer."
            ),
            "legal_domain": raw_entry["legal_domain"],
            "legal_instrument_id": raw_entry["legal_instrument_id"],
            "source_document_sha256": raw_entry["sha256"],
            "extraction_method": entry["extraction_method"],
        }
        record = build_record(text, metadata)
        records.append(record)
        by_id[source_id] = record
        added += 1
        ingested.append(source_id)
        report["documents"][source_id] = {
            "document_id": source_id,
            "title": record["title"],
            "legal_domain": record["legal_domain"],
            "source_language": record["source_language"],
            "source_url": record["source_url"],
            "source_document_sha256": record["source_document_sha256"],
            "content_sha256": record["content_sha256"],
            "pages": entry["pages"],
            "pages_with_text": entry["pages_with_text"],
            "low_text_pages": entry["low_text_pages"],
            "legal_status": "unknown",
        }
        ingested_text_files[source_id] = entry["text_file"]
        ingested_limitations[source_id] = (
            "Text-layer extraction from the Ministry of Justice portal; not visually "
            "proofread, amendments and current force unverified."
        )
        print(f"[ingest] {source_id}: {len(text)} chars -> corpus")

    # The Arabic Penal Code is reconstructed from the official Ministry of
    # Justice HTML portal because its scanned PDF has no usable text layer.
    # Provenance for that reconstruction lives in a separate manifest written
    # by scripts/fetch_html_portal.py.
    html_manifest_path = args.extracted_dir / "html_portal_manifest.json"
    html_manifest: dict[str, Any] | None = None
    if html_manifest_path.is_file() and (
        not args.only or "penal-code-ar-portal" in args.only
    ):
        html_manifest = _load_json(html_manifest_path)
        source_id = "penal-code-ar-portal"
        text_path = ROOT / html_manifest["text_file"]
        text = text_path.read_text(encoding="utf-8")
        if source_id in by_id:
            if by_id[source_id].get("content_sha256") != html_manifest["text_sha256"]:
                raise ValueError(f"Corpus already has a different text for {source_id}")
            skipped_existing += 1
        else:
            metadata: dict[str, Any] = {
                "document_id": source_id,
                "title": "قانون العقوبات الجزائري (نسخة بوابة وزارة العدل HTML)",
                "issuer": "وزارة العدل - بوابة القانون الجزائري",
                "jurisdiction": "DZ",
                "source_url": html_manifest["base_url"],
                "source_language": "ar",
                "source_locator": (
                    "بوابة القانون الجزائري، قانون العقوبات (الأمر رقم 66-156 "
                    "المعدل والمتمم)، نسخة HTML، "
                    f"{html_manifest['extracted_page_count']}/{html_manifest['page_count']} صفحة"
                ),
                "official_source_checked": True,
                "source_checked_on": checked_on,
                "verification_note": (
                    "Reconstructed from the official Ministry of Justice HTML portal "
                    f"({html_manifest['page_count']} pages, {html_manifest['extracted_page_count']} "
                    "extracted, no OCR). SHA-256 of the page list and of the text is recorded. "
                    "This is not a visual, article-by-article proofread; portal page order and "
                    "cross-references can leave a few mislabelled short chunks."
                ),
                "reuse_basis": REUSE_BASIS,
                "legal_text_type": "law",
                "reuse_evidence_url": REUSE_EVIDENCE_URL,
                "rights_checked_on": checked_on,
                "rights_verification_note": (
                    "Official Algerian legal text retrieved from a government portal. Reuse is "
                    "documented under the statutory exception for official legal texts in Article 11 "
                    "of Ordinance 03-05; this is a technical documentation basis, not a legal opinion."
                ),
                "legal_status": "unknown",
                "legal_status_checked_on": None,
                "legal_status_evidence_urls": [],
                "legal_status_verification_note": (
                    "Download date, consolidations and possible later amendments have not been "
                    "verified against the official Gazette by a qualified reviewer."
                ),
                "legal_domain": "criminal",
                "legal_instrument_id": "DZ-PENAL-CODE",
                "source_document_sha256": html_manifest["page_list_sha256"],
                "extraction_method": (
                    "HTML topic reconstruction (windows-1256 decoded, scripts/navigation stripped, "
                    "soft breaks flattened), UTF-8 text with recorded SHA-256"
                ),
            }
            record = build_record(text, metadata)
            records.append(record)
            by_id[source_id] = record
            added += 1
            ingested.append(source_id)
            raw_files[source_id] = {
                "title": metadata["title"],
                "url": html_manifest["base_url"],
                "sha256": html_manifest["page_list_sha256"],
                "legal_domain": "criminal",
                "source_language": "ar",
            }
            ingested_text_files[source_id] = html_manifest["text_file"]
            ingested_limitations[source_id] = (
                "Official HTML portal reconstruction (no OCR). Not visually proofread; "
                "some short cross-reference chunks may be mislabelled; current force and "
                "amendments unverified."
            )
            report["documents"][source_id] = {
                "document_id": source_id,
                "title": record["title"],
                "legal_domain": record["legal_domain"],
                "source_language": record["source_language"],
                "source_url": record["source_url"],
                "source_document_sha256": record["source_document_sha256"],
                "content_sha256": record["content_sha256"],
                "pages": html_manifest["page_count"],
                "pages_with_text": html_manifest["extracted_page_count"],
                "low_text_pages": [],
                "legal_status": "unknown",
                "note": "HTML portal reconstruction; no OCR involved.",
            }
            print(f"[ingest] {source_id}: {len(text)} chars -> corpus")

    if added:
        temporary = args.corpus.with_suffix(args.corpus.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        temporary.replace(args.corpus)

    # Keep the discovery log in sync without turning these into candidates.
    if args.inventory.exists() and added:
        inventory = _load_json(args.inventory)
        known_ids = {
            item.get("source_id")
            for item in inventory.get("sources", [])
            if isinstance(item, dict)
        }
        for source_id in ingested:
            if source_id in known_ids:
                continue
            raw_entry = raw_files[source_id]
            report_entry = report["documents"][source_id]
            inventory["sources"].append(
                {
                    "source_id": source_id,
                    "reference": raw_entry["title"],
                    "source_url": raw_entry["url"],
                    "source_document_sha256": raw_entry["sha256"],
                    "status": "included_official_text_unverified",
                    "legal_status": "unknown",
                    "legal_domain": raw_entry["legal_domain"],
                    "extracted_text_file": ingested_text_files[source_id],
                    "limitations": ingested_limitations[source_id],
                    "not_included_reason": (
                        "Included in the local corpus as legal_status=unknown only; answer use "
                        "requires informational mode until a qualified reviewer verifies the "
                        "text and its current force."
                    ),
                }
            )
        _write_json(args.inventory, inventory)

    report["added"] = added
    report["skipped_existing"] = skipped_existing
    report["corpus_documents"] = len(records)
    report_path = ROOT / "data" / "official_corpus_report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "added": added,
                "skipped_existing": skipped_existing,
                "corpus_documents": len(records),
                "report": str(report_path.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
