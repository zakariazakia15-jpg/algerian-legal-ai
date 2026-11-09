"""Validation and storage for human-reviewed local source documents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .policy import ALLOWED_LICENSES, validate_license_evidence_for, validate_source_url

_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_REQUIRED_METADATA = (
    "document_id",
    "title",
    "issuer",
    "jurisdiction",
    "source_url",
    "official_source_checked",
    "source_checked_on",
    "verification_note",
    "license",
    "license_evidence_url",
    "license_checked_on",
    "license_verification_note",
)


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        {key: value for key, value in record.items() if key != "record_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_record(text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"Missing metadata fields: {', '.join(missing)}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Source text must be a non-empty UTF-8 string")
    document_id = metadata["document_id"]
    if not isinstance(document_id, str) or not _DOCUMENT_ID.fullmatch(document_id):
        raise ValueError("document_id must contain only letters, digits, dot, underscore, or hyphen")
    for field in ("title", "issuer", "verification_note"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(metadata["license_verification_note"], str) or not metadata[
        "license_verification_note"
    ].strip():
        raise ValueError("license_verification_note must be a non-empty string")
    for field in ("source_checked_on", "license_checked_on"):
        try:
            date.fromisoformat(metadata[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)") from exc
    if metadata["jurisdiction"] != "DZ":
        raise ValueError("Only Algerian jurisdiction (DZ) is supported")
    if metadata["official_source_checked"] is not True:
        raise ValueError("A human reviewer must confirm official_source_checked=true")
    if not isinstance(metadata["source_url"], str):
        raise ValueError("source_url must be a URL string")
    if not isinstance(metadata["license"], str) or metadata["license"] not in ALLOWED_LICENSES:
        raise ValueError("License is unknown or unsupported; do not ingest this material")
    if not isinstance(metadata["license_evidence_url"], str):
        raise ValueError("license_evidence_url must be a URL string")

    validate_source_url(metadata["source_url"])
    validate_license_evidence_for(metadata["license"], metadata["license_evidence_url"])
    normalized_text = text.strip()
    record = {
        "document_id": document_id,
        "title": metadata["title"].strip(),
        "issuer": metadata["issuer"].strip(),
        "jurisdiction": "DZ",
        "source_url": metadata["source_url"],
        "official_source_checked": True,
        "source_checked_on": metadata["source_checked_on"],
        "verification_note": metadata["verification_note"].strip(),
        "license": metadata["license"],
        "license_evidence_url": metadata["license_evidence_url"],
        "license_checked_on": metadata["license_checked_on"],
        "license_verification_note": metadata["license_verification_note"].strip(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "text": normalized_text,
        "content_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
    }
    record["record_sha256"] = _record_digest(record)
    return record


def ingest_files(text_path: Path, metadata_path: Path, corpus_path: Path) -> dict[str, Any]:
    text = text_path.read_text(encoding="utf-8")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must be an object")
    record = build_record(text, metadata)

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    records = read_corpus(corpus_path)
    if any(item["document_id"] == record["document_id"] for item in records):
        raise ValueError(f"Document already exists: {record['document_id']}")
    records.append(record)
    temporary_path = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary_path.replace(corpus_path)
    return record


def read_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    if not corpus_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid corpus JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Corpus line {line_number} must be an object")
        validated = build_record(record.get("text", ""), record)
        if validated["content_sha256"] != record.get("content_sha256"):
            raise ValueError(f"Content hash mismatch on corpus line {line_number}")
        if _record_digest(record) != record.get("record_sha256"):
            raise ValueError(f"Record metadata hash mismatch on corpus line {line_number}")
        records.append(record)
    return records
