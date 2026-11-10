"""Validation and storage for human-reviewed local source documents."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .policy import (
    ALLOWED_REUSE_BASES,
    LEGAL_DOMAINS,
    LEGAL_TEXT_TYPES,
    validate_reuse_evidence_for,
    validate_source_url,
)

_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_METADATA = (
    "document_id",
    "title",
    "issuer",
    "jurisdiction",
    "source_url",
    "source_language",
    "source_locator",
    "official_source_checked",
    "source_checked_on",
    "verification_note",
    "reuse_basis",
    "reuse_evidence_url",
    "rights_checked_on",
    "rights_verification_note",
    "legal_status",
    "legal_status_checked_on",
    "legal_status_evidence_urls",
    "legal_status_verification_note",
    "legal_domain",
    "legal_instrument_id",
)
_LEGAL_STATUSES = {"verified_current", "historical_or_repealed", "unknown"}


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
    legal_instrument_id = metadata["legal_instrument_id"]
    if not isinstance(legal_instrument_id, str) or not _DOCUMENT_ID.fullmatch(
        legal_instrument_id
    ):
        raise ValueError(
            "legal_instrument_id must be a stable, version-independent legal text identifier"
        )
    for field in ("title", "issuer", "verification_note"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(metadata["rights_verification_note"], str) or not metadata[
        "rights_verification_note"
    ].strip():
        raise ValueError("rights_verification_note must be a non-empty string")
    for field in ("source_checked_on", "rights_checked_on"):
        try:
            parsed_date = date.fromisoformat(metadata[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD)") from exc
        if parsed_date.isoformat() != metadata[field]:
            raise ValueError(f"{field} must use the YYYY-MM-DD format")
    legal_status = metadata["legal_status"]
    if not isinstance(legal_status, str) or legal_status not in _LEGAL_STATUSES:
        raise ValueError("legal_status must be verified_current, historical_or_repealed, or unknown")
    status_checked_on = metadata["legal_status_checked_on"]
    status_evidence_urls = metadata["legal_status_evidence_urls"]
    if not isinstance(status_evidence_urls, list):
        raise ValueError("legal_status_evidence_urls must be a list")
    if legal_status == "unknown":
        if status_checked_on is not None or status_evidence_urls:
            raise ValueError(
                "Unknown legal status must not claim a review date or status evidence"
            )
    else:
        try:
            parsed_status_date = date.fromisoformat(status_checked_on)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Verified legal status requires legal_status_checked_on as an ISO date"
            ) from exc
        if parsed_status_date.isoformat() != status_checked_on:
            raise ValueError("legal_status_checked_on must use the YYYY-MM-DD format")
        if not status_evidence_urls:
            raise ValueError("Verified legal status requires official evidence URLs")
        for evidence_url in status_evidence_urls:
            validate_source_url(evidence_url)
    if not isinstance(metadata["legal_status_verification_note"], str) or not metadata[
        "legal_status_verification_note"
    ].strip():
        raise ValueError("legal_status_verification_note must be non-empty")
    if metadata["jurisdiction"] != "DZ":
        raise ValueError("Only Algerian jurisdiction (DZ) is supported")
    if not isinstance(metadata["legal_domain"], str) or metadata[
        "legal_domain"
    ] not in LEGAL_DOMAINS:
        raise ValueError("legal_domain must be a supported Algerian law domain")
    if metadata["official_source_checked"] is not True:
        raise ValueError("A human reviewer must confirm official_source_checked=true")
    if not isinstance(metadata["source_url"], str):
        raise ValueError("source_url must be a URL string")
    if not isinstance(metadata["source_language"], str) or metadata[
        "source_language"
    ] not in {"ar", "fr"}:
        raise ValueError("source_language must be ar or fr")
    if not isinstance(metadata["source_locator"], str) or not metadata[
        "source_locator"
    ].strip():
        raise ValueError("source_locator must identify the source issue and text location")
    source_document_sha256 = metadata.get("source_document_sha256")
    if source_document_sha256 is not None and (
        not isinstance(source_document_sha256, str)
        or not _SHA256.fullmatch(source_document_sha256)
    ):
        raise ValueError("source_document_sha256 must be a lowercase SHA-256 digest")
    extraction_method = metadata.get("extraction_method")
    if extraction_method is not None and (
        not isinstance(extraction_method, str) or not extraction_method.strip()
    ):
        raise ValueError("extraction_method must be a non-empty string when provided")
    reuse_basis = metadata["reuse_basis"]
    if not isinstance(reuse_basis, str) or reuse_basis not in ALLOWED_REUSE_BASES:
        raise ValueError("Reuse basis is unknown or unsupported; do not ingest this material")
    if not isinstance(metadata["reuse_evidence_url"], str):
        raise ValueError("reuse_evidence_url must be a URL string")
    legal_text_type = metadata.get("legal_text_type")
    if reuse_basis == "DZ-ORD-03-05-ART-11" and (
        not isinstance(legal_text_type, str) or legal_text_type not in LEGAL_TEXT_TYPES
    ):
        raise ValueError(
            "The Algerian statutory reuse basis requires a supported legal_text_type"
        )

    validate_source_url(metadata["source_url"])
    validate_reuse_evidence_for(reuse_basis, metadata["reuse_evidence_url"])
    normalized_text = text.strip()
    record = {
        "document_id": document_id,
        "title": metadata["title"].strip(),
        "issuer": metadata["issuer"].strip(),
        "jurisdiction": "DZ",
        "legal_domain": metadata["legal_domain"],
        "legal_instrument_id": legal_instrument_id,
        "source_url": metadata["source_url"],
        "source_language": metadata["source_language"],
        "source_locator": metadata["source_locator"].strip(),
        "official_source_checked": True,
        "source_checked_on": metadata["source_checked_on"],
        "verification_note": metadata["verification_note"].strip(),
        "reuse_basis": reuse_basis,
        "reuse_evidence_url": metadata["reuse_evidence_url"],
        "rights_checked_on": metadata["rights_checked_on"],
        "rights_verification_note": metadata["rights_verification_note"].strip(),
        "legal_status": legal_status,
        "legal_status_checked_on": status_checked_on,
        "legal_status_evidence_urls": status_evidence_urls,
        "legal_status_verification_note": metadata[
            "legal_status_verification_note"
        ].strip(),
        "legal_text_type": legal_text_type,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "text": normalized_text,
        "content_sha256": hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
    }
    if source_document_sha256 is not None:
        record["source_document_sha256"] = source_document_sha256
    if extraction_method is not None:
        record["extraction_method"] = extraction_method.strip()
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


def build_seed_corpus(corpus_path: Path) -> tuple[list[dict[str, Any]], int]:
    root = Path(__file__).resolve().parents[2]
    seed_directory = root / "data" / "seed"
    seed_files = (
        ("ordinance-03-05-fr.txt", "ordinance-03-05-fr.metadata.json"),
        ("constitution-2020-fr.txt", "constitution-2020-fr.metadata.json"),
    )
    records = read_corpus(corpus_path)
    by_id = {record["document_id"]: record for record in records}
    added = 0
    for text_name, metadata_name in seed_files:
        text_path = seed_directory / text_name
        metadata_path = seed_directory / metadata_name
        try:
            text = text_path.read_text(encoding="utf-8")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read bundled seed source: {text_name}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"Seed metadata must be a JSON object: {metadata_name}")
        record = build_record(text, metadata)
        existing = by_id.get(record["document_id"])
        if existing is not None:
            if (
                existing.get("content_sha256") != record["content_sha256"]
                or existing.get("source_document_sha256")
                != record.get("source_document_sha256")
            ):
                raise ValueError(
                    f"Existing corpus has conflicting seed document: {record['document_id']}"
                )
            continue
        records.append(record)
        by_id[record["document_id"]] = record
        added += 1

    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    temporary_path.replace(corpus_path)
    return records, added


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
