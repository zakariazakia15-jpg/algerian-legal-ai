"""Local aggregation of unreviewed sources, isolated from the answer corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .ingest import build_record
from .policy import is_foreign_jurisdiction_query
from .retrieve import retrieve

DEFAULT_INVENTORY = Path("data/legal_source_inventory.json")
DEFAULT_CANDIDATE_CORPUS = Path("data/candidate-corpus.jsonl")
_ARTICLE_PATTERNS = {
    "ar": re.compile(r"المادة\s*[:：]?\s*(\d+)"),
    "fr": re.compile(r"(?i)\bArt\.\s*(\d+)\b"),
}
_TITLE_TOKENS = re.compile(r"[^\W_]+", re.UNICODE)


def _record_digest(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        {key: value for key, value in record.items() if key != "candidate_record_sha256"},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _candidate_paths(inventory_path: Path, source: dict[str, Any]) -> tuple[Path, Path]:
    root = inventory_path.resolve().parent.parent
    text_relative = source.get("text_file")
    metadata_relative = source.get("metadata_file")
    if not isinstance(text_relative, str) or not isinstance(metadata_relative, str):
        raise ValueError(f"Candidate source is missing text or metadata path: {source.get('source_id')}")

    text_path = (root / text_relative).resolve()
    metadata_path = (root / metadata_relative).resolve()
    candidate_root = (root / "data" / "candidates").resolve()
    if not text_path.is_relative_to(candidate_root) or not metadata_path.is_relative_to(
        candidate_root
    ):
        raise ValueError(f"Candidate paths must remain under data/candidates: {source.get('source_id')}")
    return text_path, metadata_path


def _audit_text(text: str, language: str, title: str) -> dict[str, Any]:
    references = [int(value) for value in _ARTICLE_PATTERNS[language].findall(text)]
    counts = Counter(references)
    normalized_text = " ".join(text.split()).casefold()
    title_tokens = [
        token
        for token in _TITLE_TOKENS.findall(title.casefold())
        if len(token) >= 3 or token.isdigit()
    ]
    text_tokens = set(_TITLE_TOKENS.findall(normalized_text))
    found_title_tokens = [token for token in title_tokens if token in text_tokens]
    return {
        "utf8_valid": True,
        "replacement_characters": text.count("\ufffd"),
        "text_characters": len(text),
        "title_tokens_found": found_title_tokens,
        "title_tokens_missing": [
            token for token in title_tokens if token not in text_tokens
        ],
        "title_token_coverage": (
            round(len(found_title_tokens) / len(title_tokens), 3) if title_tokens else 0
        ),
        "article_reference_count": len(references),
        "unique_article_reference_count": len(counts),
        "minimum_article_reference": min(counts) if counts else None,
        "maximum_article_reference": max(counts) if counts else None,
        "duplicate_article_references": sorted(
            reference for reference, count in counts.items() if count > 1
        ),
        "first_article_references": references[:10],
        "last_article_references": references[-10:],
    }


def _build_record(
    inventory_path: Path,
    inventory: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    text_path, metadata_path = _candidate_paths(inventory_path, source)
    try:
        source_text_bytes = text_path.read_bytes()
        text = source_text_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Candidate text must be readable UTF-8: {source.get('source_id')}") from exc
    metadata = _read_json_object(metadata_path, "candidate metadata")
    text_sha256 = hashlib.sha256(source_text_bytes).hexdigest()

    if inventory.get("jurisdiction") != "DZ":
        raise ValueError("Candidate inventory must be limited to Algerian jurisdiction (DZ)")
    if not source.get("status", "").startswith("candidate_"):
        raise ValueError(f"Source is not marked as a candidate: {source.get('source_id')}")
    if source.get("legal_status") != "unknown" or metadata.get("legal_status") != "unknown":
        raise ValueError(f"Candidate legal status must remain unknown: {source.get('source_id')}")
    if metadata.get("jurisdiction") != "DZ":
        raise ValueError(f"Candidate metadata must be limited to jurisdiction DZ: {source.get('source_id')}")
    expected_document_id = source.get("candidate_document_id", source.get("source_id"))
    if metadata.get("document_id") != expected_document_id:
        raise ValueError(f"Candidate ID does not match its inventory record: {source.get('source_id')}")
    if metadata.get("source_url") != source.get("source_url"):
        raise ValueError(f"Candidate source URL mismatch: {source.get('source_id')}")
    if metadata.get("source_document_sha256") != source.get("source_document_sha256"):
        raise ValueError(f"Candidate source-document hash mismatch: {source.get('source_id')}")
    if metadata.get("candidate_text_sha256") != source.get("candidate_text_sha256"):
        raise ValueError(f"Candidate-text hash metadata mismatch: {source.get('source_id')}")
    if text_sha256 != source.get("candidate_text_sha256"):
        raise ValueError(f"Candidate text hash mismatch: {source.get('source_id')}")
    language = metadata.get("source_language")
    if language not in _ARTICLE_PATTERNS:
        raise ValueError(f"Unsupported candidate language: {source.get('source_id')}")
    if not text.strip():
        raise ValueError(f"Candidate text is empty: {source.get('source_id')}")
    build_record(text, metadata)

    record = {
        "schema_version": 1,
        "corpus_kind": "candidate_unreviewed",
        "candidate_only": True,
        "answer_eligible": False,
        "training_eligible": False,
        "review_state": "unreviewed",
        "source_id": source["source_id"],
        "document_id": metadata["document_id"],
        "title": metadata["title"],
        "issuer": metadata["issuer"],
        "jurisdiction": "DZ",
        "legal_domain": metadata["legal_domain"],
        "legal_instrument_id": metadata["legal_instrument_id"],
        "legal_status": "unknown",
        "source_url": metadata["source_url"],
        "source_language": language,
        "source_locator": metadata["source_locator"],
        "source_checked_on": metadata["source_checked_on"],
        "source_document_sha256": metadata["source_document_sha256"],
        "candidate_text_sha256": text_sha256,
        "verification_note": metadata["verification_note"],
        "rights_verification_note": metadata["rights_verification_note"],
        "legal_status_verification_note": metadata["legal_status_verification_note"],
        "extraction_method": metadata.get("extraction_method"),
        "text_audit": _audit_text(text, language, metadata["title"]),
        "text": text,
    }
    record["candidate_record_sha256"] = _record_digest(record)
    return record


def _validate_candidate_record(record: dict[str, Any], line_number: int) -> None:
    if (
        record.get("schema_version") != 1
        or record.get("corpus_kind") != "candidate_unreviewed"
        or record.get("candidate_only") is not True
        or record.get("answer_eligible") is not False
        or record.get("training_eligible") is not False
        or record.get("review_state") != "unreviewed"
        or record.get("jurisdiction") != "DZ"
        or record.get("legal_status") != "unknown"
    ):
        raise ValueError(f"Candidate safety flags are invalid on line {line_number}")
    if not isinstance(record.get("text_audit"), dict):
        raise ValueError(f"Candidate text audit is missing on line {line_number}")
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Candidate text is empty on line {line_number}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if digest != record.get("candidate_text_sha256"):
        raise ValueError(f"Candidate text hash mismatch on line {line_number}")
    if _record_digest(record) != record.get("candidate_record_sha256"):
        raise ValueError(f"Candidate record hash mismatch on line {line_number}")


def read_candidate_corpus(corpus_path: Path) -> list[dict[str, Any]]:
    try:
        lines = corpus_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read candidate corpus: {corpus_path}") from exc
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid candidate JSON on line {line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Candidate corpus line {line_number} must be an object")
        _validate_candidate_record(record, line_number)
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or document_id in seen_ids:
            raise ValueError(f"Missing or duplicate candidate document ID on line {line_number}")
        seen_ids.add(document_id)
        records.append(record)
    return records


def build_candidate_corpus(
    inventory_path: Path = DEFAULT_INVENTORY,
    corpus_path: Path = DEFAULT_CANDIDATE_CORPUS,
) -> list[dict[str, Any]]:
    inventory = _read_json_object(inventory_path, "candidate inventory")
    sources = inventory.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Candidate inventory must contain a sources list")
    candidate_sources = [
        source
        for source in sources
        if isinstance(source, dict)
        and isinstance(source.get("status"), str)
        and source["status"].startswith("candidate_")
        and source.get("text_file")
    ]
    records = [
        _build_record(inventory_path, inventory, source) for source in candidate_sources
    ]
    if len({record["document_id"] for record in records}) != len(records):
        raise ValueError("Candidate inventory contains duplicate document IDs")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = corpus_path.with_suffix(corpus_path.suffix + ".tmp")
    temporary_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary_path.replace(corpus_path)
    return records


def candidate_corpus_status(corpus_path: Path) -> dict[str, Any]:
    if not corpus_path.exists():
        return {
            "available": False,
            "candidate_documents": 0,
            "document_ids": [],
            "candidate_only": True,
            "answer_eligible": False,
            "training_eligible": False,
        }
    records = read_candidate_corpus(corpus_path)
    return {
        "available": True,
        "candidate_documents": len(records),
        "document_ids": [record["document_id"] for record in records],
        "candidate_only": True,
        "answer_eligible": False,
        "training_eligible": False,
        "text_audits": {
            record["document_id"]: record["text_audit"] for record in records
        },
    }


def lookup_candidate_sources(
    query: str, corpus_path: Path, *, limit: int = 3
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if is_foreign_jurisdiction_query(query):
        return {
            "answer": "لا أجيب إلا عن القانون الجزائري. لا أقدّم معلومات عن قوانين دول أخرى.",
            "legal_answer_provided": False,
            "citations": [],
        }
    candidates = read_candidate_corpus(corpus_path)
    searchable = [
        {
            **record,
            "official_source_checked": True,
            "legal_status_checked_on": None,
            "legal_status_evidence_urls": [],
            "content_sha256": record["candidate_text_sha256"],
        }
        for record in candidates
    ]
    evidence = retrieve(query, searchable, limit=limit)
    citations = [
        {
            "document_id": item.document_id,
            "title": item.title,
            "url": item.source_url,
            "source_language": item.source_language,
            "source_locator": item.source_locator,
            "legal_status": "unknown",
            "candidate_text_sha256": item.content_sha256,
            "excerpt": item.excerpt,
            "segment": item.segment,
        }
        for item in evidence
    ]
    if not evidence:
        message = "لم أعثر على مقتطف مطابق في ملفات المرشحين المحلية."
    else:
        message = (
            "مقتطفات من تفريغات مرشحة غير مراجعة؛ ليست إجابة قانونية ولا دليلًا "
            "على النفاذ أو اكتمال النص. تحقق من الأصل الرسمي قبل الاعتماد.\n\n"
            + "\n\n".join(
                f"«{item.excerpt}»\n"
                f"مرشح: {item.title} — {item.source_locator} ({item.source_url})"
                for item in evidence
            )
        )
    return {
        "answer": message,
        "legal_answer_provided": False,
        "citations": citations,
        "disclaimer": (
            "المقتطفات من تفريغات آلية/غير مراجعة، وليست استشارة قانونية أو دليلًا "
            "على النفاذ. راجع النص الرسمي الساري."
        ),
    }
