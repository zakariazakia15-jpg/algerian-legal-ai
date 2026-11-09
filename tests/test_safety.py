import json
import hashlib

import pytest

from dz_legal_ai.ingest import build_record, read_corpus
from dz_legal_ai.evaluation import evaluate_cases
from dz_legal_ai.policy import validate_source_url
from dz_legal_ai.service import ABSTENTION, answer_query


def metadata(**updates):
    values = {
        "document_id": "synthetic-test-only",
        "title": "Synthetic test fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "license": "CC0-1.0",
        "license_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "license_checked_on": "2026-11-09",
        "license_verification_note": "Synthetic fixture only; not actual rights evidence.",
    }
    values.update(updates)
    return values


def test_refuses_when_no_corpus_evidence():
    response = answer_query("سؤال لا دليل عليه", [])
    assert response["abstained"] is True
    assert response["answer"] == ABSTENTION
    assert response["citations"] == []
    assert response["disclaimer"]


def test_answer_quotes_only_cited_source_text():
    text = "synthetic token alpha beta"
    record = build_record(text, metadata())
    response = answer_query("synthetic token alpha", [record])
    assert response["abstained"] is False
    assert response["citations"][0]["excerpt"] == text
    assert response["citations"][0]["url"] == record["source_url"]
    assert response["citations"][0]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert response["disclaimer"]


def test_rejects_non_algerian_and_unapproved_sources():
    with pytest.raises(ValueError, match="Only Algerian"):
        build_record("test", metadata(jurisdiction="FR"))
    with pytest.raises(ValueError, match="official-source host"):
        validate_source_url("https://example.com/law")


def test_rejects_unlicensed_content():
    with pytest.raises(ValueError, match="unknown or unsupported"):
        build_record("test", metadata(license="unknown"))


def test_detects_tampered_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    record = build_record("synthetic token alpha", metadata())
    record["content_sha256"] = "0" * 64
    corpus.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Content hash mismatch"):
        read_corpus(corpus)


def test_detects_tampered_source_metadata(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    record = build_record("synthetic token alpha", metadata())
    record["source_url"] = "https://www.joradp.dz/altered-source"
    corpus.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Record metadata hash mismatch"):
        read_corpus(corpus)


def test_requires_canonical_cc_license_evidence():
    with pytest.raises(ValueError, match="canonical Creative Commons"):
        build_record("test", metadata(license="CC-BY-4.0"))


def test_evaluation_checks_expected_abstention_and_citation():
    record = build_record("synthetic token alpha beta", metadata())
    results = evaluate_cases(
        [
            {
                "query": "synthetic token alpha",
                "expected_abstained": False,
                "expected_document_ids": ["synthetic-test-only"],
            },
            {
                "query": "no matching words",
                "expected_abstained": True,
                "expected_document_ids": [],
            },
        ],
        [record],
    )
    assert results == {"evaluated": 2, "passed": 2, "failed": 0}
