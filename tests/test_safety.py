import hashlib
import json
from pathlib import Path

import pytest

from dz_legal_ai.evaluation import evaluate_cases
from dz_legal_ai.ingest import build_record, ingest_files, read_corpus
from dz_legal_ai.policy import validate_source_url
from dz_legal_ai.service import ABSTENTION, answer_query


def metadata(**updates):
    values = {
        "document_id": "synthetic-test-only",
        "title": "Synthetic test fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": "fr",
        "source_locator": "Synthetic test fixture, page 1",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only; not actual rights evidence.",
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
    text = "unrelated opening material\n\nsynthetic token alpha beta\n\nunrelated final material"
    record = build_record(text, metadata())
    response = answer_query("synthetic token alpha", [record])
    assert response["abstained"] is False
    assert response["citations"][0]["excerpt"] == "synthetic token alpha beta"
    assert "unrelated opening material" not in response["answer"]
    assert "unrelated final material" not in response["answer"]
    assert response["citations"][0]["url"] == record["source_url"]
    assert response["citations"][0]["source_language"] == record["source_language"]
    assert response["citations"][0]["source_locator"] == record["source_locator"]
    assert response["citations"][0]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert response["disclaimer"]
    assert "المقطع المستورد رقم 2" in response["answer"]


def test_retrieval_cites_individual_lines():
    record = build_record(
        "first irrelevant line\nsecond matching phrase\nthird irrelevant line",
        metadata(),
    )
    response = answer_query("second matching phrase", [record])
    assert len(response["citations"]) == 1
    assert response["citations"][0]["excerpt"] == "second matching phrase"
    assert response["citations"][0]["segment"] == 2


def test_arabic_search_normalizes_harakat():
    record = build_record("القَانُون الجزائري", metadata())
    response = answer_query("القانون الجزائري", [record])
    assert response["abstained"] is False
    assert response["citations"][0]["excerpt"] == "القَانُون الجزائري"


def test_rejects_non_algerian_and_unapproved_sources():
    with pytest.raises(ValueError, match="Only Algerian"):
        build_record("test", metadata(jurisdiction="FR"))
    with pytest.raises(ValueError, match="official-source host"):
        validate_source_url("https://example.com/law")
    with pytest.raises(ValueError, match="official-source host"):
        validate_source_url(None)


@pytest.mark.parametrize(
    "updates",
    [
        {"source_language": "en"},
        {"source_locator": "  "},
    ],
)
def test_rejects_invalid_source_language_or_locator(updates):
    with pytest.raises(ValueError):
        build_record("test", metadata(**updates))


def test_rejects_unlicensed_content():
    with pytest.raises(ValueError, match="Reuse basis is unknown or unsupported"):
        build_record("test", metadata(reuse_basis="unknown"))


def test_requires_canonical_review_dates():
    with pytest.raises(ValueError, match="YYYY-MM-DD format"):
        build_record("test", metadata(source_checked_on="20261109"))


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
    with pytest.raises(ValueError, match="canonical rights evidence URL"):
        build_record("test", metadata(reuse_basis="CC-BY-4.0"))


def test_accepts_statutory_basis_for_classified_official_legal_text():
    record = build_record(
        "synthetic legal text",
        metadata(
            reuse_basis="DZ-ORD-03-05-ART-11",
            reuse_evidence_url=(
                "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
            ),
            legal_text_type="law",
            rights_verification_note=(
                "Human reviewed; extracted legal text only, not the Gazette issue."
            ),
        ),
    )
    assert record["reuse_basis"] == "DZ-ORD-03-05-ART-11"
    assert record["legal_text_type"] == "law"


def test_official_seed_is_french_and_bounded_to_ordinance_03_05():
    seed = Path(__file__).parent.parent / "data" / "seed"
    text = (seed / "ordinance-03-05-fr.txt").read_text(encoding="utf-8")
    source_metadata = json.loads(
        (seed / "ordinance-03-05-fr.metadata.json").read_text(encoding="utf-8")
    )
    record = build_record(text, source_metadata)
    assert record["source_language"] == "fr"
    assert "Art. 11." in record["text"]
    assert "Art. 164." in record["text"]
    assert "Ordonnance n° 03-06" not in record["text"]
    assert "\ufffd" not in record["text"]


def test_official_seed_end_to_end_cli_ingest_and_cited_answer(tmp_path, capsys):
    from dz_legal_ai.cli import main

    seed = Path(__file__).parent.parent / "data" / "seed"
    corpus = tmp_path / "corpus.jsonl"

    assert main(
        [
            "ingest",
            "--text",
            str(seed / "ordinance-03-05-fr.txt"),
            "--metadata",
            str(seed / "ordinance-03-05-fr.metadata.json"),
            "--corpus",
            str(corpus),
        ]
    ) == 0
    ingested = json.loads(capsys.readouterr().out)
    assert ingested["ingested"] is True
    assert ingested["document_id"] == "algerian-ord-03-05-fr"

    assert main(["ask", "les décisions de justice", "--corpus", str(corpus)]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["abstained"] is False
    assert answer["citations"][0]["source_language"] == "fr"
    assert answer["citations"][0]["url"] == (
        "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
    )
    assert answer["citations"][0]["excerpt"].startswith("Art. 11.")
    assert "ليست استشارة قانونية" in answer["disclaimer"]

    assert main(["ask", "unmatched-phrase-xyz", "--corpus", str(corpus)]) == 0
    abstention = json.loads(capsys.readouterr().out)
    assert abstention["abstained"] is True
    assert abstention["citations"] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"legal_text_type": "portal_page"},
        {
            "legal_text_type": "law",
            "reuse_evidence_url": "https://www.joradp.dz/HAR/Accueil.htm",
        },
    ],
)
def test_rejects_invalid_scope_or_evidence_for_statutory_basis(updates):
    values = {
        "reuse_basis": "DZ-ORD-03-05-ART-11",
        "reuse_evidence_url": (
            "https://www.joradp.dz/FTP/jo-francais/2003/F2003044.pdf"
        ),
        "legal_text_type": "law",
    }
    values.update(updates)
    with pytest.raises(ValueError):
        build_record("synthetic legal text", metadata(**values))


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


def test_evaluation_rejects_empty_set():
    with pytest.raises(ValueError, match="at least one case"):
        evaluate_cases([], [])


def test_ingest_round_trip_and_duplicate_id_rejection(tmp_path):
    source = tmp_path / "source.txt"
    metadata_file = tmp_path / "metadata.json"
    corpus = tmp_path / "data" / "corpus.jsonl"
    source.write_text("synthetic source", encoding="utf-8")
    metadata_file.write_text(json.dumps(metadata()), encoding="utf-8")

    record = ingest_files(source, metadata_file, corpus)
    assert read_corpus(corpus) == [record]
    with pytest.raises(ValueError, match="already exists"):
        ingest_files(source, metadata_file, corpus)


def test_optional_api_health_ask_and_validation(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import HTTPException

    from dz_legal_ai.api import AskRequest, ask, health

    assert health() == {"status": "ok"}
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(tmp_path / "empty-corpus.jsonl"))
    response = ask(AskRequest(query="أي سؤال"))
    assert response["abstained"] is True
    with pytest.raises(HTTPException) as error:
        ask(AskRequest(query="   "))
    assert error.value.status_code == 422
