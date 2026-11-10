import json
from pathlib import Path

import pytest

from dz_legal_ai.candidate_corpus import build_candidate_corpus, candidate_corpus_status
from dz_legal_ai.cli import main
from dz_legal_ai.candidate_corpus import lookup_candidate_sources
from dz_legal_ai.ingest import build_record, build_seed_corpus, read_corpus
from dz_legal_ai.local_model import LocalQwenGenerator
from dz_legal_ai.retrieve import retrieve
from dz_legal_ai.service import answer_query, lookup_source_text

ROOT = Path(__file__).parent.parent
INVENTORY = ROOT / "data" / "legal_source_inventory.json"


def test_candidate_builder_preserves_provenance_and_never_activates_candidates(tmp_path):
    output = tmp_path / "candidates.jsonl"
    records = build_candidate_corpus(INVENTORY, output)

    assert len(records) == 3
    assert {record["source_language"] for record in records} == {"ar", "fr"}
    for record in records:
        assert record["jurisdiction"] == "DZ"
        assert record["legal_status"] == "unknown"
        assert record["review_state"] == "unreviewed"
        assert record["candidate_only"] is True
        assert record["answer_eligible"] is False
        assert record["training_eligible"] is False
        assert record["text_audit"]["utf8_valid"] is True
        assert record["text_audit"]["replacement_characters"] == 0
        assert record["text_audit"]["article_reference_count"] > 0
        assert record["candidate_text_sha256"]
        assert record["candidate_record_sha256"]

    cpca = next(
        record
        for record in records
        if record["document_id"] == "algerian-law-08-09-2008-cpca-fr-candidate"
    )
    procurement = next(
        record
        for record in records
        if record["document_id"] == "algerian-decree-15-247-2015-procurement-fr-candidate"
    )
    arabic_constitution = next(
        record
        for record in records
        if record["document_id"] == "algerian-constitution-2020-ar-ocr-candidate"
    )
    assert cpca["text_audit"]["title_token_coverage"] == 1
    assert cpca["text_audit"]["maximum_article_reference"] == 1065
    assert procurement["text_audit"]["title_token_coverage"] == 1
    assert procurement["text_audit"]["maximum_article_reference"] == 220
    assert arabic_constitution["text_audit"]["unique_article_reference_count"] < (
        arabic_constitution["text_audit"]["article_reference_count"]
    )
    status = candidate_corpus_status(output)
    assert status["candidate_documents"] == 3
    assert status["answer_eligible"] is False
    assert status["training_eligible"] is False

    with pytest.raises(ValueError):
        read_corpus(output)


def test_candidate_cli_build_and_check(tmp_path, capsys):
    output = tmp_path / "candidates.jsonl"
    assert main(
        [
            "candidate-corpus-build",
            "--inventory",
            str(INVENTORY),
            "--output",
            str(output),
        ]
    ) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["candidate_documents"] == 3
    assert built["answer_eligible"] is False
    assert built["training_eligible"] is False

    assert main(["candidate-corpus-check", "--candidate-corpus", str(output)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["available"] is True
    assert checked["candidate_documents"] == 3


def test_candidate_builder_rejects_modified_candidate_text(tmp_path):
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    source = next(
        item for item in inventory["sources"] if item.get("candidate_text_sha256")
    )
    root = tmp_path
    text_path = root / source["text_file"]
    metadata_path = root / source["metadata_file"]
    text_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        (ROOT / source["metadata_file"]).read_text(encoding="utf-8"), encoding="utf-8"
    )
    text_path.write_text(
        (ROOT / source["text_file"]).read_text(encoding="utf-8") + "tampered",
        encoding="utf-8",
    )
    inventory_path = root / "data" / "legal_source_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

    with pytest.raises(ValueError, match="Candidate text hash mismatch"):
        build_candidate_corpus(inventory_path, root / "candidate.jsonl")


def test_candidate_api_status_never_feeds_the_ask_endpoint(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dz_legal_ai.api import app

    candidate_path = tmp_path / "candidates.jsonl"
    build_candidate_corpus(INVENTORY, candidate_path)
    monkeypatch.setenv("DZ_LEGAL_AI_CANDIDATE_CORPUS", str(candidate_path))
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(tmp_path / "active-corpus.jsonl"))

    with TestClient(app) as client:
        status_response = client.get("/candidate-corpus/status")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["candidate_documents"] == 3
        assert status["answer_eligible"] is False
        assert status["training_eligible"] is False
        assert all("text" not in audit for audit in status["text_audits"].values())

        ask_response = client.post("/ask", json={"query": "marchés publics"})
        assert ask_response.status_code == 200
        answer = ask_response.json()
        assert answer["abstained"] is True
        assert answer["citations"] == []

        generate_request = client.post(
            "/ask", json={"query": "marchés publics", "generate": True}
        )
        assert generate_request.status_code == 200
        assert generate_request.json()["abstained"] is True
        assert generate_request.json()["citations"] == []

        candidate_lookup_response = client.post(
            "/candidate-lookup", json={"query": "Art. 220 publication décret"}
        )
        assert candidate_lookup_response.status_code == 200
        candidate_answer = candidate_lookup_response.json()
        assert candidate_answer["legal_answer_provided"] is False
        assert candidate_answer["citations"]
        assert all(
            citation["legal_status"] == "unknown"
            for citation in candidate_answer["citations"]
        )

        lookup_response = client.post(
            "/lookup", json={"query": "marchés publics"}
        )
        assert lookup_response.status_code == 200
        assert lookup_response.json()["legal_answer_provided"] is False
        assert lookup_response.json()["citations"] == []

        candidate_path.write_text('{"candidate_only": true}\n', encoding="utf-8")
        invalid_response = client.get("/candidate-corpus/status")
        assert invalid_response.status_code == 500


def test_seed_corpus_and_source_lookup_are_runnable_without_claiming_current_law(
    tmp_path, capsys, monkeypatch
):
    corpus = tmp_path / "corpus.jsonl"
    records, added = build_seed_corpus(corpus)
    assert len(records) == 2
    assert added == 2
    assert all(record["legal_status"] == "unknown" for record in records)

    records, added = build_seed_corpus(corpus)
    assert len(records) == 2
    assert added == 0

    query = "La Constitution est au-dessus de tous"
    lookup = lookup_source_text(query, records)
    assert lookup["abstained"] is True
    assert lookup["legal_answer_provided"] is False
    assert lookup["citations"]
    assert "La Constitution est au-dessus de tous" in lookup["citations"][0]["excerpt"]
    assert lookup["citations"][0]["legal_status"] == "unknown"

    legal_answer = answer_query(query, records)
    assert legal_answer["abstained"] is True
    assert legal_answer["citations"] == []

    assert main(["lookup", query, "--corpus", str(corpus)]) == 0
    cli_lookup = json.loads(capsys.readouterr().out)
    assert cli_lookup["legal_answer_provided"] is False
    assert cli_lookup["citations"][0]["document_id"] == (
        "algerian-constitution-2020-fr"
    )

    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from dz_legal_ai.api import app

    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(corpus))
    with TestClient(app) as client:
        response = client.post("/lookup", json={"query": query})
        assert response.status_code == 200
        assert response.json()["legal_answer_provided"] is False
        assert response.json()["citations"][0]["legal_status"] == "unknown"

        ask_response = client.post("/ask", json={"query": query})
        assert ask_response.status_code == 200
        assert ask_response.json()["abstained"] is True
        assert ask_response.json()["citations"] == []

    assert main(["seed-corpus", "--corpus", str(corpus)]) == 0
    seeded = json.loads(capsys.readouterr().out)
    assert seeded["added"] == 0


def test_candidate_lookup_returns_only_labeled_unreviewed_excerpts(tmp_path):
    candidate_path = tmp_path / "candidates.jsonl"
    build_candidate_corpus(INVENTORY, candidate_path)
    result = lookup_candidate_sources("Art. 220 publication décret", candidate_path)
    assert result["legal_answer_provided"] is False
    assert result["citations"]
    assert all(citation["legal_status"] == "unknown" for citation in result["citations"])
    assert "غير مراجعة" in result["answer"]

    foreign = lookup_candidate_sources("Que dit le droit en France ?", candidate_path)
    assert foreign["citations"] == []
    assert foreign["legal_answer_provided"] is False


class _FakeGenerator:
    def __init__(self):
        self.called = False

    def generate_grounded_answer(self, query, evidence):
        self.called = True
        assert query == "match synthetic text"
        assert evidence
        return "Generated from reviewed Algerian-law source."


def _reviewed_test_record():
    metadata = {
        "document_id": "verified-synthetic-test",
        "title": "Synthetic test fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": "fr",
        "source_locator": "Synthetic fixture",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic test only",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic test only",
        "legal_status": "verified_current",
        "legal_status_checked_on": "2026-11-09",
        "legal_status_evidence_urls": ["https://www.joradp.dz/test-status"],
        "legal_status_verification_note": "Synthetic test only",
        "legal_domain": "civil",
        "legal_instrument_id": "DZ-SYNTHETIC-TEST",
    }
    return build_record("match synthetic text", metadata)


def test_local_generation_is_only_called_with_retrieved_verified_current_evidence():
    record = _reviewed_test_record()
    generator = _FakeGenerator()
    result = answer_query(
        "match synthetic text", [record], generator=generator
    )
    assert generator.called is True
    assert result["abstained"] is False
    assert result["answer"].startswith("Generated from")
    assert result["citations"][0]["document_id"] == record["document_id"]

    unknown_record = {**record, "legal_status": "unknown"}
    unknown_generator = _FakeGenerator()
    unknown_result = answer_query(
        "match synthetic text", [unknown_record], generator=unknown_generator
    )
    assert unknown_generator.called is False
    assert unknown_result["abstained"] is True


def test_local_model_reports_missing_weights_without_attempting_a_download(tmp_path):
    generator = LocalQwenGenerator(tmp_path)
    with pytest.raises(FileNotFoundError, match="no model download is attempted"):
        generator.smoke_test()
