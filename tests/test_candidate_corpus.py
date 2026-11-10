import json
from pathlib import Path

import pytest

from dz_legal_ai.candidate_corpus import build_candidate_corpus, candidate_corpus_status
from dz_legal_ai.cli import main
from dz_legal_ai.ingest import build_seed_corpus, read_corpus
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
