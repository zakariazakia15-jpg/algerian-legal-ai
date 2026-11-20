"""The simple local web interface must stay grounded and informational only."""

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from dz_legal_ai import config as config_module
from dz_legal_ai.api import app
from dz_legal_ai.ingest import build_record


def _secondary_record() -> dict:
    metadata = {
        "document_id": "web-ui-fixture",
        "title": "Web UI fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://oraconstantine.com/laws/test.php",
        "source_language": "fr",
        "source_locator": "Synthetic fixture, page 1",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only; not actual rights evidence.",
        "legal_status": "unknown",
        "legal_status_checked_on": None,
        "legal_status_evidence_urls": [],
        "legal_status_verification_note": "Synthetic fixture only; not a legal review.",
        "legal_domain": "civil",
        "legal_instrument_id": "DZ-SYNTHETIC-WEB-UI",
        "source_tier": "secondary",
    }
    text = (
        "Article 1er. Première disposition.\n\n"
        "Art. 2. Une seconde disposition pour les tests."
    )
    return build_record(text, metadata)


def test_ui_is_plain_black_and_white_and_reachable_from_root():
    with TestClient(app) as client:
        response = client.get("/ui")
        assert response.status_code == 200
        body = response.text
        assert "المساعد القانوني الجزائري" in body
        assert "اسأل" in body
        assert "/assist" in body
        assert "color: #000" in body
        assert "background: #fff" in body

        root = client.get("/", follow_redirects=True)
        assert root.status_code == 200
        assert "المساعد القانوني الجزائري" in root.text


def test_assist_endpoint_returns_cited_informational_extractive_answer(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(_secondary_record(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(corpus))
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: {"retrieval": {"dense_enabled": False}},
    )

    with TestClient(app) as client:
        response = client.post("/assist", json={"query": "seconde disposition"})
        assert response.status_code == 200
        result = response.json()
        assert result["abstained"] is False
        assert result["mode"] == "extractive-grounded"
        assert result["informational_only"] is True
        assert "seconde disposition" in result["answer"]
        assert result["citations"][0]["document_id"] == "web-ui-fixture"
        assert result["citations"][0]["article_label"] == "Article 2"
        assert "مصدر ثانوي غير رسمي" in result["disclaimer"]
        assert "ليست استشارة قانونية" in result["disclaimer"]

        blank = client.post("/assist", json={"query": "   "})
        assert blank.status_code == 422


def test_assist_endpoint_abstains_when_corpus_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(tmp_path / "missing.jsonl"))
    with TestClient(app) as client:
        response = client.post("/assist", json={"query": "أي سؤال"})
    assert response.status_code == 200
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []
