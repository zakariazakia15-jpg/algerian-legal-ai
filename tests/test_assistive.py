"""The opt-in advisory toolbox must never become a legal-ruling engine."""

import json
import math

import pytest

from dz_legal_ai.assistive import (
    DRAFT_WARNING,
    INHERITANCE_WARNING,
    REVIEW_WARNING,
    STATUS_WARNING,
    analyze_issue,
    compute_inheritance,
    draft_contract,
    explain_term,
    list_tools,
    risk_flags,
    review_contract,
)
from dz_legal_ai.ingest import build_record


def _record(document_id: str, text: str, *, title: str = "عقد القانون المدني") -> dict:
    metadata = {
        "document_id": document_id,
        "title": title,
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": "ar",
        "source_locator": "Synthetic fixture",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only.",
        "legal_status": "unknown",
        "legal_status_checked_on": None,
        "legal_status_evidence_urls": [],
        "legal_status_verification_note": "Synthetic fixture only.",
        "legal_domain": "civil",
        "legal_instrument_id": f"DZ-{document_id}",
    }
    return build_record(text, metadata)


CORPUS = [
    _record("fixture-civil", "المادة 1: للذكر مثل حظ الأنثيين في الميراث والتركة."),
    _record("fixture-contracts", "المادة 2: العقد شريعة المتعاقدين ويجب تحديد الثمن والمكان والتوقيع."),
]


def test_inheritance_arithmetic_is_transparent_and_flagged():
    result = compute_inheritance(sons=4, daughters=2, estate=10_000_000_000, documents=CORPUS)
    assert result["mode"] == "opt-in-calculation"
    assert result["arithmetic_only"] is True
    assert result["informational_only"] is True
    assert result["shares"]["each_son"] == 2_000_000_000
    assert result["shares"]["each_daughter"] == 1_000_000_000
    assert result["shares"]["total_sons"] + result["shares"]["total_daughters"] == 10_000_000_000
    assert INHERITANCE_WARNING in result["warnings"]
    assert STATUS_WARNING in result["warnings"]
    assert result["citations"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sons": -1, "daughters": 2, "estate": 10},
        {"sons": 0, "daughters": 0, "estate": 10},
        {"sons": True, "daughters": 1, "estate": 10},
        {"sons": 1, "daughters": 1, "estate": -10},
        {"sons": 1, "daughters": 1, "estate": math.nan},
    ],
)
def test_inheritance_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        compute_inheritance(**kwargs, documents=CORPUS)


def test_draft_is_a_template_with_missing_fields_and_warnings():
    result = draft_contract(contract_type="sale", documents=CORPUS)
    assert result["draft_only"] is True
    assert "مسودة" in result["draft"]
    assert "اسم البائع" in result["missing_fields"]
    assert DRAFT_WARNING in result["warnings"]
    assert STATUS_WARNING in result["warnings"]
    assert result["citations"]

    filled = draft_contract(
        contract_type="sale",
        documents=CORPUS,
        values={"اسم البائع": "أحمد"},
    )
    assert "أحمد" in filled["draft"]
    assert "اسم البائع" not in filled["missing_fields"]


def test_draft_rejects_unknown_type_and_bad_values():
    with pytest.raises(ValueError):
        draft_contract(contract_type="unknown", documents=CORPUS)
    with pytest.raises(ValueError):
        draft_contract(contract_type="sale", documents=CORPUS, values={"x": 1})


def test_contract_review_is_a_checklist_not_a_legal_opinion():
    result = review_contract(
        text="العقد بين طرفين، الثمن معلوم، والمكان محدد، والتوقيع موجود.",
        documents=CORPUS,
    )
    assert result["mode"] == "opt-in-review"
    assert REVIEW_WARNING in result["warnings"]
    assert "تحديد الأطراف" in result["mentioned"]
    assert result["questions_for_lawyer"]
    assert result["citations"]


def test_issue_analysis_returns_topics_and_questions_only():
    result = analyze_issue(question="توفي والدي وترك تركة", documents=CORPUS)
    assert result["mode"] == "opt-in-issue-analysis"
    assert result["topics_to_verify"]
    assert result["questions_for_lawyer"]
    assert result["citations"]
    assert "حكم" not in " ".join(result["questions_for_lawyer"])


def test_term_occurrences_are_literal_and_cited():
    result = explain_term(term="الذكر", documents=CORPUS)
    assert result["mode"] == "opt-in-term-occurrences"
    assert result["occurrences"]
    assert "للذكر مثل حظ الأنثيين" in result["occurrences"][0]["excerpt"]
    assert result["citations"]


def test_risk_flags_are_keyword_based_and_do_not_conclude():
    result = risk_flags(
        text="الورثة يطالبون بعقار وبالديون قبل أجل التوقيع",
        documents=CORPUS,
    )
    keys = {flag["key"] for flag in result["flags"]}
    assert {"estate", "real_estate", "debt", "deadline", "signature"} <= keys
    assert result["flag_count"] == len(result["flags"])
    assert result["warnings"]

    empty = risk_flags(text="لا كلمات مفتاحية هنا", documents=CORPUS)
    assert empty["flags"] == []
    assert empty["citations"] == []


def test_tool_directory_declares_default_behaviour_unchanged():
    tools = list_tools()
    assert tools["default_behaviour_unchanged"] is True
    assert "compute-inheritance" in tools["tools"]
    assert "draft-contract" in tools["tools"]
    assert len(tools["contract_types"]) == 10
    assert "citations" not in tools


def test_text_inputs_are_validated():
    with pytest.raises(ValueError):
        review_contract(text="   ", documents=CORPUS)
    with pytest.raises(ValueError):
        explain_term(term="", documents=CORPUS)
    with pytest.raises(ValueError):
        analyze_issue(question=123, documents=CORPUS)


# ---- HTTP gate: the tools must not be reachable without explicit opt-in ----

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from dz_legal_ai import config as config_module  # noqa: E402
from dz_legal_ai.api import app  # noqa: E402


def _write_corpus(tmp_path):
    path = tmp_path / "toolbox-corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in CORPUS) + "\n",
        encoding="utf-8",
    )
    return path


def test_http_tools_require_explicit_opt_in_header(tmp_path, monkeypatch):
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(corpus))
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: {"retrieval": {"dense_enabled": False}},
    )
    with TestClient(app) as client:
        denied = client.post(
            "/compute/inheritance",
            json={"sons": 4, "daughters": 2, "estate": 10_000_000_000},
        )
        assert denied.status_code == 403

        allowed = client.post(
            "/compute/inheritance",
            json={"sons": 4, "daughters": 2, "estate": 10_000_000_000},
            headers={"X-DZ-LEGAL-AI-OPT-IN": "advisory-tools"},
        )
        assert allowed.status_code == 200
        body = allowed.json()
        assert body["shares"]["each_son"] == 2_000_000_000
        assert body["informational_only"] is True

        no_heir = client.post(
            "/compute/inheritance",
            json={"sons": 0, "daughters": 0, "estate": 10},
            headers={"X-DZ-LEGAL-AI-OPT-IN": "advisory-tools"},
        )
        assert no_heir.status_code == 422

        draft = client.post(
            "/drafts/contract",
            json={"contract_type": "will", "values": {}},
            headers={"X-DZ-LEGAL-AI-OPT-IN": "advisory-tools"},
        )
        assert draft.status_code == 200
        assert draft.json()["draft_only"] is True

        tools = client.get("/tools", headers={"X-DZ-LEGAL-AI-OPT-IN": "advisory-tools"})
        assert tools.status_code == 200
        assert tools.json()["default_behaviour_unchanged"] is True


def test_cli_set_fields_and_tool_directory(tmp_path, capsys):
    from dz_legal_ai.cli import main

    corpus = _write_corpus(tmp_path)
    code = main(
        [
            "draft-contract",
            "--type",
            "sale",
            "--set",
            "اسم البائع=أحمد",
            "--corpus",
            str(corpus),
        ]
    )
    assert code == 0
    body = json.loads(capsys.readouterr().out)
    assert "أحمد" in body["draft"]
    assert body["draft_only"] is True

    assert main(["toolbox", "--corpus", str(corpus)]) == 0
    assert json.loads(capsys.readouterr().out)["default_behaviour_unchanged"] is True


def test_http_all_remaining_advisory_endpoints(tmp_path, monkeypatch):
    corpus = _write_corpus(tmp_path)
    monkeypatch.setenv("DZ_LEGAL_AI_CORPUS", str(corpus))
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda *args, **kwargs: {"retrieval": {"dense_enabled": False}},
    )
    headers = {"X-DZ-LEGAL-AI-OPT-IN": "advisory-tools"}
    requests = [
        ("/reviews/contract", {"text": "الثمن معلوم والتوقيع موجود"}),
        ("/analysis/issue", {"query": "توفي والدي وترك تركة"}),
        ("/terms/explain", {"term": "الذكر"}),
        ("/risks/flags", {"text": "ورثة وعقار وديون"}),
    ]
    with TestClient(app) as client:
        for endpoint, body in requests:
            response = client.post(endpoint, json=body, headers=headers)
            assert response.status_code == 200, endpoint
            payload = response.json()
            assert payload["informational_only"] is True
            assert payload["warnings"]
            assert "disclaimer" in payload
