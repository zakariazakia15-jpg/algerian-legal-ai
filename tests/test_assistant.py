import json

from dz_legal_ai.answer import assist_query
from dz_legal_ai.chunking import split_into_chunks
from dz_legal_ai.cli import main
from dz_legal_ai.hybrid import ArticleEvidence, expand_query_tokens, retrieve_articles
from dz_legal_ai.ingest import build_record
from dz_legal_ai.verify import verify_answer


def metadata(**updates):
    values = {
        "document_id": "synthetic-assistant",
        "title": "Synthetic assistant fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": "fr",
        "source_locator": "Synthetic fixture, page 1",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only; not actual rights evidence.",
        "legal_status": "verified_current",
        "legal_status_checked_on": "2026-11-09",
        "legal_status_evidence_urls": ["https://www.joradp.dz/test-status"],
        "legal_status_verification_note": "Synthetic fixture only; not a legal review.",
        "legal_domain": "civil",
        "legal_instrument_id": "DZ-SYNTHETIC-ASSISTANT",
    }
    values.update(updates)
    return values


def current_record():
    text = (
        "Article 1er. Disposition synthétique sur le contrat.\n\n"
        "Art. 2. Une seconde disposition pour les tests.\n\n"
        "Art. 3. Une dernière disposition sans rapport."
    )
    return build_record(text, metadata())


def test_french_and_arabic_article_chunking():
    french = split_into_chunks(
        "Article 1er. Première disposition.\n\nArt. 2. Deuxième disposition.", "fr"
    )
    assert [chunk.article_label for chunk in french] == ["Article 1", "Article 2"]

    arabic = split_into_chunks(
        "مقدمة غير مرتبطة.\nالمادة ١١ الحكم الأول.\nالمادة ١٢ الحكم الثاني.", "ar"
    )
    assert arabic[0].article_label is None
    assert [chunk.article_label for chunk in arabic[1:]] == ["المادة 11", "المادة 12"]


def test_synonyms_expand_arabic_and_french_queries():
    assert "contrat" in expand_query_tokens("عقد")
    assert "عقد" in expand_query_tokens("contrat")
    assert "constitution" in expand_query_tokens("دستور")


def test_article_retrieval_keeps_legal_locator():
    evidence = retrieve_articles("seconde disposition", [current_record()])
    assert evidence
    assert evidence[0].article_label == "Article 2"
    assert "seconde disposition" in evidence[0].excerpt


def test_extractive_assistant_cites_article_from_verified_corpus():
    result = assist_query(
        "seconde disposition", [current_record()], prefer_generation=False
    )
    assert result["abstained"] is False
    assert result["mode"] == "extractive-grounded"
    assert result["citations"][0]["article_label"] == "Article 2"
    assert "seconde disposition" in result["answer"]
    assert result["verification"]["supported"] is True


def test_assistant_abstains_for_unverified_texts_unless_informational_mode():
    record = build_record("Article 1er. disposition synthétique", metadata(legal_status="unknown", legal_status_checked_on=None, legal_status_evidence_urls=[]))
    blocked = assist_query("disposition synthétique", [record], prefer_generation=False)
    assert blocked["abstained"] is True
    assert blocked["citations"] == []

    informational = assist_query(
        "disposition synthétique",
        [record],
        prefer_generation=False,
        allow_unverified=True,
    )
    assert informational["abstained"] is False
    assert informational["informational_only"] is True
    assert "لم تُكتمل مراجعتها" in informational["disclaimer"]


def test_grounded_generation_is_accepted_as_advisory_answer(monkeypatch):
    """The assistant path turns a model output into a cited answer, not a lookup."""
    from dz_legal_ai import answer
    from dz_legal_ai.generate import GenerationResult

    grounded = "«Une seconde disposition pour les tests.» [synthetic-assistant]"
    monkeypatch.setattr(
        answer,
        "generate_text",
        lambda *args, **kwargs: GenerationResult(grounded, "fake-local-model", True),
    )
    result = assist_query("seconde disposition", [current_record()], prefer_generation=True)
    assert result["abstained"] is False
    assert result["mode"] == "generated-grounded"
    assert result["verification"]["supported"] is True
    assert result["verification"]["cited_document_ids"] == ["synthetic-assistant"]
    assert result["citations"][0]["article_label"] == "Article 2"
    assert "لا استشارة قانونية" in result["disclaimer"] or "استشارة قانونية" in result["disclaimer"]


def test_unsupported_generation_falls_back_to_extractive_answer(monkeypatch):
    from dz_legal_ai import answer
    from dz_legal_ai.generate import GenerationResult

    unsupported = "حكم مختلف تمامًا عن عقد الإيجار [synthetic-assistant]"
    monkeypatch.setattr(
        answer,
        "generate_text",
        lambda *args, **kwargs: GenerationResult(unsupported, "fake-local-model", True),
    )
    result = assist_query("seconde disposition", [current_record()], prefer_generation=True)
    assert result["mode"] == "extractive-grounded"
    assert result["verification"]["reason"] == "extractive-fallback"
    assert "seconde disposition" in result["answer"]


def test_extractive_fallback_reads_as_an_advisory_answer():
    result = assist_query("seconde disposition", [current_record()], prefer_generation=False)
    assert result["answer"].startswith("الجواب الاسترشادي:")
    assert result["answer_style"] == "advisory-grounded"
    assert "حدود هذه الإجابة" in result["answer"]
    assert "ليست فتوى" in result["answer"]


def test_advisory_answer_flags_two_conflicting_versions_of_one_article():
    from dz_legal_ai.advisory import build_advisory_answer

    first = build_record(
        "المادة 10: يعد مرتكب جريمة التحرش الجنسي ويعاقب بالحبس من شهرين إلى سنة "
        "وبغرامة من 50.000 دج إلى 100.000 دج.",
        metadata(document_id="first-source", legal_domain="criminal", source_language="ar"),
    )
    second = build_record(
        "المادة 10: يعد مرتكب جريمة التحرش الجنسي ويعاقب بالحبس من سنة إلى ثلاث سنوات "
        "وبغرامة من 100.000 دج إلى 300.000 دج.",
        metadata(document_id="second-source", legal_domain="criminal", source_language="ar"),
    )
    evidence = retrieve_articles("عقوبة التحرش", [first, second])
    assert len({item.article_label for item in evidence}) == 1
    answer = build_advisory_answer("ماهي العقوبة", evidence)
    assert "روايتان مختلفتان" in answer
    assert "first-source" in answer and "second-source" in answer
    assert "50.000 دج" in answer and "300.000 دج" in answer
    assert "(مصدر صنّفه المتن رسمياً)" in answer


def test_verifier_rejects_unsupported_answer():
    evidence = retrieve_articles("seconde disposition", [current_record()])
    bad = verify_answer("حكم مختلف تمامًا عن عقد الإيجار [synthetic-assistant]", evidence)
    assert bad.supported is False
    good = verify_answer("«Une seconde disposition pour les tests.» [synthetic-assistant]", evidence)
    assert good.supported is True


def test_cli_assist_runs_without_transformers(tmp_path, capsys):
    corpus = tmp_path / "corpus.jsonl"
    record = current_record()
    corpus.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(
        ["assist", "seconde disposition", "--corpus", str(corpus), "--extractive-only"]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["abstained"] is False
    assert result["citations"][0]["article_label"] == "Article 2"
