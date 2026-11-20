"""Regression tests for concept-based relevance, not raw word overlap."""

from dz_legal_ai.hybrid import expand_query_tokens, retrieve_articles
from dz_legal_ai.ingest import build_record


def metadata(document_id: str, language: str = "ar") -> dict:
    return {
        "document_id": document_id,
        "title": "Relevance fixture (not law)",
        "issuer": "Test fixture",
        "jurisdiction": "DZ",
        "source_url": "https://www.joradp.dz/test-fixture",
        "source_language": language,
        "source_locator": "Synthetic fixture",
        "official_source_checked": True,
        "source_checked_on": "2026-11-09",
        "verification_note": "Synthetic fixture only; not an actual legal source.",
        "reuse_basis": "CC0-1.0",
        "reuse_evidence_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "rights_checked_on": "2026-11-09",
        "rights_verification_note": "Synthetic fixture only.",
        "legal_status": "verified_current",
        "legal_status_checked_on": "2026-11-09",
        "legal_status_evidence_urls": ["https://www.joradp.dz/test-status"],
        "legal_status_verification_note": "Synthetic fixture only.",
        "legal_domain": "civil",
        "legal_instrument_id": f"DZ-{document_id}",
    }


def record(document_id: str, text: str, language: str = "ar") -> dict:
    return build_record(text, metadata(document_id, language))


def test_generic_punishment_word_does_not_override_theft_concept():
    pregnancy = record(
        "generic-punishment",
        "المادة 14: إذا كانت المحكوم عليها حاملا تأجل تنفيذ العقوبة إلى حين وضعها.",
    )
    theft = record(
        "theft-punishment",
        "المادة 350: كل من سرق شيئا غير مملوك له يعد سارقا ويعاقب بالحبس من سنة إلى خمس سنوات.",
    )
    evidence = retrieve_articles("ماهي عقوبة السارق", [pregnancy, theft])
    assert evidence
    assert evidence[0].document_id == "theft-punishment"
    assert all(item.document_id != "generic-punishment" for item in evidence)


def test_inheritance_verb_finds_french_succession_articles():
    family = record(
        "family-succession",
        "Art. 176. – La part du frère est égale à celle de la sœur dans la succession; "
        "le partage des héritiers est fait par tête.",
        "fr",
    )
    evidence = retrieve_articles(
        "أربع أخوة رجال وخمس نساء ورثنا عن أبنا مليار دينار كم حصة كل واحد", [family]
    )
    assert evidence
    assert evidence[0].article_label == "Article 176"


def test_currency_amount_does_not_pull_unrelated_tax_article():
    tax = record(
        "tax-amount",
        "المادة 151: كل مؤسسة لديها رقم أعمال يفوق مليار دينار تخضع للتصريح السنوي.",
    )
    inheritance = record(
        "inheritance-share",
        "المادة 176: للذكر مثل حظ الأنثيين في الإرث، ويكون نصيب الأخوة من التركة معلوما.",
    )
    evidence = retrieve_articles(
        "ورثنا مليار دينار كم حصة الأخوة في الإرث", [tax, inheritance]
    )
    assert evidence
    assert all(item.document_id != "tax-amount" for item in evidence)
    assert evidence[0].document_id == "inheritance-share"


def test_arabic_definite_article_and_hamza_variants_expand_together():
    tokens = expand_query_tokens("السرقة والإرث")
    assert "سارق" in tokens
    assert "ميراث" in tokens
    assert "succession" in tokens


def test_death_and_kinship_without_the_word_inheritance_finds_family_law():
    """A natural question never says "ميراث" but still is an inheritance one."""
    family = record(
        "family-succession",
        "المادة 176: للذكر مثل حظ الأنثيين في الإرث، ويكون نصيب الأخوة من التركة معلوما.",
    )
    evidence = retrieve_articles(
        "توفي والدنا وترك مالا ونحن أبناء وبنات كم نصيب كل واحد", [family]
    )
    assert evidence
    assert evidence[0].document_id == "family-succession"


def test_persian_yeh_and_tatweel_do_not_hide_a_legal_term():
    from dz_legal_ai.retrieve import tokenize

    assert tokenize("قانون التسجیل") == ["قانون", "تسجيل"]
    assert "بـ" not in tokenize("متعلق بـ قانون")


def test_question_about_the_harasser_reaches_the_harassment_article():
    """The user asks about "المتحرش"; the code says "التحرش الجنسي".

    This is a retrieval regression only: it must surface the cited article,
    never turn the excerpt into a verdict or a legal conclusion.
    """
    unrelated = record(
        "generic-punishment",
        "المادة 20: كل من أخل النظام بمخالفة توجب عقوبة يعاقب بغرامة زجرية.",
    )
    harassment = record(
        "penal-code-ar-portal",
        "المادة 341 مكرر: يعد مرتكبا لجريمة التحرش الجنسي ويعاقب بالحبس من شهرين "
        "إلى سنة وبغرامة من 50.000 دج إلى 100.000 دج.",
    )
    evidence = retrieve_articles("ماهي عقوبة المتحرش", [unrelated, harassment])
    assert evidence
    assert evidence[0].document_id == "penal-code-ar-portal"
    assert evidence[0].article_label == "المادة 341 مكرر"
    assert "التحرش الجنسي" in evidence[0].excerpt
    assert all(item.document_id != "generic-punishment" for item in evidence)
