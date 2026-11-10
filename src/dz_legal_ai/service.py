"""User-facing response policy."""

from __future__ import annotations

from typing import Any

from .policy import asks_current_legal_status, is_foreign_jurisdiction_query
from .retrieve import retrieve

DISCLAIMER = (
    "معلومات عامة فقط وليست استشارة قانونية. تحقّق من النص الرسمي الساري "
    "واستشر محاميًا جزائريًا مؤهلًا قبل اتخاذ أي إجراء."
)
ABSTENTION = (
    "لا أستطيع الإجابة اعتمادًا على corpus الحالي: لم أجد مقطعًا موثّقًا "
    "ومطابقًا بما يكفي. لا أستنتج حكمًا قانونيًا من خارج المصادر المستوردة."
)
OUTSIDE_JURISDICTION = (
    "لا أجيب إلا عن القانون الجزائري. لا أقدّم معلومات عن قوانين دول أخرى."
)
UNVERIFIED_STATUS_ABSTENTION = (
    "لا أستطيع تأكيد الوضع القانوني الحالي: لم أجد مراجعة رسمية موثقة لتعديلات النص "
    "أو إلغائه أو تاريخ سريانه."
)
UNVERIFIED_EVIDENCE_ABSTENTION = (
    "لا أجيب استنادًا إلى نص لا توجد له مراجعة رسمية موثقة للنفاذ الحالي. "
    "المصادر المتاحة محفوظة كمواد تاريخية/مرشحة خارج corpus القانون النافذ."
)


def answer_query(
    query: str,
    documents: list[dict[str, Any]],
    *,
    generator: Any | None = None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if is_foreign_jurisdiction_query(query):
        return {
            "answer": OUTSIDE_JURISDICTION,
            "abstained": True,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }
    evidence = retrieve(query, documents)
    if not evidence:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }
    if asks_current_legal_status(query):
        evidence = [item for item in evidence if item.legal_status == "verified_current"]
        if not evidence:
            return {
                "answer": UNVERIFIED_STATUS_ABSTENTION,
                "abstained": True,
                "citations": [],
                "disclaimer": DISCLAIMER,
            }
    else:
        evidence = [item for item in evidence if item.legal_status == "verified_current"]
        if not evidence:
            return {
                "answer": UNVERIFIED_EVIDENCE_ABSTENTION,
                "abstained": True,
                "citations": [],
                "disclaimer": DISCLAIMER,
            }

    citations = [
        {
            "document_id": item.document_id,
            "legal_domain": item.legal_domain,
            "title": item.title,
            "issuer": item.issuer,
            "url": item.source_url,
            "source_language": item.source_language,
            "source_locator": item.source_locator,
            "source_checked_on": item.source_checked_on,
            "legal_status": item.legal_status,
            "legal_status_checked_on": item.legal_status_checked_on,
            "legal_status_evidence_urls": item.legal_status_evidence_urls,
            "legal_status_verification_note": item.legal_status_verification_note,
            "source_document_sha256": item.source_document_sha256,
            "content_sha256": item.content_sha256,
            "segment": item.segment,
            "excerpt": item.excerpt,
        }
        for item in evidence
    ]
    answer_text = (
        generator.generate_grounded_answer(query, evidence)
        if generator is not None
        else None
    )
    answer_text = answer_text or "\n\n".join(
        f"«{item.excerpt}»\n"
        f"المصدر: {item.title} — {item.source_locator}، "
        f"المقطع المستورد رقم {item.segment} "
        f"({item.source_url})"
        for item in evidence
    )
    return {
        "answer": answer_text,
        "abstained": False,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }


def lookup_source_text(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Return exact source excerpts without asserting they state current law."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if is_foreign_jurisdiction_query(query):
        return {
            "answer": OUTSIDE_JURISDICTION,
            "abstained": True,
            "legal_answer_provided": False,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    evidence = retrieve(query, documents)
    if not evidence:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "legal_answer_provided": False,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    citations = [
        {
            "document_id": item.document_id,
            "title": item.title,
            "issuer": item.issuer,
            "url": item.source_url,
            "source_language": item.source_language,
            "source_locator": item.source_locator,
            "source_checked_on": item.source_checked_on,
            "legal_status": item.legal_status,
            "legal_status_verification_note": item.legal_status_verification_note,
            "source_document_sha256": item.source_document_sha256,
            "content_sha256": item.content_sha256,
            "segment": item.segment,
            "excerpt": item.excerpt,
        }
        for item in evidence
    ]
    answer_text = (
        "هذه مقتطفات حرفية من المواد المستوردة وليست جوابًا قانونيًا. "
        "حالة النفاذ والتعديلات غير متحققة؛ لا تعتمد عليها كقانون ساري.\n\n"
        + "\n\n".join(
            f"«{item.excerpt}»\n"
            f"المصدر: {item.title} — {item.source_locator} "
            f"({item.source_url})"
            for item in evidence
        )
    )
    return {
        "answer": answer_text,
        "abstained": True,
        "legal_answer_provided": False,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }
