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


def answer_query(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
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
        current_evidence = [
            item for item in evidence if item.legal_status == "verified_current"
        ]
        if current_evidence:
            evidence = current_evidence

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
            "content_sha256": item.content_sha256,
            "segment": item.segment,
            "excerpt": item.excerpt,
        }
        for item in evidence
    ]
    answer_text = "\n\n".join(
        (
            "تنبيه: حالة النفاذ والتعديلات غير متحققة. النص مقتطف من المصدر التاريخي فقط.\n"
            if item.legal_status == "unknown"
            else "تنبيه: المصدر مصنف تاريخيًا/ملغى، ولا يمثل القانون الحالي.\n"
            if item.legal_status == "historical_or_repealed"
            else ""
        )
        + f"«{item.excerpt}»\n"
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
