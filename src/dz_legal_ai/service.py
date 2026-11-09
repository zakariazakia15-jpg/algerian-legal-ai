"""User-facing response policy."""

from __future__ import annotations

from typing import Any

from .retrieve import retrieve

DISCLAIMER = (
    "معلومات عامة فقط وليست استشارة قانونية. تحقّق من النص الرسمي الساري "
    "واستشر محاميًا جزائريًا مؤهلًا قبل اتخاذ أي إجراء."
)
ABSTENTION = (
    "لا أستطيع الإجابة اعتمادًا على corpus الحالي: لم أجد مقطعًا موثّقًا "
    "ومطابقًا بما يكفي. لا أستنتج حكمًا قانونيًا من خارج المصادر المستوردة."
)


def answer_query(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query must not be empty")
    evidence = retrieve(query, documents)
    if not evidence:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    citations = [
        {
            "document_id": item.document_id,
            "title": item.title,
            "issuer": item.issuer,
            "url": item.source_url,
            "source_checked_on": item.source_checked_on,
            "content_sha256": item.content_sha256,
            "paragraph": item.paragraph,
            "excerpt": item.excerpt,
        }
        for item in evidence
    ]
    answer_text = "\n\n".join(
        f"«{item.excerpt}»\n"
        f"المصدر: {item.title}، الفقرة المستوردة رقم {item.paragraph} "
        f"({item.source_url})"
        for item in evidence
    )
    return {
        "answer": answer_text,
        "abstained": False,
        "citations": citations,
        "disclaimer": DISCLAIMER,
    }
