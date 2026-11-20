"""Grounded advisory synthesis: answer like a consultant, without inventing law.

The old fallback only printed retrieved excerpts, which behaved like a search
engine. This module keeps every substantive claim as a verbatim quote from a
retrieved article, but wraps the quotes in Arabic advisory scaffolding: what
each text says, where it conflicts with another text, and what must be checked
by a qualified human. It is deliberately not a fatwa and not a ruling.
"""

from __future__ import annotations

import re

from .hybrid import ArticleEvidence
from .verify import build_extractive_answer

_HEADER = re.compile(r"^\s*(المادة|مادة|Art\.|Article|الفصل)\b", re.IGNORECASE)
_AMENDMENT = re.compile(r"^\s*[\(（]")
_SENTENCE = re.compile(r"(?<=[.!?؟؛])\s+")
_MAX_QUOTE = 520

LIMITS = (
    "حدود هذه الإجابة: هذه إجابة استرشادية مبنية على نصوص متاحة لم تُكتمل مراجعتها "
    "الرسمية؛ ليست فتوى ولا حكماً قضائياً، ولا يجوز وصف أي نص بأنه نافذ. "
    "راجع النص الرسمي الساري ومحامياً جزائرياً مختصاً قبل أي إجراء."
)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _operative_quote(item: ArticleEvidence) -> str:
    """Pick the operative sentence, dropping the article and amendment headers."""
    lines = [line.strip() for line in (item.excerpt or "").splitlines() if line.strip()]
    kept = []
    for line in lines:
        if _HEADER.match(line) or _AMENDMENT.match(line):
            continue
        kept.append(line)
    body = _clean(" ".join(kept)) or _clean(item.excerpt)
    if not body:
        return ""
    sentences = [part for part in _SENTENCE.split(body) if len(part.strip()) >= 20]
    quote = sentences[0].strip() if sentences else body
    if len(quote) > _MAX_QUOTE:
        quote = quote[:_MAX_QUOTE].rsplit(" ", 1)[0] + " …"
    return quote


def _tier_label(item: ArticleEvidence) -> str:
    if item.source_tier == "secondary":
        return "مصدر ثانوي غير رسمي"
    return "مصدر صنّفه المتن رسمياً"


def _groups(evidence: list[ArticleEvidence]) -> list[tuple[str, list[ArticleEvidence]]]:
    grouped: dict[str, list[ArticleEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.article_label or "مقتطف دون رقم مادة", []).append(item)
    return list(grouped.items())


def build_advisory_answer(query: str, evidence: list[ArticleEvidence]) -> str:
    """Compose a cited, advisory-style answer from article-level evidence."""
    del query  # The scaffolding is language-neutral; claims come only from evidence.
    if not evidence:
        raise ValueError("advisory answer requires evidence")

    lines: list[str] = ["الجواب الاسترشادي:", ""]
    for label, items in _groups(evidence):
        lines.append(f"الموضع: {label}")
        for item in items:
            quote = _operative_quote(item)
            if not quote:
                continue
            lines.append(
                f"- بحسب النص المستشهد به في «{item.title}» ({_tier_label(item)}): "
                f"«{quote}» [{item.document_id} | {label}] — {item.source_url}"
            )
        if len(items) > 1 and len({item.content_sha256 for item in items}) > 1:
            lines.append(
                "توجد روايتان مختلفتان للمادة نفسها في المصادر المتاحة؛ "
                "لا يمكن الجزم بأيّهما المعتمد دون مطابقة الجريدة الرسمية."
            )
        lines.append("")

    lines.append("النصوص الحرفية الكاملة للرجوع:")
    lines.append(build_extractive_answer(evidence))
    lines.append("")
    lines.append(LIMITS)
    return "\n".join(lines)
