"""Lightweight verification: refuse or flag answers not supported by evidence.

This is not a formal proof of correctness. It is a guardrail that rejects
obvious unsupported claims and uncited statements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .hybrid import ArticleEvidence
from .retrieve import tokenize

_QUOTE = re.compile(r"[«\"]([^»\"]{8,})[»\"]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟؛])\s+|\n+")


@dataclass(frozen=True)
class VerificationReport:
    supported: bool
    reason: str
    cited_ids: set[str]
    support_ratio: float


def _normalize(text: str) -> str:
    return " ".join(tokenize(text))


def _best_support(sentence: str, evidence: list[ArticleEvidence]) -> float:
    sentence_terms = set(tokenize(sentence))
    if len(sentence_terms) < 3:
        return 1.0
    best = 0.0
    for item in evidence:
        excerpt_terms = set(tokenize(item.excerpt))
        overlap = len(sentence_terms & excerpt_terms) / max(len(sentence_terms), 1)
        best = max(best, overlap)
        if sentence.strip() and sentence.strip() in item.excerpt:
            best = 1.0
    return best


def verify_answer(
    answer: str,
    evidence: list[ArticleEvidence],
    *,
    minimum_sentence_support: float = 0.45,
    require_quote: bool = True,
) -> VerificationReport:
    """Return a support report for a generated answer against retrieved evidence."""
    if not answer or not answer.strip():
        return VerificationReport(False, "empty-answer", set(), 0.0)
    if not evidence:
        return VerificationReport(False, "no-evidence", set(), 0.0)

    text = answer.strip()
    if "لا أستطيع" in text or "لا يوجد" in text and "مصدر" in text:
        return VerificationReport(True, "explicit-abstention", set(), 1.0)

    evidence_ids = {item.document_id for item in evidence}
    cited_ids = {
        match.group(1)
        for match in re.finditer(r"\[([A-Za-z0-9][A-Za-z0-9._-]{0,99})\]", text)
        if match.group(1) in evidence_ids
    }
    if not cited_ids:
        return VerificationReport(False, "no-citation", set(), 0.0)

    if require_quote and not _QUOTE.search(text):
        return VerificationReport(False, "no-quote", cited_ids, 0.0)

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    ratios = [_best_support(sentence, evidence) for sentence in sentences] or [0.0]
    supported_sentences = sum(1 for ratio in ratios if ratio >= minimum_sentence_support)
    support_ratio = supported_sentences / len(ratios)
    if support_ratio < 0.6:
        return VerificationReport(False, "insufficient-support", cited_ids, support_ratio)
    return VerificationReport(True, "supported", cited_ids, support_ratio)


def build_extractive_answer(evidence: list[ArticleEvidence]) -> str:
    """Deterministic, fully cited fallback answer when generation is unavailable."""
    lines = []
    for item in evidence:
        label = f" — {item.article_label}" if item.article_label else ""
        lines.append(
            f"«{item.excerpt}»\n"
            f"[{item.document_id}] {item.title}{label} ({item.source_locator}) — {item.source_url}"
        )
    return "\n\n".join(lines)
