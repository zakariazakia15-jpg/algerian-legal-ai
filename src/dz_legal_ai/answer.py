"""Grounded legal assistant: retrieve, generate with guardrails, verify, cite.

This is the path that turns the project from a lookup tool into an assistant.
It never invents sources, and it falls back to extractive cited passages when
the local model is unavailable or the answer cannot be verified.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .config import GROUNDED_SYSTEM_PROMPT_SUFFIX, generator_settings, system_prompt, verification_settings
from .generate import GenerationUnavailable, generate_text
from .hybrid import ArticleEvidence, retrieve_articles
from .policy import asks_current_legal_status, is_foreign_jurisdiction_query
from .service import ABSTENTION, DISCLAIMER, OUTSIDE_JURISDICTION, UNVERIFIED_EVIDENCE_ABSTENTION, UNVERIFIED_STATUS_ABSTENTION
from .verify import build_extractive_answer, verify_answer


def _filter_by_status(evidence: list[ArticleEvidence], require_current: bool) -> list[ArticleEvidence]:
    if require_current:
        return [item for item in evidence if item.legal_status == "verified_current"]
    return evidence


def _citation_payload(evidence: list[ArticleEvidence]) -> list[dict[str, Any]]:
    result = []
    for item in evidence:
        payload = asdict(item)
        payload["excerpt"] = item.excerpt
        payload["url"] = item.source_url
        result.append(payload)
    return result


def _build_prompt(query: str, evidence: list[ArticleEvidence]) -> str:
    blocks = []
    for index, item in enumerate(evidence, start=1):
        article = item.article_label or "مقتطف"
        blocks.append(
            f"[{index}] document_id={item.document_id}\n"
            f"العنوان: {item.title}\n"
            f"المادة/الموضع: {article} — {item.source_locator}\n"
            f"حالة النفاذ المصرّح بها: {item.legal_status}\n"
            f"النص:\n{item.excerpt}"
        )
    context = "\n\n".join(blocks)
    return (
        "السؤال:\n"
        f"{query}\n\n"
        "المقتطفات الرسمية المتاحة (استخدمها فقط ولا تضف معرفة خارجها):\n"
        f"{context}\n\n"
        "المطلوب: أجب بالعربية الفصحى الواضحة، واذكر أساس الإجابة من المقتطفات أعلاه "
        "مع [document_id] ورقم المادة والاقتباس الحرفي «...». "
        "إن كانت المقتطفات غير كافية أو لا تتعلق بالسؤال فقل: "
        "\"لا أستطيع الإجابة اعتمادًا على المقتطفات المتاحة\" مع سبب موجز."
    )


def assist_query(
    query: str,
    documents: list[dict[str, Any]],
    *,
    config: dict[str, dict[str, Any]] | None = None,
    prefer_generation: bool = True,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Return a grounded advisory answer or an explicit abstention."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if is_foreign_jurisdiction_query(query):
        return {
            "answer": OUTSIDE_JURISDICTION,
            "abstained": True,
            "mode": "policy",
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    evidence = retrieve_articles(query, documents, config=config)
    if not evidence:
        return {
            "answer": ABSTENTION,
            "abstained": True,
            "mode": "no-evidence",
            "citations": [],
            "disclaimer": DISCLAIMER,
        }

    require_current = asks_current_legal_status(query)
    eligible = _filter_by_status(evidence, require_current=True)
    informational_only = False
    if require_current and not eligible:
        return {
            "answer": UNVERIFIED_STATUS_ABSTENTION,
            "abstained": True,
            "mode": "status-unverified",
            "citations": [],
            "disclaimer": DISCLAIMER,
        }
    if not eligible:
        if not allow_unverified:
            # Strict default: do not present unverified texts as legal answers.
            return {
                "answer": UNVERIFIED_EVIDENCE_ABSTENTION,
                "abstained": True,
                "mode": "status-unverified",
                "citations": [],
                "disclaimer": DISCLAIMER,
            }
        # Draft corpus mode: useful while building the corpus, never presented as current law.
        evidence = [item for item in evidence if item.legal_status == "unknown"]
        informational_only = True
    else:
        evidence = eligible

    citations = _citation_payload(evidence)
    disclaimer = DISCLAIMER
    if informational_only:
        disclaimer = (
            "هذه مقتطفات من نصوص لم تُكتمل مراجعتها الرسمية؛ معلومات استرشادية فقط، "
            "ولا يجوز وصفها بالقانون النافذ. " + DISCLAIMER
        )
    generator = generator_settings(config)
    verification = verification_settings(config)
    full_system = system_prompt(config) + GROUNDED_SYSTEM_PROMPT_SUFFIX

    if prefer_generation and generator["enabled"]:
        try:
            generated = generate_text(
                _build_prompt(query, evidence), system=full_system, config=config
            )
            report = verify_answer(
                generated.text,
                evidence,
                minimum_sentence_support=verification["minimum_sentence_support"],
                require_quote=verification["require_quote"],
            )
            if report.supported:
                return {
                    "answer": generated.text,
                    "abstained": False,
                    "mode": "generated-grounded",
                    "verification": {
                        "supported": report.supported,
                        "reason": report.reason,
                        "support_ratio": round(report.support_ratio, 3),
                        "cited_document_ids": sorted(report.cited_ids),
                    },
                    "citations": citations,
                    "informational_only": informational_only,
                    "disclaimer": disclaimer,
                }
            # verification failed -> fall through to extractive
        except GenerationUnavailable:
            pass

    extractive = build_extractive_answer(evidence)
    return {
        "answer": extractive,
        "abstained": False,
        "mode": "extractive-grounded",
        "verification": {"supported": True, "reason": "extractive-fallback", "support_ratio": 1.0},
        "citations": citations,
        "informational_only": informational_only,
        "disclaimer": disclaimer,
    }
