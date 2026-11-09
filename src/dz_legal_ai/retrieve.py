"""Small deterministic lexical retriever; it never generates legal conclusions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .policy import validate_source_url

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670]")
_ARABIC_NORMALIZATION = str.maketrans(
    {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0649": "\u064A",
    }
)


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _DIACRITICS.sub("", normalized).translate(_ARABIC_NORMALIZATION)
    return _TOKEN.findall(normalized)


def split_segments(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?:\r?\n)+", text) if segment.strip()]


@dataclass(frozen=True)
class Evidence:
    document_id: str
    legal_domain: str
    title: str
    issuer: str
    source_url: str
    source_language: str
    source_locator: str
    source_checked_on: str
    legal_status: str
    legal_status_checked_on: str | None
    legal_status_evidence_urls: list[str]
    legal_status_verification_note: str
    content_sha256: str
    excerpt: str
    segment: int
    score: float


def retrieve(
    query: str,
    documents: list[dict[str, Any]],
    *,
    minimum_coverage: float = 0.6,
    limit: int = 3,
) -> list[Evidence]:
    query_terms = set(tokenize(query))
    if not query_terms:
        return []
    candidates: list[Evidence] = []
    for document in documents:
        if document.get("jurisdiction") != "DZ":
            continue
        validate_source_url(document.get("source_url", ""))
        if document.get("official_source_checked") is not True:
            continue
        if not document.get("content_sha256"):
            continue
        segments = split_segments(document.get("text", ""))
        for index, segment in enumerate(segments, start=1):
            paragraph_terms = set(tokenize(segment))
            coverage = len(query_terms & paragraph_terms) / len(query_terms)
            if coverage < minimum_coverage:
                continue
            candidates.append(
                Evidence(
                    document_id=document["document_id"],
                    legal_domain=document["legal_domain"],
                    title=document["title"],
                    issuer=document["issuer"],
                    source_url=document["source_url"],
                    source_language=document["source_language"],
                    source_locator=document["source_locator"],
                    source_checked_on=document["source_checked_on"],
                    legal_status=document["legal_status"],
                    legal_status_checked_on=document["legal_status_checked_on"],
                    legal_status_evidence_urls=document["legal_status_evidence_urls"],
                    legal_status_verification_note=document[
                        "legal_status_verification_note"
                    ],
                    content_sha256=document["content_sha256"],
                    excerpt=segment,
                    segment=index,
                    score=coverage,
                )
            )
    candidates.sort(key=lambda item: (-item.score, item.document_id, item.segment))
    return candidates[:limit]
