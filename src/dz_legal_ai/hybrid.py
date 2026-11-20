"""Hybrid, article-level retrieval for the grounded assistant.

The original lexical retriever stays untouched for the existing CLI/API.
This module is used only by the assistant path and combines:
  * Arabic/French legal synonym expansion,
  * lexical coverage on article-level chunks,
  * optional local dense embeddings (never downloaded implicitly).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from .bm25 import BM25Index
from .chunking import chunk_record
from .config import corpus_quality_settings, retrieval_settings
from .policy import validate_source_url
from .retrieve import tokenize
from .vector_store import VectorStore, passage_prefix, query_prefix

logger = logging.getLogger(__name__)

# Bidirectional legal vocabulary: Arabic <-> French. Each key adds its values to the query.
LEGAL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "قانون": ("loi", "droit"),
    "loi": ("قانون",),
    "دستور": ("constitution",),
    "constitution": ("دستور",),
    "مادة": ("article",),
    "article": ("مادة",),
    "عقد": ("contrat",),
    "contrat": ("عقد",),
    "بيع": ("vente",),
    "vente": ("بيع",),
    "إيجار": ("location", "louage"),
    "location": ("إيجار",),
    "محكمة": ("tribunal", "juridiction"),
    "tribunal": ("محكمة",),
    "قاض": ("juge",),
    "juge": ("قاض",),
    "حكم": ("jugement", "arret"),
    "jugement": ("حكم",),
    "استئناف": ("appel",),
    "appel": ("استئناف",),
    "نقض": ("cassation",),
    "cassation": ("نقض",),
    "جريمة": ("infraction", "delit", "crime"),
    "infraction": ("جريمة",),
    "عقوبة": ("peine", "sanction"),
    "peine": ("عقوبة",),
    "سجن": ("emprisonnement", "prison"),
    "غرامة": ("amende",),
    "amende": ("غرامة",),
    "زواج": ("mariage",),
    "mariage": ("زواج",),
    "طلاق": ("divorce",),
    "divorce": ("طلاق",),
    "ميراث": ("succession", "heritage"),
    "succession": ("ميراث",),
    "عمل": ("travail", "emploi"),
    "travail": ("عمل",),
    "عامل": ("salarie", "travailleur"),
    "salaries": ("عامل",),
    "ضريبة": ("impot", "fiscal"),
    "impot": ("ضريبة",),
    "مدني": ("civil",),
    "civil": ("مدني",),
    "جزائي": ("penal", "criminel"),
    "penal": ("جزائي",),
    "إجراءات": ("procedure",),
    "procedure": ("إجراءات",),
    "شهادة": ("temoignage", "témoignage"),
    "دفاع": ("defense",),
    "موكل": ("client",),
    "وكيل": ("avocat", "mandataire"),
    "مسؤولية": ("responsabilite", "responsabilité"),
    "تعاقد": ("contrat",),
    "ملكية": ("propriete", "propriété"),
    "أرض": ("terrain", "immeuble"),
    "عقار": ("immeuble",),
}

_DENSE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class ArticleEvidence:
    """One retrieved article-level passage ready for citation."""

    document_id: str
    title: str
    issuer: str
    legal_domain: str
    legal_instrument_id: str
    source_url: str
    source_language: str
    source_locator: str
    legal_status: str
    legal_status_checked_on: str | None
    legal_status_evidence_urls: list[str]
    legal_status_verification_note: str
    source_document_sha256: str | None
    content_sha256: str
    article_label: str | None
    chunk_index: int
    excerpt: str
    score: float
    book: str | None = None
    chapter: str | None = None
    section: str | None = None


def expand_query_tokens(query: str) -> set[str]:
    tokens = set(tokenize(query))
    expanded = set(tokens)
    for token in tokens:
        for synonym in LEGAL_SYNONYMS.get(token, ()):
            expanded.update(tokenize(synonym))
    return expanded


def _load_dense(model_name: str) -> Any | None:
    if model_name in _DENSE_CACHE:
        return _DENSE_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        try:
            encoder = SentenceTransformer(model_name, local_files_only=True)
        except TypeError:
            encoder = SentenceTransformer(model_name)
        _DENSE_CACHE[model_name] = encoder
        return encoder
    except Exception as exc:  # optional dependency / model not downloaded
        logger.info("Dense retrieval disabled: %s", exc)
        _DENSE_CACHE[model_name] = None
        return None


def _cosine(left: list[float], right: list[float]) -> float:
    try:
        import numpy as np

        a, b = np.asarray(left), np.asarray(right)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denominator) if denominator else 0.0
    except ImportError:
        numerator = sum(x * y for x, y in zip(left, right))
        norm_left = sum(x * x for x in left) ** 0.5
        norm_right = sum(y * y for y in right) ** 0.5
        return numerator / (norm_left * norm_right) if norm_left and norm_right else 0.0


def retrieve_articles(
    query: str,
    documents: list[dict[str, Any]],
    *,
    config: dict[str, dict[str, Any]] | None = None,
    top_k: int | None = None,
) -> list[ArticleEvidence]:
    """Retrieve the most relevant article chunks with provenance intact."""
    settings = retrieval_settings(config)
    quality = corpus_quality_settings(config)
    limit = top_k if top_k is not None else settings["top_k"]
    excluded = set(quality["excluded_document_ids"])
    query_terms = expand_query_tokens(query)
    if not query_terms or not documents:
        return []

    encoder = None
    query_vector: list[float] | None = None
    dense_by_chunk: dict[tuple[str, int], float] = {}
    if settings["dense_enabled"]:
        store = VectorStore(settings["index_dir"])
        if store.load():
            # A prebuilt local index avoids encoding thousands of chunks per query.
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                encoder = None
            else:
                try:
                    encoder = SentenceTransformer(
                        settings["dense_model"], local_files_only=True
                    )
                except Exception:
                    encoder = None
            if encoder is not None:
                query_text = query_prefix(settings["dense_model"]) + query
                query_vector = encoder.encode(
                    query_text, normalize_embeddings=True
                ).tolist()
                dense_by_chunk = store.similarity_by_chunk(query_vector)
        else:
            encoder = _load_dense(settings["dense_model"])
            if encoder is not None:
                query_text = query_prefix(settings["dense_model"]) + query
                query_vector = encoder.encode(
                    query_text, normalize_embeddings=True
                ).tolist()

    # Collect eligible article chunks first so BM25 can use corpus-wide IDF.
    candidates: list[dict[str, Any]] = []
    for document in documents:
        if document.get("jurisdiction") != "DZ":
            continue
        if document.get("document_id") in excluded:
            continue
        validate_source_url(document.get("source_url", ""))
        if document.get("official_source_checked") is not True:
            continue
        if not document.get("content_sha256"):
            continue
        for chunk in chunk_record(document):
            # Table of contents, covers and gazette headers are not legal evidence.
            if not chunk.get("article_label"):
                continue
            candidates.append(chunk)

    ranked = [
        (index, chunk)
        for index, chunk in enumerate(candidates)
        if len(query_terms & set(tokenize(chunk["text"]))) / len(query_terms)
        >= settings["minimum_coverage"]
    ]
    if not ranked:
        return []
    bm25 = BM25Index([tokenize(chunk["text"]) for _, chunk in ranked])
    bm25_scores = [bm25.score(query_terms, local_index) for local_index, _ in enumerate(ranked)]
    max_bm25 = max(bm25_scores) or 1.0

    results: list[ArticleEvidence] = []
    for (local_index, chunk), raw_score in zip(ranked, bm25_scores):
        lexical = raw_score / max_bm25
        dense = 0.0
        if query_vector is not None and encoder is not None:
            if dense_by_chunk:
                dense = dense_by_chunk.get(
                    (chunk["document_id"], chunk["chunk_index"]), 0.0
                )
            else:
                chunk_text = passage_prefix(settings["dense_model"]) + chunk["text"]
                vector = encoder.encode(chunk_text, normalize_embeddings=True).tolist()
                dense = _cosine(query_vector, vector)
        label = chunk.get("article_label") or ""
        label_bonus = 0.08 if any(term in label for term in query_terms) else 0.0
        structural_text = " ".join(
            str(chunk.get(key) or "") for key in ("book", "chapter", "section")
        )
        structural_bonus = (
            settings["structural_weight"]
            if any(term in structural_text for term in query_terms)
            else 0.0
        )
        score = (
            settings["lexical_weight"] * lexical
            + settings["dense_weight"] * dense
            + label_bonus
            + structural_bonus
        )
        results.append(
            ArticleEvidence(
                document_id=chunk["document_id"],
                title=chunk["title"],
                issuer=chunk["issuer"],
                legal_domain=chunk["legal_domain"],
                legal_instrument_id=chunk["legal_instrument_id"],
                source_url=chunk["source_url"],
                source_language=chunk["source_language"],
                source_locator=chunk["source_locator"],
                legal_status=chunk["legal_status"],
                legal_status_checked_on=chunk["legal_status_checked_on"],
                legal_status_evidence_urls=list(chunk["legal_status_evidence_urls"] or []),
                legal_status_verification_note=chunk["legal_status_verification_note"],
                source_document_sha256=chunk["source_document_sha256"],
                content_sha256=chunk["content_sha256"],
                article_label=chunk["article_label"],
                chunk_index=chunk["chunk_index"],
                excerpt=chunk["text"],
                score=score,
                book=chunk.get("book"),
                chapter=chunk.get("chapter"),
                section=chunk.get("section"),
            )
        )
    results.sort(key=lambda item: (-item.score, item.document_id, item.chunk_index))
    diversified: list[ArticleEvidence] = []
    per_document: dict[str, int] = {}
    for item in results:
        count = per_document.get(item.document_id, 0)
        if count >= quality["max_chunks_per_document"]:
            continue
        diversified.append(item)
        per_document[item.document_id] = count + 1
        if len(diversified) >= limit:
            break
    return diversified
