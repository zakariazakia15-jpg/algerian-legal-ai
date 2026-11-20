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
import os
from typing import Any

from .bm25 import BM25Index
from .chunking import chunk_record
from .config import corpus_quality_settings, retrieval_settings
from .policy import validate_secondary_source_url
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
    "عقوبة": ("عقوبات", "peine", "peines", "sanction", "sanctions"),
    "عقوبات": ("عقوبة", "peine", "peines"),
    "peine": ("عقوبة", "عقوبات"),
    "سارق": ("سرقة", "سرق", "vol", "voleur", "theft"),
    "سرقة": ("سارق", "سرق", "vol", "voleur", "theft"),
    "سرق": ("سرقة", "سارق", "vol", "voleur"),
    "vol": ("سرقة", "سارق"),
    "سجن": ("emprisonnement", "prison"),
    "غرامة": ("amende",),
    "amende": ("غرامة",),
    # A question about the *person* ("المتحرش") must reach articles written
    # about the *act* ("التحرش الجنسي"); neither word contains the other.
    "متحرش": ("تحرش", "التحرش الجنسي", "harassment", "harcèlement"),
    "تحرش": ("متحرش", "التحرش الجنسي", "harassment", "harcèlement"),
    "harassment": ("تحرش", "متحرش", "harcèlement"),
    "harcèlement": ("تحرش", "متحرش", "harassment"),
    "زواج": ("mariage",),
    "mariage": ("زواج",),
    "طلاق": ("divorce",),
    "divorce": ("طلاق",),
    "ميراث": ("إرث", "ورثة", "وارث", "ورث", "succession", "heritage", "héritage", "héritier"),
    "إرث": ("ميراث", "ورثة", "وارث", "succession", "heritage"),
    "ورث": ("ميراث", "إرث", "ورثة", "succession"),
    "ورثنا": ("ميراث", "إرث", "ورثة", "وارث", "succession", "héritage"),
    "ورثة": ("ميراث", "إرث", "وارث", "succession", "héritier"),
    "وارث": ("ميراث", "إرث", "ورثة", "héritier"),
    "succession": ("ميراث", "إرث", "ورثة"),
    "حصة": ("أنصبة", "نصيب", "part", "partage"),
    "نصيب": ("حصة", "أنصبة", "part"),
    "وترك": ("ترك", "تركة", "ميراث", "إرث", "succession"),
    "ترك": ("تركة", "ميراث", "إرث", "succession"),
    "وحقوق": ("حقوق",),
    "وبنتان": ("بنتان", "ابنة", "بنات", "fille"),
    "ابناء": ("ابن", "أولاد", "fils"),
    "أخوة": ("أخ", "أخوان", "إخوة", "frère", "freres"),
    "أخ": ("أخوة", "أخوان", "frère"),
    "نساء": ("امرأة", "نساء", "femme", "femmes"),
    "دينار": ("دج",),
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
# Chunking a 6 MB corpus on every question costs ~60 s on CPU. Cache by
# immutable document hash so repeated questions reuse the same article chunks.
_CHUNK_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _chunks_for(document: dict[str, Any]) -> list[dict[str, Any]]:
    key = (str(document.get("document_id", "")), str(document.get("content_sha256", "")))
    cached = _CHUNK_CACHE.get(key)
    if cached is None:
        cached = chunk_record(document)
        _CHUNK_CACHE[key] = cached
    return cached


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
    source_tier: str = "official"


# Synonym keys are normalized at import time: a query token such as "ارث"
# must find the entry written as "إرث" in the vocabulary above.
_NORMALIZED_SYNONYMS: dict[str, tuple[str, ...]] = {}
for _key, _values in LEGAL_SYNONYMS.items():
    for _token in tokenize(_key):
        merged = list(_NORMALIZED_SYNONYMS.get(_token, ()))
        for _value in _values:
            merged.extend(tokenize(_value))
        _NORMALIZED_SYNONYMS[_token] = tuple(dict.fromkeys(merged))

# Words that appear in almost every law but do not identify a legal concept.
# A chunk matching only these must never be offered as an answer.
GENERIC_TERMS = frozenset(
    tokenize(
        "عقوبة عقوبات peine peines sanction sanctions قانون loi droit "
        "حكم jugement arret جزائي جزائري الجزائر حالة حكام "
        "مليار مليون دينار دج اربع خمس وخمس واحد "
        "احكام اهم وثيقة متعلقة تنص"
    )
)


# Natural questions often omit the legal topic: "توفي والدنا وترك مالاً" is an
# inheritance question without ever saying "ميراث". A small explicit hint table
# adds the topic concept in that case; it never adds a legal conclusion.
_TOPIC_HINTS: tuple[tuple[frozenset[str], tuple[str, ...]], ...] = (
    (
        frozenset(tokenize("توفي توفى توفيت وفاة مات موت وترك ترك")),
        tuple(tokenize("ميراث إرث succession ورثة وارث")),
    ),
)


def expand_query_tokens(query: str) -> set[str]:
    tokens = set(tokenize(query))
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_NORMALIZED_SYNONYMS.get(token, ()))
    return expanded


def query_concept_groups(query: str) -> list[frozenset[str]]:
    """Group each significant query word with its legal equivalents.

    Relevance is judged per concept, not per raw word: "ورثنا" and "الميراث"
    are one concept, while "ماهي" and "كم" are not concepts at all.
    """
    groups: list[frozenset[str]] = []
    seen: set[frozenset[str]] = set()
    for token in tokenize(query):
        group = frozenset(expand_query_tokens(token))
        if group and group not in seen:
            seen.add(group)
            groups.append(group)
    tokens = set(tokenize(query))
    for markers, topics in _TOPIC_HINTS:
        if tokens & markers:
            for topic in topics:
                group = frozenset(expand_query_tokens(topic))
                if group and group not in seen:
                    seen.add(group)
                    groups.append(group)
    return groups


def _load_dense(model_name: str) -> Any | None:
    if model_name in _DENSE_CACHE:
        return _DENSE_CACHE[model_name]
    try:
        # A broken/partial TensorFlow install must not break the local encoder.
        os.environ.setdefault("USE_TF", "0")
        from sentence_transformers import SentenceTransformer

        # Never download weights at answer time; build-index is the only path
        # that is allowed to fetch a model, and that is an explicit CLI step.
        encoder = SentenceTransformer(model_name, local_files_only=True)
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
    query_groups = query_concept_groups(query)
    query_terms = set().union(*query_groups) if query_groups else set()
    if not query_groups or not documents:
        return []

    encoder = None
    query_vector: list[float] | None = None
    dense_by_chunk: dict[tuple[str, int], float] = {}
    if settings["dense_enabled"]:
        store = VectorStore(settings["index_dir"])
        if store.load():
            # A prebuilt local index avoids encoding thousands of chunks per query.
            try:
                os.environ.setdefault("USE_TF", "0")
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
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
        # When the prebuilt index is missing, stay lexical. Loading the encoder
        # at answer time only to discard its vector would add minutes of model
        # startup and can trigger network lookups in offline deployments.

    # Collect eligible article chunks first so BM25 can use corpus-wide IDF.
    candidates: list[dict[str, Any]] = []
    for document in documents:
        if document.get("jurisdiction") != "DZ":
            continue
        if document.get("document_id") in excluded:
            continue
        validate_secondary_source_url(document.get("source_url", ""))
        if document.get("official_source_checked") is not True:
            continue
        if not document.get("content_sha256"):
            continue
        for chunk in _chunks_for(document):
            # Table of contents, covers and gazette headers are not legal evidence.
            if not chunk.get("article_label"):
                continue
            candidates.append(chunk)

    # The domain is often written only in the title ("قانون التوجيه العقاري")
    # and not repeated in every article. Include the structural metadata in the
    # searchable text so the title can identify the right instrument.
    def search_text(chunk: dict[str, Any]) -> str:
        # Only the instrument title/domain is added here. Book and chapter
        # headings often contain generic words such as "أحكام" that would
        # otherwise overwhelm the article text; they feed the structural bonus
        # below instead.
        metadata = " ".join(
            str(chunk.get(key) or "") for key in ("title", "legal_domain")
        )
        return f"{chunk['text']} {metadata}"

    # A passage qualifies only if it covers enough *concepts* and at least one
    # of them is specific. This stops generic words such as "عقوبة" from
    # pulling unrelated articles about any kind of sentence.
    specific_groups = [group for group in query_groups if not (group <= GENERIC_TERMS)]
    has_specific_query = bool(specific_groups)
    # Generic words such as "أحكام" appear in almost every code and drown out the
    # rare concept that actually identifies the instrument ("بحري", "تسجيل").
    scoring_terms = set().union(*specific_groups) if specific_groups else query_terms
    ranked = []
    for index, chunk in enumerate(candidates):
        chunk_tokens = set(tokenize(search_text(chunk)))
        matched = [group for group in query_groups if group & chunk_tokens]
        if not matched:
            continue
        specific_matched = [
            group for group in matched if not (group <= GENERIC_TERMS)
        ]
        if has_specific_query and not specific_matched:
            continue
        # Long natural questions contain many filler words. Two independent
        # specific concepts are as informative as the configured ratio.
        if (
            len(matched) / len(query_groups) < settings["minimum_coverage"]
            and len(specific_matched) < 2
        ):
            continue
        ranked.append((index, chunk))
    if not ranked:
        return []
    bm25 = BM25Index([tokenize(search_text(chunk)) for _, chunk in ranked])
    bm25_scores = [bm25.score(scoring_terms, local_index) for local_index, _ in enumerate(ranked)]
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
        label_bonus = 0.08 if any(term in label for term in scoring_terms) else 0.0
        structural_text = " ".join(
            str(chunk.get(key) or "") for key in ("book", "chapter", "section")
        )
        structural_bonus = (
            settings["structural_weight"]
            if any(term in structural_text for term in scoring_terms)
            else 0.0
        )
        # An instrument whose title names the topic is a stronger signal than a
        # passing mention inside an unrelated law. Cap the bonus so it cannot
        # outweigh the textual evidence.
        title_tokens = set(
            tokenize(" ".join(str(chunk.get(key) or "") for key in ("title", "legal_domain")))
        )
        title_matches = sum(1 for group in query_groups if group & title_tokens)
        title_bonus = min(0.36, 0.12 * title_matches)
        score = (
            settings["lexical_weight"] * lexical
            + settings["dense_weight"] * dense
            + label_bonus
            + structural_bonus
            + title_bonus
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
                source_tier=chunk.get("source_tier") or "official",
            )
        )
    results.sort(key=lambda item: (-item.score, item.document_id, item.chunk_index))
    # Keep passages that are actually close to the best match. Without this,
    # a question about inheritance also drags in tax articles merely because
    # both mention "مليار دينار".
    if results:
        top_score = results[0].score
        if top_score > 0:
            results = [item for item in results if item.score >= 0.55 * top_score]
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
