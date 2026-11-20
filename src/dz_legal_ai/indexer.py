"""On-demand construction of the local dense index over article chunks."""

from __future__ import annotations

import os
from typing import Any

from .chunking import chunk_record
from .config import corpus_quality_settings, retrieval_settings
from .policy import validate_secondary_source_url
from .vector_store import VectorStore


class IndexUnavailable(RuntimeError):
    """Raised when the local embedding model/dependency is not available."""


def build_vector_index(
    documents: list[dict[str, Any]],
    *,
    config: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Encode all eligible article chunks and persist the index locally."""
    settings = retrieval_settings(config)
    quality = corpus_quality_settings(config)
    excluded = set(quality["excluded_document_ids"])
    try:
        # Ignore an unrelated broken TensorFlow install if one is present, and
        # guarantee that an explicit index build never reaches the network.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise IndexUnavailable(
            "Building the vector index requires sentence-transformers. "
            "Install the optional retrieval dependencies first."
        ) from exc
    try:
        encoder = SentenceTransformer(settings["dense_model"], local_files_only=True)
    except Exception as exc:
        raise IndexUnavailable(
            f"Embedding model {settings['dense_model']!r} is not available locally: {exc}"
        ) from exc

    chunks: list[dict[str, Any]] = []
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
        for chunk in chunk_record(document):
            if not chunk.get("article_label"):
                continue
            chunks.append(chunk)

    store = VectorStore(settings["index_dir"])
    count = store.build(encoder, chunks, settings["dense_model"])
    return {
        "built": True,
        "indexed_chunks": count,
        "model": settings["dense_model"],
        "index_dir": str(settings["index_dir"]),
        "dense_enabled": settings["dense_enabled"],
    }
