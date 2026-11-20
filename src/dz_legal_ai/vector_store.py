"""Local dense-vector index for article chunks.

The index is built only on demand from a locally available sentence-transformer
model (BGE-M3 or multilingual-e5-large by default). Nothing is downloaded
implicitly: if the model or FAISS is unavailable, callers fall back to lexical
retrieval. FAISS is used when installed; otherwise an exact NumPy cosine scan
is used, which is sufficient for a local research corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # NumPy comes with the finetune/retrieval extras.
    np = None  # type: ignore[assignment]


DEFAULT_INDEX_DIR = Path("data/vector_index")
INDEX_FILE = "embeddings.npy"
META_FILE = "chunks.json"


def query_prefix(model_name: str) -> str:
    """E5 models need asymmetric prefixes; BGE-M3 does not."""
    lowered = model_name.lower()
    if "e5" in lowered:
        return "query: "
    return ""


def passage_prefix(model_name: str) -> str:
    lowered = model_name.lower()
    if "e5" in lowered:
        return "passage: "
    return ""


class VectorStore:
    """Persist and query normalized embeddings with their chunk provenance."""

    def __init__(self, directory: Path = DEFAULT_INDEX_DIR) -> None:
        self.directory = Path(directory)
        self.vectors: Any = None
        self.metadata: list[dict[str, Any]] = []
        self._faiss_index: Any = None

    def build(
        self,
        encoder: Any,
        chunks: Iterable[dict[str, Any]],
        model_name: str,
    ) -> int:
        """Encode and store chunks; return the number of indexed vectors."""
        if np is None:
            raise RuntimeError("NumPy is required to build the vector index.")
        entries = list(chunks)
        self.metadata = [
            {
                "document_id": chunk["document_id"],
                "article_label": chunk.get("article_label"),
                "chunk_index": chunk["chunk_index"],
                "book": chunk.get("book"),
                "chapter": chunk.get("chapter"),
                "section": chunk.get("section"),
                "text": chunk["text"],
            }
            for chunk in entries
        ]
        prefix = passage_prefix(model_name)
        texts = [prefix + chunk["text"] for chunk in entries]
        vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        self.vectors = np.asarray(vectors, dtype="float32")
        self._try_faiss()
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / INDEX_FILE, self.vectors)
        (self.directory / META_FILE).write_text(
            json.dumps(
                {"model_name": model_name, "chunks": self.metadata},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return len(self.metadata)

    def _try_faiss(self) -> None:
        if self.vectors is None:
            return
        try:
            import faiss
        except ImportError:
            return
        self._faiss_index = faiss.IndexFlatIP(self.vectors.shape[1])
        self._faiss_index.add(self.vectors)

    def load(self) -> bool:
        """Load a previously built index; return False when absent/invalid."""
        if np is None:
            return False
        vectors_path = self.directory / INDEX_FILE
        meta_path = self.directory / META_FILE
        if not vectors_path.is_file() or not meta_path.is_file():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.metadata = list(meta.get("chunks", []))
        self.vectors = np.load(vectors_path)
        if not self.metadata or self.vectors.shape[0] != len(self.metadata):
            return False
        self._try_faiss()
        return True

    def similarity_by_chunk(self, query_vector: list[float]) -> dict[tuple[str, int], float]:
        """Return cosine similarity keyed by (document_id, chunk_index)."""
        if np is None or self.vectors is None:
            return {}
        query = np.asarray(query_vector, dtype="float32")
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1), len(self.metadata))
            pairs = zip(indices[0].tolist(), scores[0].tolist())
        else:
            scores = self.vectors @ query
            pairs = enumerate(scores.tolist())
        return {
            (self.metadata[index]["document_id"], self.metadata[index]["chunk_index"]): float(score)
            for index, score in pairs
            if 0 <= index < len(self.metadata)
        }
