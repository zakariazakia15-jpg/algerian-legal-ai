"""Optional local-only HTTP interface (install the `api` extra)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from .candidate_corpus import candidate_corpus_status, lookup_candidate_sources
from .ingest import read_corpus
from .local_model import DEFAULT_MODEL_DIRECTORY, LocalQwenGenerator
from .service import answer_query, lookup_source_text

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError(
        "HTTP support requires optional dependencies; install with `pip install -e '.[api]'`"
    ) from exc

app = FastAPI(
    title="Algerian Legal AI",
    description="Local citation-first retrieval. Informational only, not legal advice.",
    version="0.1.8",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    generate: bool = False


@lru_cache(maxsize=2)
def _local_generator(model_directory: str) -> LocalQwenGenerator:
    return LocalQwenGenerator(Path(model_directory))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/candidate-corpus/status")
def candidate_corpus_health() -> dict[str, object]:
    corpus_path = Path(
        os.environ.get("DZ_LEGAL_AI_CANDIDATE_CORPUS", "data/candidate-corpus.jsonl")
    )
    try:
        return candidate_corpus_status(corpus_path)
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured local candidate corpus")
        raise HTTPException(
            status_code=500, detail="Local candidate corpus could not be read"
        ) from exc


@app.post("/candidate-lookup")
def candidate_lookup(request: AskRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    corpus_path = Path(
        os.environ.get("DZ_LEGAL_AI_CANDIDATE_CORPUS", "data/candidate-corpus.jsonl")
    )
    try:
        return lookup_candidate_sources(request.query, corpus_path)
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured candidate corpus")
        raise HTTPException(
            status_code=500, detail="Local candidate corpus could not be read"
        ) from exc


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    corpus_path = Path(os.environ.get("DZ_LEGAL_AI_CORPUS", "data/corpus.jsonl"))
    try:
        documents = read_corpus(corpus_path)
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured local corpus")
        raise HTTPException(status_code=500, detail="Local corpus could not be read") from exc
    generator = None
    if request.generate:
        model_directory = os.environ.get(
            "DZ_LEGAL_AI_MODEL_DIR", str(DEFAULT_MODEL_DIRECTORY)
        )
        generator = _local_generator(str(Path(model_directory).expanduser().resolve()))
    try:
        return answer_query(request.query, documents, generator=generator)
    except (OSError, RuntimeError) as exc:
        logger.exception("Local Qwen inference failed")
        raise HTTPException(
            status_code=503, detail="Local model inference unavailable"
        ) from exc


@app.post("/lookup")
def lookup(request: AskRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    corpus_path = Path(os.environ.get("DZ_LEGAL_AI_CORPUS", "data/corpus.jsonl"))
    try:
        return lookup_source_text(request.query, read_corpus(corpus_path))
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured local corpus for source lookup")
        raise HTTPException(status_code=500, detail="Local corpus could not be read") from exc


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
