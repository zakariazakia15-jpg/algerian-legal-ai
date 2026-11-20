"""Optional local-only HTTP interface (install the `api` extra)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .answer import assist_query
from .candidate_corpus import candidate_corpus_status
from .ingest import read_corpus
from .service import answer_query, lookup_source_text

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, RedirectResponse
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

UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


class AssistRequest(AskRequest):
    """Same input as /ask, served by the grounded assistant path."""


@lru_cache(maxsize=4)
def _load_corpus(
    path: str, mtime_ns: int | None, size: int | None
) -> tuple[dict[str, Any], ...]:
    return tuple(read_corpus(Path(path)))


def _corpus_documents(path: Path) -> list[dict[str, Any]]:
    """Cache the local corpus until its path, timestamp or size changes."""
    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (None, None)
    return list(_load_corpus(str(path), *signature))


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui() -> FileResponse:
    if not UI_PATH.is_file():
        raise HTTPException(status_code=500, detail="Local UI file is missing")
    return FileResponse(UI_PATH, media_type="text/html; charset=utf-8")


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


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    corpus_path = Path(os.environ.get("DZ_LEGAL_AI_CORPUS", "data/corpus.jsonl"))
    try:
        return answer_query(request.query, read_corpus(corpus_path))
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured local corpus")
        raise HTTPException(status_code=500, detail="Local corpus could not be read") from exc


@app.post("/assist")
def assist(request: AssistRequest) -> dict[str, object]:
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="query must not be blank")
    corpus_path = Path(os.environ.get("DZ_LEGAL_AI_CORPUS", "data/corpus.jsonl"))
    try:
        # HTTP stays informational and extractive: local model generation is CLI-only.
        return assist_query(
            request.query,
            _corpus_documents(corpus_path),
            allow_unverified=True,
            prefer_generation=False,
        )
    except (OSError, ValueError) as exc:
        logger.exception("Could not read the configured local corpus for assistance")
        raise HTTPException(status_code=500, detail="Local corpus could not be read") from exc


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
