"""Optional local-only HTTP interface (install the `api` extra)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .ingest import read_corpus
from .service import answer_query

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
    version="0.1.1",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
