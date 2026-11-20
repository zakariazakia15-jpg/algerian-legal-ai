"""Optional local-only HTTP interface (install the `api` extra)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .answer import assist_query
from .assistive import (
    analyze_issue,
    compute_inheritance,
    draft_contract,
    explain_term,
    list_tools,
    risk_flags,
    review_contract,
)
from .candidate_corpus import candidate_corpus_status
from .ingest import read_corpus
from .service import answer_query, lookup_source_text

logger = logging.getLogger(__name__)

try:
    from fastapi import Depends, FastAPI, Header, HTTPException
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


# ---- Opt-in advisory toolbox models and gate ----
OPT_IN_HEADER = "X-DZ-LEGAL-AI-OPT-IN"
OPT_IN_VALUE = "advisory-tools"


def _require_opt_in(
    x_dz_legal_ai_opt_in: str | None = Header(default=None, alias=OPT_IN_HEADER),
) -> None:
    """Advisory tools must be explicitly enabled by the caller."""
    if x_dz_legal_ai_opt_in != OPT_IN_VALUE:
        raise HTTPException(
            status_code=403,
            detail=(
                "هذه أداة اختيارية؛ أرسل الرأس "
                f"{OPT_IN_HEADER}: {OPT_IN_VALUE} بعد الاطلاع على التنبيهات."
            ),
        )


class InheritanceRequest(BaseModel):
    sons: int = Field(ge=0, le=100)
    daughters: int = Field(ge=0, le=100)
    estate: float = Field(ge=0, le=1e15)

    @property
    def has_heir(self) -> bool:
        return self.sons + self.daughters > 0


class ContractDraftRequest(BaseModel):
    contract_type: str = Field(min_length=1, max_length=50)
    values: dict[str, str] = Field(default_factory=dict)


class ContractReviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class IssueRequest(AskRequest):
    """A question turned into verification topics, never into a ruling."""


class TermRequest(BaseModel):
    term: str = Field(min_length=1, max_length=200)


class RiskFlagsRequest(ContractReviewRequest):
    """Keywords turned into a generic verification checklist."""


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


# ---- Opt-in advisory toolbox endpoints ----
# These are separate from /assist. The explicit header is part of the opt-in:
# no caller can reach them accidentally through the default assistant path.


def _tool_response(call: Any, **kwargs: Any) -> dict[str, object]:
    try:
        return call(**kwargs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        logger.exception("Could not read the configured local corpus for an advisory tool")
        raise HTTPException(status_code=500, detail="Local corpus could not be read") from exc


def _tool_corpus() -> list[dict[str, Any]]:
    corpus_path = Path(os.environ.get("DZ_LEGAL_AI_CORPUS", "data/corpus.jsonl"))
    return _corpus_documents(corpus_path)


@app.get("/tools")
def toolbox(_: None = Depends(_require_opt_in)) -> dict[str, object]:
    return list_tools()


@app.post("/compute/inheritance")
def compute_shares(
    request: InheritanceRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    if not request.has_heir:
        raise HTTPException(status_code=422, detail="at least one heir is required")
    return _tool_response(
        compute_inheritance,
        sons=request.sons,
        daughters=request.daughters,
        estate=request.estate,
        documents=_tool_corpus(),
    )


@app.post("/drafts/contract")
def contract_draft(
    request: ContractDraftRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    return _tool_response(
        draft_contract,
        contract_type=request.contract_type,
        values=request.values,
        documents=_tool_corpus(),
    )


@app.post("/reviews/contract")
def contract_review(
    request: ContractReviewRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    return _tool_response(
        review_contract, text=request.text, documents=_tool_corpus()
    )


@app.post("/analysis/issue")
def issue_analysis(
    request: IssueRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    return _tool_response(
        analyze_issue, question=request.query, documents=_tool_corpus()
    )


@app.post("/terms/explain")
def term_occurrences(
    request: TermRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    return _tool_response(
        explain_term, term=request.term, documents=_tool_corpus()
    )


@app.post("/risks/flags")
def risk_checklist(
    request: RiskFlagsRequest, _: None = Depends(_require_opt_in)
) -> dict[str, object]:
    return _tool_response(risk_flags, text=request.text, documents=_tool_corpus())


def run() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
