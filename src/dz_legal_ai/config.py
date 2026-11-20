"""Loading of the local model/generation profile without hard dependencies."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .policy import ALGERIAN_LEGAL_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "models" / "model-config.toml"
DEFAULT_MODEL_PATH = ROOT / "models" / "base" / "Qwen2.5-1.5B-Instruct"


def _parse_value(raw: str) -> Any:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_toml_subset(content: str) -> dict[str, dict[str, Any]]:
    """Parse the small TOML subset used by model-config.toml (no tomli needed)."""
    sections: dict[str, dict[str, Any]] = {}
    current = "_default"
    sections[current] = {}
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if "=" not in stripped:
            continue
        key, raw = stripped.split("=", 1)
        sections[current][key.strip()] = _parse_value(raw)
    return sections


def load_config(path: Path | None = None) -> dict[str, dict[str, Any]]:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return {}
    return _parse_toml_subset(config_path.read_text(encoding="utf-8"))


def _section(config: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
    return dict(config.get(name, {}))


def generator_settings(config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    settings = _section(config or load_config(), "generator")
    model_path = os.environ.get("DZ_LEGAL_AI_MODEL_PATH") or settings.get("model_path")
    resolved = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    return {
        "enabled": bool(settings.get("enabled", False)),
        "model_path": resolved,
        "model_id": settings.get("model_id", "Qwen/Qwen2.5-1.5B-Instruct"),
        "temperature": float(settings.get("temperature", 0.2)),
        "top_p": float(settings.get("top_p", 0.9)),
        "top_k": int(settings.get("top_k", 20)),
        "repetition_penalty": float(settings.get("repetition_penalty", 1.05)),
        "max_new_tokens": int(settings.get("max_new_tokens", 512)),
        "timeout_seconds": int(settings.get("timeout_seconds", 180)),
    }


def retrieval_settings(config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    settings = _section(config or load_config(), "retrieval")
    index_dir = Path(settings.get("index_dir", "data/vector_index"))
    if not index_dir.is_absolute():
        index_dir = ROOT / index_dir
    return {
        "top_k": int(settings.get("top_k", 8)),
        "minimum_coverage": float(settings.get("minimum_coverage", 0.3)),
        "dense_enabled": bool(settings.get("dense_enabled", False)),
        "dense_model": settings.get("dense_model", "BAAI/bge-m3"),
        "index_dir": index_dir,
        "lexical_weight": float(settings.get("lexical_weight", 0.7)),
        "dense_weight": float(settings.get("dense_weight", 0.3)),
        "structural_weight": float(settings.get("structural_weight", 0.05)),
    }


def corpus_quality_settings(config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    settings = _section(config or load_config(), "corpus_quality")
    return {
        "excluded_document_ids": tuple(
            value.strip()
            for value in str(settings.get("excluded_document_ids", "")).split(",")
            if value.strip()
        ),
        "max_chunks_per_document": int(settings.get("max_chunks_per_document", 3)),
    }


def verification_settings(config: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    settings = _section(config or load_config(), "verification")
    return {
        "require_quote": bool(settings.get("require_quote", True)),
        "minimum_sentence_support": float(settings.get("minimum_sentence_support", 0.45)),
        "require_article_citation": bool(settings.get("require_article_citation", True)),
    }


def system_prompt(config: dict[str, dict[str, Any]] | None = None) -> str:
    settings = _section(config or load_config(), "policy")
    return str(settings.get("system_prompt") or ALGERIAN_LEGAL_SYSTEM_PROMPT)


GROUNDED_SYSTEM_PROMPT_SUFFIX = (
    " ارتباطًا بما سبق: عند الإجابة، اذكر رقم المادة أو معرّف المستند بين قوسين "
    "مثل [document_id] بعد كل فقرة، واقتبس العبارة الداعمة حرفيًا بين « ». "
    "إن لم تكفِ المقتطفات المقدمة فامتنع بوضوح ولا تكمل من معرفتك."
)
