"""Strict validation for locally reviewed supervised fine-tuning examples."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from .ingest import read_corpus
from .policy import (
    ALGERIAN_LEGAL_SYSTEM_PROMPT,
    ALLOWED_REUSE_BASES,
    validate_reuse_evidence_for,
    validate_source_url,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXAMPLE_LICENSES = {"CC0-1.0", "CC-BY-4.0"}
_TASKS = {"grounded_answer", "abstention"}


def _validate_example(
    example: Any,
    *,
    line_number: int,
    expected_split: str,
    corpus_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    location = f"{expected_split} line {line_number}"
    if not isinstance(example, dict):
        raise ValueError(f"{location}: example must be a JSON object")
    required = {
        "schema_version",
        "example_id",
        "split",
        "jurisdiction",
        "language",
        "task",
        "messages",
        "sources",
        "example_license",
        "review_status",
        "reviewer_role",
        "reviewed_on",
    }
    missing = sorted(required - example.keys())
    if missing:
        raise ValueError(f"{location}: missing fields: {', '.join(missing)}")
    if example["schema_version"] != 1:
        raise ValueError(f"{location}: schema_version must be 1")
    if not isinstance(example["example_id"], str) or not example["example_id"].strip():
        raise ValueError(f"{location}: example_id must be non-empty")
    if example["split"] != expected_split:
        raise ValueError(f"{location}: split must be {expected_split}")
    if example["jurisdiction"] != "DZ":
        raise ValueError(f"{location}: only Algerian jurisdiction (DZ) is supported")
    if not isinstance(example["language"], str) or example["language"] not in {"ar", "fr"}:
        raise ValueError(f"{location}: language must be ar or fr")
    if not isinstance(example["task"], str) or example["task"] not in _TASKS:
        raise ValueError(f"{location}: task must be grounded_answer or abstention")
    if example["review_status"] != "approved":
        raise ValueError(f"{location}: only legally reviewed, approved examples can be used")
    if not isinstance(example["reviewer_role"], str) or not example["reviewer_role"].strip():
        raise ValueError(f"{location}: reviewer_role must identify the review qualification")
    try:
        parsed_date = date.fromisoformat(example["reviewed_on"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location}: reviewed_on must be an ISO date") from exc
    if parsed_date.isoformat() != example["reviewed_on"]:
        raise ValueError(f"{location}: reviewed_on must use YYYY-MM-DD")
    if not isinstance(example["example_license"], str) or example[
        "example_license"
    ] not in _EXAMPLE_LICENSES:
        raise ValueError(
            f"{location}: authored example license must be CC0-1.0 or CC-BY-4.0"
        )

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"{location}: messages must include user and assistant turns")
    allowed_roles = {"system", "user", "assistant"}
    if any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or message.get("role") not in allowed_roles
        or not isinstance(message.get("content"), str)
        or not message["content"].strip()
        for message in messages
    ):
        raise ValueError(f"{location}: each message needs a supported role and non-empty content")
    roles = [message["role"] for message in messages]
    if roles.count("user") != 1 or roles[-1] != "assistant" or roles.count("assistant") != 1:
        raise ValueError(
            f"{location}: one user turn and one final assistant turn are required"
        )
    if roles.count("system") != 1 or roles[0] != "system":
        raise ValueError(f"{location}: the fixed Algerian-law-only system message is required first")
    if messages[0]["content"] != ALGERIAN_LEGAL_SYSTEM_PROMPT:
        raise ValueError(
            f"{location}: system prompt must enforce Algerian-law-only scope and abstention"
        )

    sources = example["sources"]
    if not isinstance(sources, list):
        raise ValueError(f"{location}: sources must be an array")
    if example["task"] == "grounded_answer" and not sources:
        raise ValueError(f"{location}: grounded answers require at least one source")
    if example["task"] == "abstention" and sources:
        raise ValueError(f"{location}: abstention examples must not claim legal sources")
    seen_sources: set[str] = set()
    answer = messages[-1]["content"]
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"{location}: each source must be an object")
        source_fields = {
            "document_id",
            "legal_instrument_id",
            "legal_domain",
            "source_url",
            "source_locator",
            "content_sha256",
            "reuse_basis",
            "reuse_evidence_url",
            "excerpt",
            "legal_status",
            "legal_status_checked_on",
            "legal_status_evidence_urls",
        }
        missing_source = sorted(source_fields - source.keys())
        if missing_source:
            raise ValueError(
                f"{location}: source missing fields: {', '.join(missing_source)}"
            )
        document_id = source["document_id"]
        if not isinstance(document_id, str) or not document_id or document_id in seen_sources:
            raise ValueError(f"{location}: source document IDs must be unique non-empty strings")
        seen_sources.add(document_id)
        if not isinstance(source["source_locator"], str) or not source["source_locator"].strip():
            raise ValueError(f"{location}: source_locator must be non-empty")
        if not isinstance(source["content_sha256"], str) or not _SHA256.fullmatch(
            source["content_sha256"]
        ):
            raise ValueError(f"{location}: source content_sha256 must be lowercase SHA-256")
        if not isinstance(source["reuse_basis"], str) or source[
            "reuse_basis"
        ] not in ALLOWED_REUSE_BASES:
            raise ValueError(f"{location}: source reuse_basis is unsupported")
        excerpt = source["excerpt"]
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise ValueError(f"{location}: source excerpt must be non-empty")
        validate_source_url(source["source_url"])
        validate_reuse_evidence_for(source["reuse_basis"], source["reuse_evidence_url"])
        corpus_source = corpus_by_id.get(document_id)
        if corpus_source is None:
            raise ValueError(f"{location}: source {document_id} is absent from the validated corpus")
        for field in ("source_url", "source_locator", "content_sha256", "reuse_basis"):
            if source[field] != corpus_source[field]:
                raise ValueError(
                    f"{location}: source {document_id} {field} does not match the corpus"
                )
        if source["reuse_evidence_url"] != corpus_source["reuse_evidence_url"]:
            raise ValueError(
                f"{location}: source {document_id} rights evidence does not match the corpus"
            )
        if source["legal_status"] != corpus_source["legal_status"]:
            raise ValueError(f"{location}: source {document_id} legal status does not match corpus"            )
        if source["legal_domain"] != corpus_source["legal_domain"]:
            raise ValueError(f"{location}: source {document_id} legal domain does not match corpus")
        if source["legal_instrument_id"] != corpus_source["legal_instrument_id"]:
            raise ValueError(
                f"{location}: source {document_id} legal instrument does not match corpus"
            )
        if source["legal_status"] != "verified_current":
            raise ValueError(
                f"{location}: grounded SFT answers require officially verified current law"
            )
        if source["legal_status_checked_on"] != corpus_source["legal_status_checked_on"]:
            raise ValueError(
                f"{location}: source {document_id} legal-status date does not match corpus"
            )
        if source["legal_status_evidence_urls"] != corpus_source["legal_status_evidence_urls"]:
            raise ValueError(
                f"{location}: source {document_id} legal-status evidence does not match corpus"
            )
        prompt_text = "\n".join(message["content"] for message in messages[:-1])
        if excerpt not in corpus_source["text"] or excerpt not in prompt_text:
            raise ValueError(
                f"{location}: source excerpt must match the corpus and appear in the prompt"
            )
        if f"[{document_id}]" not in answer:
            raise ValueError(
                f"{location}: grounded answer must cite each source as [{document_id}]"
            )
    if example["task"] == "abstention":
        if example.get("abstention_reason") not in {
            "insufficient_evidence",
            "outside_jurisdiction",
            "personalized_advice",
            "unverified_current_status",
        }:
            raise ValueError(f"{location}: abstention_reason must identify a supported reason")
    elif "abstention_reason" in example:
        raise ValueError(f"{location}: grounded answers must not include abstention_reason")
    return example


def load_training_splits(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    corpus_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load reviewed SFT examples and enforce provenance and document-level split isolation."""
    corpus = read_corpus(corpus_path)
    if not corpus:
        raise ValueError("A non-empty, validated legal corpus is required for SFT data")
    corpus_by_id = {record["document_id"]: record for record in corpus}
    paths = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    seen_examples: set[str] = set()
    source_splits: dict[str, str] = {}
    for split, path in paths.items():
        examples: list[dict[str, Any]] = []
        if not path.is_file():
            raise ValueError(f"Missing {split} dataset file: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{split} line {line_number}: invalid JSON") from exc
            example = _validate_example(
                example,
                line_number=line_number,
                expected_split=split,
                corpus_by_id=corpus_by_id,
            )
            example_id = example["example_id"]
            if example_id in seen_examples:
                raise ValueError(f"Duplicate example_id across splits: {example_id}")
            seen_examples.add(example_id)
            for source in example["sources"]:
                instrument_id = source["legal_instrument_id"]
                prior_split = source_splits.setdefault(instrument_id, split)
                if prior_split != split:
                    raise ValueError(
                        f"Legal instrument {instrument_id} appears in both {prior_split} and "
                        f"{split}; split by instrument to avoid version leakage"
                    )
            examples.append(example)
        if not examples:
            raise ValueError(f"{split} dataset must contain at least one approved example")
        loaded[split] = examples
    return loaded


def load_test_examples(*, test_path: Path, corpus_path: Path) -> list[dict[str, Any]]:
    corpus = read_corpus(corpus_path)
    if not corpus:
        raise ValueError("A non-empty, validated legal corpus is required for evaluation")
    corpus_by_id = {record["document_id"]: record for record in corpus}
    if not test_path.is_file():
        raise ValueError(f"Missing test dataset file: {test_path}")
    examples = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(test_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"test line {line_number}: invalid JSON") from exc
        example = _validate_example(
            example,
            line_number=line_number,
            expected_split="test",
            corpus_by_id=corpus_by_id,
        )
        if example["example_id"] in seen_ids:
            raise ValueError(f"Duplicate test example_id: {example['example_id']}")
        seen_ids.add(example["example_id"])
        examples.append(example)
    if not examples:
        raise ValueError("test dataset must contain at least one approved example")
    return examples


def training_data_sha256(splits: dict[str, list[dict[str, Any]]]) -> str:
    """Hash training and validation records, excluding the held-out test split."""
    training_splits = {name: splits[name] for name in ("train", "validation")}
    canonical = json.dumps(
        training_splits, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def heldout_data_sha256(splits: dict[str, list[dict[str, Any]]]) -> str:
    canonical = json.dumps(
        splits["test"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
