"""Command-line entry point."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Sequence

from .candidate_corpus import (
    DEFAULT_CANDIDATE_CORPUS,
    build_candidate_corpus,
    candidate_corpus_status,
    lookup_candidate_sources,
)
from .evaluation import evaluate_cases
from .ingest import build_seed_corpus, ingest_files, read_corpus
from .local_model import DEFAULT_MODEL_DIRECTORY, LocalQwenGenerator
from .service import answer_query, lookup_source_text
from .training_data import (
    heldout_data_sha256,
    load_training_splits,
    training_data_sha256,
)

DEFAULT_CORPUS = Path("data/corpus.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dz-legal-ai",
        description="Local, citation-first Algerian legal information retrieval",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="Ask using the local verified corpus")
    ask.add_argument("query")
    ask.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ask.add_argument("--generate", action="store_true", help="Use the local Qwen model")
    ask.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)

    lookup = commands.add_parser(
        "lookup",
        help="Find exact excerpts in imported sources; does not answer legal questions",
    )
    lookup.add_argument("query")
    lookup.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    seed = commands.add_parser(
        "seed-corpus",
        help="Load bundled historical samples for source lookup (not current-law answers)",
    )
    seed.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    ingest = commands.add_parser("ingest", help="Ingest one reviewed UTF-8 text file")
    ingest.add_argument("--text", type=Path, required=True)
    ingest.add_argument("--metadata", type=Path, required=True)
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    evaluate = commands.add_parser("evaluate", help="Check locally supplied expected outcomes")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    check = commands.add_parser("corpus-check", help="Validate local corpus records and hashes")
    check.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    candidate_build = commands.add_parser(
        "candidate-corpus-build",
        help="Build a local unreviewed candidate corpus isolated from legal answers",
    )
    candidate_build.add_argument(
        "--inventory", type=Path, default=Path("data/legal_source_inventory.json")
    )
    candidate_build.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE_CORPUS)

    candidate_check = commands.add_parser(
        "candidate-corpus-check",
        help="Validate candidate corpus provenance and answer/training exclusion flags",
    )
    candidate_check.add_argument(
        "--candidate-corpus", type=Path, default=DEFAULT_CANDIDATE_CORPUS
    )

    candidate_lookup = commands.add_parser(
        "candidate-lookup",
        help="Search unreviewed candidate excerpts without treating them as legal answers",
    )
    candidate_lookup.add_argument("query")
    candidate_lookup.add_argument(
        "--candidate-corpus", type=Path, default=DEFAULT_CANDIDATE_CORPUS
    )

    model_check = commands.add_parser(
        "model-check", help="Run a local-only Qwen inference smoke test"
    )
    model_check.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIRECTORY)

    training_check = commands.add_parser(
        "training-check",
        help="Validate reviewed Algerian-law SFT data, provenance, and split isolation",
    )
    training_check.add_argument("--train", type=Path, required=True)
    training_check.add_argument("--validation", type=Path, required=True)
    training_check.add_argument("--test", type=Path, required=True)
    training_check.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower().replace("-", "")
    if encoding != "utf8":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        record = ingest_files(args.text, args.metadata, args.corpus)
        print(
            json.dumps(
                {
                    "ingested": True,
                    "document_id": record["document_id"],
                    "content_sha256": record["content_sha256"],
                    "source_document_sha256": record.get("source_document_sha256"),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "seed-corpus":
        records, added = build_seed_corpus(args.corpus)
        print(
            json.dumps(
                {
                    "built": True,
                    "documents": len(records),
                    "added": added,
                    "all_statuses_unverified": all(
                        record["legal_status"] == "unknown" for record in records
                    ),
                    "corpus": str(args.corpus),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "lookup":
        result = lookup_source_text(args.query, read_corpus(args.corpus))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate":
        cases = []
        for line_number, line in enumerate(args.cases.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid evaluation JSON on line {line_number}") from exc
            if not isinstance(case, dict):
                raise ValueError(f"Evaluation case on line {line_number} must be an object")
            cases.append(case)
        result = evaluate_cases(cases, read_corpus(args.corpus))
        print(json.dumps(result, ensure_ascii=False))
        return int(result["failed"] > 0)

    if args.command == "corpus-check":
        documents = read_corpus(args.corpus)
        print(
            json.dumps(
                {
                    "valid": True,
                    "documents": len(documents),
                    "document_ids": [document["document_id"] for document in documents],
                    "legal_domain_counts": dict(
                        sorted(Counter(document["legal_domain"] for document in documents).items())
                    ),
                    "legal_status_counts": dict(
                        sorted(Counter(document["legal_status"] for document in documents).items())
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "candidate-corpus-build":
        records = build_candidate_corpus(args.inventory, args.output)
        print(
            json.dumps(
                {
                    "built": True,
                    "candidate_documents": len(records),
                    "candidate_corpus": str(args.output),
                    "answer_eligible": False,
                    "training_eligible": False,
                    "document_ids": [record["document_id"] for record in records],
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "candidate-corpus-check":
        result = candidate_corpus_status(args.candidate_corpus)
        print(json.dumps(result, ensure_ascii=False))
        return int(not result["available"])

    if args.command == "candidate-lookup":
        result = lookup_candidate_sources(args.query, args.candidate_corpus)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "model-check":
        generator = LocalQwenGenerator(args.model_dir)
        result = generator.smoke_test()
        print(
            json.dumps(
                {
                    "model_directory": str(generator.model_directory),
                    "device": generator.device,
                    "local_only": True,
                    "smoke_test_output": result,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.command == "training-check":
        splits = load_training_splits(
            train_path=args.train,
            validation_path=args.validation,
            test_path=args.test,
            corpus_path=args.corpus,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "example_counts": {name: len(items) for name, items in splits.items()},
                    "training_validation_sha256": training_data_sha256(splits),
                    "test_sha256": heldout_data_sha256(splits),
                    "jurisdiction": "DZ",
                    "legal_answers_generated": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0

    generator = LocalQwenGenerator(args.model_dir) if args.generate else None
    result = answer_query(
        args.query, read_corpus(args.corpus), generator=generator
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
