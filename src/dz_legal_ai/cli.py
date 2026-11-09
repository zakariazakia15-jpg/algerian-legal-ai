"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evaluation import evaluate_cases
from .ingest import ingest_files, read_corpus
from .service import answer_query

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

    ingest = commands.add_parser("ingest", help="Ingest one reviewed UTF-8 text file")
    ingest.add_argument("--text", type=Path, required=True)
    ingest.add_argument("--metadata", type=Path, required=True)
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    evaluate = commands.add_parser("evaluate", help="Check locally supplied expected outcomes")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    check = commands.add_parser("corpus-check", help="Validate local corpus records and hashes")
    check.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
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
                },
                ensure_ascii=False,
            )
        )
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
                },
                ensure_ascii=False,
            )
        )
        return 0

    result = answer_query(args.query, read_corpus(args.corpus))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
