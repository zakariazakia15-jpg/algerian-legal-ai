"""Run the (unapproved) behavioural benchmark against the grounded assistant.

This measures retrieval/safety behaviour, not legal accuracy. Every case with
approved_by=null is reported as NOT APPROVED and must be signed by a qualified
reviewer before the numbers can be quoted.

Usage:
    $env:PYTHONPATH="src"
    $env:PYTHONIOENCODING="utf-8"
    python scripts/evaluate_benchmark.py
    python scripts/evaluate_benchmark.py --with-generation
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus.jsonl"
DEFAULT_BENCHMARK = ROOT / "data" / "evaluation" / "benchmark.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from dz_legal_ai.answer import assist_query
    from dz_legal_ai.ingest import read_corpus

    corpus = read_corpus(args.corpus)
    cases = _load_jsonl(args.benchmark)
    if args.limit:
        cases = cases[: args.limit]
    passed = 0
    approved = 0

    for index, case in enumerate(cases, start=1):
        case_started = time.time()
        result = assist_query(
            case["query"],
            corpus,
            allow_unverified=True,
            prefer_generation=args.with_generation,
        )
        citations = result.get("citations", [])
        got_documents = sorted({item["document_id"] for item in citations})
        got_labels = sorted({item.get("article_label") or "" for item in citations})
        expected_documents = list(case.get("expected_document_ids", []))
        expected_labels = list(case.get("expected_article_labels", []))

        if not expected_documents:
            document_match = not got_documents
        elif any("*" in item or "?" in item for item in expected_documents):
            # Wildcards describe acceptable alternatives, e.g. "orac-*".
            document_match = any(
                any(fnmatch.fnmatch(got, expected) for got in got_documents)
                for expected in expected_documents
            )
        else:
            document_match = all(
                any(fnmatch.fnmatch(got, expected) for got in got_documents)
                for expected in expected_documents
            )
        label_match = (
            not expected_labels
            or "any" in expected_labels
            or any(label in got_labels for label in expected_labels)
        )
        mode_match = result["mode"] == case["expected_mode"]
        abstention_match = result["abstained"] == case.get("expected_abstention", False)
        ok = document_match and label_match and mode_match and abstention_match
        passed += int(ok)
        is_approved = bool(case.get("approved_by") and case.get("approved_on"))
        approved += int(is_approved)

        print(
            f"[{index:02d}] {'OK  ' if ok else 'FAIL'} ({time.time() - case_started:.1f}s) | {case['query']}",
            flush=True,
        )
        print(
            f"     mode={result['mode']} abstained={result['abstained']} "
            f"documents={got_documents} articles={got_labels}"
        )
        if not ok:
            print(
                f"     expected mode={case['expected_mode']} "
                f"abstention={case.get('expected_abstention')} "
                f"documents={expected_documents} articles={expected_labels}"
            )
        if not is_approved:
            print("     WARNING: not approved by a qualified reviewer (template only)")

    print(
        f"\n{passed}/{len(cases)} behavioural cases matched; "
        f"{approved}/{len(cases)} approved by a reviewer."
    )
    print("These are safety/retrieval checks only; no legal accuracy is claimed.")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
