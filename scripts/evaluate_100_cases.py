"""Run the 100 behavioural cases and write a failure report.

This measures retrieval and abstention behaviour only. It never checks whether a
legal conclusion is correct, and every case remains unsigned until reviewed.

Usage:
    $env:PYTHONPATH="src"
    $env:PYTHONIOENCODING="utf-8"
    python scripts/evaluate_100_cases.py
    python scripts/evaluate_100_cases.py --limit 5
"""

from __future__ import annotations

import argparse
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
DEFAULT_CASES = ROOT / "data" / "evaluation" / "cases_100.jsonl"
DEFAULT_REPORT = ROOT / "data" / "evaluation" / "cases_100_report.json"
DEFAULT_DOC = ROOT / "docs" / "BEHAVIORAL_100_CASES.md"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    expected_decision = case["expected_decision"]
    actual_decision = "abstain" if result["abstained"] else "answer"
    if expected_decision != actual_decision:
        problems.append(f"القرار: المتوقع {expected_decision}، الحاصل {actual_decision}")

    citations = result.get("citations", [])
    got_documents = sorted({item["document_id"] for item in citations})

    if expected_decision == "abstain":
        if citations:
            problems.append(f"امتناع مع استشهاد: {got_documents}")
    else:
        expected_documents = set(case["expected_document_ids"])
        found = expected_documents & set(got_documents)
        if not found:
            problems.append(
                f"المصدر المتوقع {sorted(expected_documents)} غير موجود؛ الحاصل {got_documents}"
            )
        if result["mode"] not in {"extractive-grounded", "generated-grounded"}:
            problems.append(f"الوضع غير متوقع: {result['mode']}")
    return not problems, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from dz_legal_ai.answer import assist_query
    from dz_legal_ai.ingest import read_corpus

    corpus = read_corpus(args.corpus)
    cases = _load_jsonl(args.cases)
    if args.limit:
        cases = cases[: args.limit]

    details: list[dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        case_started = time.time()
        result = assist_query(
            case["query"], corpus, allow_unverified=True, prefer_generation=False
        )
        ok, problems = _check(case, result)
        citations = result.get("citations", [])
        details.append(
            {
                "case_id": case["case_id"],
                "domain": case["domain"],
                "query": case["query"],
                "expected_decision": case["expected_decision"],
                "expected_document_ids": case["expected_document_ids"],
                "actual_decision": "abstain" if result["abstained"] else "answer",
                "actual_mode": result["mode"],
                "retrieved_document_ids": sorted({item["document_id"] for item in citations}),
                "retrieved_articles": [
                    {"document_id": item["document_id"], "article_label": item["article_label"]}
                    for item in citations[:5]
                ],
                "ok": ok,
                "problems": problems,
                "seconds": round(time.time() - case_started, 2),
                "approved_by": case.get("approved_by"),
            }
        )
        if index % 10 == 0 or index == len(cases):
            print(f"[{index}/{len(cases)}] elapsed={time.time() - started:.0f}s", flush=True)

    passed = sum(int(item["ok"]) for item in details)
    failures = [item for item in details if not item["ok"]]
    approved = sum(bool(item["approved_by"]) for item in details)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(details),
        "passed": passed,
        "failed": len(failures),
        "approved_by_reviewer": approved,
        "measurement": "retrieval and abstention behaviour only, no legal accuracy",
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# نتائج الاختبار السلوكي: 100 قضية في 100 مجال",
        "",
        f"- الحالات: {len(details)}",
        f"- الناجحة: {passed}",
        f"- الفاشلة: {len(failures)}",
        f"- الموقّعة من مختص: {approved}/{len(details)}",
        f"- الزمن: {time.time() - started:.0f} ثانية",
        "",
        "هذا قياس لسلوك **الاسترجاع والامتناع** فقط، ولا يدّعي دقة قانونية.",
        "كل الحالات `approved_by = null` إلى حين توقيع مختص.",
        "",
        "## الحالات الفاشلة",
        "",
    ]
    if failures:
        for item in failures:
            lines.extend(
                [
                    f"### `{item['case_id']}` — {item['domain']}",
                    "",
                    f"- السؤال: {item['query']}",
                    f"- المتوقع: {item['expected_decision']} / {item['expected_document_ids']}",
                    f"- الحاصل: {item['actual_decision']} / {item['actual_mode']} / {item['retrieved_document_ids']}",
                    f"- السبب: {'؛ '.join(item['problems'])}",
                    "",
                ]
            )
    else:
        lines.append("لا توجد حالات فاشلة.")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{passed}/{len(details)} passed; report: {args.report}; summary: {args.doc}")
    return 0 if passed == len(details) else 1


if __name__ == "__main__":
    raise SystemExit(main())
