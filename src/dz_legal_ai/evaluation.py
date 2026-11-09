"""Evaluation against locally supplied, human-reviewed expectations."""

from __future__ import annotations

from typing import Any

from .service import answer_query


def evaluate_cases(
    cases: list[dict[str, Any]], documents: list[dict[str, Any]]
) -> dict[str, int]:
    passed = 0
    for index, case in enumerate(cases, start=1):
        query = case.get("query")
        expected_abstained = case.get("expected_abstained")
        expected_ids = case.get("expected_document_ids")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Evaluation case {index} requires a non-empty query")
        if not isinstance(expected_abstained, bool):
            raise ValueError(f"Evaluation case {index} requires expected_abstained=true or false")
        if not isinstance(expected_ids, list) or not all(
            isinstance(item, str) for item in expected_ids
        ):
            raise ValueError(f"Evaluation case {index} requires expected_document_ids as a list")
        response = answer_query(query, documents)
        actual_ids = {item["document_id"] for item in response["citations"]}
        case_passed = response["abstained"] is expected_abstained
        if not expected_abstained:
            case_passed = case_passed and bool(expected_ids) and set(expected_ids) <= actual_ids
        elif expected_ids:
            case_passed = False
        passed += int(case_passed)
    return {"evaluated": len(cases), "passed": passed, "failed": len(cases) - passed}
