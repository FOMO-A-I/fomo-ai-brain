"""Coding-response checks using labels or outputs supplied by an external runner.

This module compares data only. It never imports, compiles, or executes model code.
"""

from __future__ import annotations

from typing import Any, Mapping

from .common import Check, candidate_response, check, expectation, text_contract


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}

    labeled_tests = expected.get("tests")
    if labeled_tests is not None:
        valid_labels = isinstance(labeled_tests, list) and all(
            isinstance(test, Mapping) and isinstance(test.get("id"), str)
            and "expected" in test
            for test in labeled_tests
        ) and bool(labeled_tests)
        actual_tests = candidate.get("execution_results", [])
        valid_actual = isinstance(actual_tests, list) and all(
            isinstance(test, Mapping) and isinstance(test.get("id"), str)
            and "actual" in test
            for test in actual_tests
        )
        actual_by_id = {
            test["id"]: test["actual"] for test in actual_tests
            if isinstance(test, Mapping) and isinstance(test.get("id"), str)
            and "actual" in test
        } if isinstance(actual_tests, list) else {}
        expected_ids = [test["id"] for test in labeled_tests] if valid_labels else []
        unique_ids = len(expected_ids) == len(set(expected_ids))
        passed = (
            valid_labels and valid_actual and unique_ids
            and len(actual_by_id) == len(actual_tests)
            and all(actual_by_id.get(test["id"]) == test["expected"] for test in labeled_tests)
        )
        checks.append(check(
            "externally_reported_test_outputs",
            passed,
            "Supplied execution outputs match every labeled output."
            if passed else
            "Execution output data is missing, malformed, duplicated, or differs from the labels. "
            "The evaluator did not run or independently verify the code.",
        ))

    has_text_labels = any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    ))
    if labeled_tests is None and not has_text_labels:
        checks.append(check(
            "coding_ground_truth",
            False,
            "Provide explicit text constraints or externally produced labeled test outputs.",
        ))
    if not candidate_response(candidate).strip():
        checks.append(check("non_empty_response", False, "Candidate response is empty."))
    return checks