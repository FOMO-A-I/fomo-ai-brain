"""Shared validation and deterministic text checks for evaluation cases."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_text(value: str) -> str:
    """Normalize whitespace and case for exact, deliberately lexical checks."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def candidate_response(candidate: Any) -> str:
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, Mapping):
        response = candidate.get("response", "")
        if isinstance(response, str):
            return response
    return ""


def expectation(case: Mapping[str, Any]) -> Mapping[str, Any]:
    value = case.get("expect", {})
    return value if isinstance(value, Mapping) else {}


def text_contract(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    """Apply only explicit gold-label constraints; absent labels are not passes."""
    expected = expectation(case)
    response = candidate_response(candidate)
    checks: list[Check] = []

    exact = expected.get("exact")
    if exact is not None:
        valid = isinstance(exact, str)
        checks.append(Check(
            "exact_response",
            valid and normalize_text(response) == normalize_text(exact),
            "Response matches the labeled exact answer." if valid and normalize_text(response) == normalize_text(exact)
            else "Response does not match the labeled exact answer.",
        ))

    for key, name, should_exist in (
        ("required_terms", "required_terms", True),
        ("forbidden_terms", "forbidden_terms", False),
    ):
        terms = expected.get(key)
        if terms is None:
            continue
        valid = isinstance(terms, list) and all(isinstance(term, str) and term for term in terms)
        if not valid:
            checks.append(Check(name, False, f"{key} must be a list of non-empty strings."))
            continue
        normalized = normalize_text(response)
        missing_or_present = [
            term for term in terms
            if (normalize_text(term) in normalized) != should_exist
        ]
        checks.append(Check(
            name,
            not missing_or_present,
            "All labeled terms satisfy the constraint."
            if not missing_or_present
            else f"Constraint failed for: {missing_or_present!r}.",
        ))

    for key, name, comparator in (
        ("min_chars", "minimum_response_length", lambda size, limit: size >= limit),
        ("max_chars", "maximum_response_length", lambda size, limit: size <= limit),
    ):
        limit = expected.get(key)
        if limit is None:
            continue
        valid = isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0
        passed = valid and comparator(len(response), limit)
        checks.append(Check(
            name,
            passed,
            f"Response has {len(response)} characters; labeled limit is {limit!r}."
            if valid else f"{key} must be a non-negative integer.",
        ))

    return checks


def list_of_strings(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return value


def check(name: str, passed: bool, detail: str) -> Check:
    return Check(name=name, passed=bool(passed), detail=detail)