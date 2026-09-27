"""Exact, measurable checks for explicitly labeled instruction constraints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .common import Check, candidate_response, check, expectation, normalize_text, text_contract


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    """Check visible structure and literal constraints, not semantic compliance."""
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}
    response = candidate_response(candidate)
    configured = False

    for key, label, comparator in (
        ("min_words", "minimum_word_count", lambda count, limit: count >= limit),
        ("max_words", "maximum_word_count", lambda count, limit: count <= limit),
    ):
        if key not in expected:
            continue
        configured = True
        limit = expected[key]
        valid = isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0
        passed = valid and comparator(len(response.split()), limit)
        checks.append(check(
            label,
            passed,
            f"Response has {len(response.split())} whitespace-separated words; limit is {limit!r}."
            if valid else f"{key} must be a non-negative integer.",
        ))

    if "required_order" in expected:
        configured = True
        phrases = expected["required_order"]
        valid = isinstance(phrases, list) and all(isinstance(item, str) and item for item in phrases)
        cursor = 0
        found_in_order = valid
        normalized_response = normalize_text(response)
        for phrase in phrases if valid else []:
            position = normalized_response.find(normalize_text(phrase), cursor)
            if position < 0:
                found_in_order = False
                break
            cursor = position + len(normalize_text(phrase))
        checks.append(check(
            "required_phrase_order",
            found_in_order,
            "Every labeled phrase appears in the required order."
            if found_in_order else "One or more phrases are missing or out of order.",
        ))

    if "exact_items" in expected:
        configured = True
        items = expected["exact_items"]
        actual = candidate.get("items")
        valid = isinstance(items, list) and isinstance(actual, list)
        passed = valid and actual == items
        checks.append(check(
            "exact_items",
            passed,
            "Structured items match the labeled list exactly."
            if passed else "Structured items differ from the labeled list or are missing.",
        ))

    structured = candidate.get("structured")
    if "required_fields" in expected:
        configured = True
        fields = expected["required_fields"]
        valid = isinstance(fields, list) and all(isinstance(field, str) for field in fields)
        passed = valid and isinstance(structured, Mapping) and all(field in structured for field in fields)
        checks.append(check(
            "required_structured_fields",
            passed,
            "All required structured fields are present."
            if passed else "Required structured fields are malformed or missing.",
        ))

    if "exact_fields" in expected:
        configured = True
        fields = expected["exact_fields"]
        valid = isinstance(fields, Mapping)
        passed = valid and isinstance(structured, Mapping) and all(
            key in structured and structured[key] == value for key, value in fields.items()
        )
        checks.append(check(
            "exact_structured_fields",
            passed,
            "Labeled structured field values match exactly."
            if passed else "One or more labeled structured field values differ or are missing.",
        ))

    if "required_sections" in expected:
        configured = True
        sections = expected["required_sections"]
        actual_sections = candidate.get("sections")
        valid = isinstance(sections, list) and all(isinstance(item, str) and item for item in sections)
        passed = (
            valid and isinstance(actual_sections, Mapping)
            and all(
                isinstance(actual_sections.get(section), str)
                and bool(actual_sections[section].strip())
                for section in sections
            )
        )
        checks.append(check(
            "required_sections",
            passed,
            "Every required section is present and non-empty."
            if passed else "One or more required sections are missing or empty.",
        ))

    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "instruction_ground_truth",
            False,
            "Provide a labeled text constraint, word limit, phrase order, item list, fields, or sections.",
        ))
    return checks