"""Deterministic retrieval/retention checks against explicit memory labels."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .common import Check, candidate_response, check, expectation, normalize_text, text_contract


def _id_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item for item in value)


def _threshold(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    """Score supplied memory IDs/facts; this module neither stores nor retrieves them."""
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}
    configured = False

    if "relevant_memory_ids" in expected:
        configured = True
        gold = expected["relevant_memory_ids"]
        predicted = candidate.get("memory_ids")
        min_precision = expected.get("min_precision", 1.0)
        min_recall = expected.get("min_recall", 1.0)
        valid = (
            _id_list(gold)
            and _id_list(predicted)
            and len(gold) == len(set(gold))
            and len(predicted) == len(set(predicted))
            and _threshold(min_precision)
            and _threshold(min_recall)
        )
        if valid:
            gold_set = set(gold)
            predicted_set = set(predicted)
            relevant = len(gold_set & predicted_set)
            precision = relevant / len(predicted_set) if predicted_set else (1.0 if not gold_set else 0.0)
            recall = relevant / len(gold_set) if gold_set else 1.0
            passed = precision >= min_precision and recall >= min_recall
            detail = (
                f"Memory-ID precision={precision:.3f}, recall={recall:.3f}; "
                f"required at least {min_precision:.3f}/{min_recall:.3f}."
            )
        else:
            passed = False
            detail = (
                "relevant_memory_ids and candidate memory_ids must be unique string lists; "
                "precision/recall thresholds must be finite numbers from 0 to 1."
            )
        checks.append(check("memory_retrieval", passed, detail))

    if "expected_memories" in expected:
        configured = True
        gold_facts = expected["expected_memories"]
        actual_facts = candidate.get("memories")
        valid = (
            isinstance(gold_facts, Mapping)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in gold_facts.items())
            and isinstance(actual_facts, Mapping)
            and all(isinstance(key, str) and isinstance(value, str) for key, value in actual_facts.items())
        )
        passed = valid and {
            key: normalize_text(value) for key, value in actual_facts.items()
        } == {
            key: normalize_text(value) for key, value in gold_facts.items()
        }
        checks.append(check(
            "memory_facts",
            passed,
            "Remembered fact IDs and normalized values exactly match the labels."
            if passed else "Remembered facts are missing, extra, malformed, or differ from the labels.",
        ))

    if "forbidden_memory_ids" in expected:
        configured = True
        forbidden = expected["forbidden_memory_ids"]
        predicted = candidate.get("memory_ids")
        valid = _id_list(forbidden) and _id_list(predicted)
        leaked = set(forbidden) & set(predicted) if valid else set()
        checks.append(check(
            "memory_exclusion",
            valid and not leaked,
            "No labeled-excluded memory ID was retrieved."
            if valid and not leaked
            else f"Excluded or malformed memory IDs detected: {sorted(leaked)!r}."
            if valid else "forbidden_memory_ids and candidate memory_ids must be string lists.",
        ))

    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "memory_ground_truth",
            False,
            "Provide relevant_memory_ids, expected_memories, forbidden_memory_ids, or text constraints.",
        ))
    if not candidate_response(candidate).strip() and not any(
        key in candidate for key in ("memory_ids", "memories")
    ):
        checks.append(check("memory_candidate_present", False, "No response or structured memory result was supplied."))
    return checks