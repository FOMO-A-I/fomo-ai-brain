"""Exact and numeric answer checks for reasoning cases."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .common import Check, candidate_response, check, expectation, text_contract

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    target = expected.get("numeric_answer")
    if target is not None:
        valid_target = isinstance(target, (int, float, str)) and not isinstance(target, bool)
        response = candidate_response(candidate).strip()
        valid_response = bool(_NUMBER.fullmatch(response))
        try:
            target_value = Decimal(str(target))
            actual_value = Decimal(response) if valid_response else Decimal("NaN")
            tolerance_raw = expected.get("tolerance", 0)
            tolerance = Decimal(str(tolerance_raw))
            valid_numbers = (
                target_value.is_finite() and actual_value.is_finite()
                and tolerance.is_finite() and tolerance >= 0
            )
            passed = valid_target and valid_numbers and abs(actual_value - target_value) <= tolerance
        except (InvalidOperation, ValueError):
            passed = False
            valid_numbers = False
        checks.append(check(
            "numeric_answer",
            passed,
            "Numeric answer is within the labeled absolute tolerance."
            if passed else
            "Response must contain only a finite number equal to the label within tolerance."
            if valid_target and valid_numbers else
            "numeric_answer and tolerance must be finite numeric values; tolerance must be non-negative.",
        ))

    if target is None and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "reasoning_ground_truth",
            False,
            "Provide a numeric answer or explicit labeled text constraints.",
        ))
    return checks