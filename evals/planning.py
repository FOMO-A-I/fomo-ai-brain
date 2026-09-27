"""Deterministic structural checks for labeled plans."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .common import Check, candidate_response, check, expectation, text_contract


def _steps(candidate: Any) -> Any:
    if isinstance(candidate, Mapping) and "steps" in candidate:
        return candidate["steps"]
    response = candidate_response(candidate)
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed.get("steps") if isinstance(parsed, Mapping) else None


def _label(step: Mapping[str, Any]) -> str | None:
    for key in ("label", "title", "task", "name"):
        value = step.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    steps = _steps(candidate)
    steps_valid = isinstance(steps, list) and all(
        isinstance(step, Mapping) and _label(step) is not None for step in steps
    )
    labeled = steps if steps_valid else []
    configured = False

    if "required_steps" in expected:
        configured = True
        required = expected["required_steps"]
        valid = isinstance(required, list) and all(isinstance(item, str) for item in required)
        actual = [_label(step) for step in labeled]
        passed = valid and steps_valid and all(item in actual for item in required)
        checks.append(check(
            "required_plan_steps",
            passed,
            "All labeled steps are present." if passed else "The plan is malformed or misses labeled steps.",
        ))

    if "expected_steps" in expected:
        configured = True
        gold = expected["expected_steps"]
        valid = isinstance(gold, list) and all(isinstance(item, str) for item in gold)
        actual = [_label(step) for step in labeled]
        passed = valid and steps_valid and actual == gold
        checks.append(check(
            "ordered_plan_steps",
            passed,
            "Plan step labels and order match the reference."
            if passed else "Plan steps or their order differ from the reference.",
        ))

    if "max_steps" in expected:
        configured = True
        limit = expected["max_steps"]
        valid = isinstance(limit, int) and not isinstance(limit, bool) and limit >= 0
        passed = valid and steps_valid and len(labeled) <= limit
        checks.append(check(
            "plan_step_limit",
            passed,
            f"Plan has {len(labeled)} steps; maximum is {limit!r}."
            if valid else "max_steps must be a non-negative integer.",
        ))

    if "check_dependencies" in expected:
        configured = True
        ids: list[str] = []
        dependency_shape_valid = steps_valid
        earlier: set[str] = set()
        passed = steps_valid
        for step in labeled:
            step_id = step.get("id")
            dependencies = step.get("depends_on", [])
            if not isinstance(step_id, str) or not step_id or step_id in ids:
                dependency_shape_valid = False
                passed = False
                continue
            if not isinstance(dependencies, list) or not all(isinstance(dep, str) for dep in dependencies):
                dependency_shape_valid = False
                passed = False
            elif any(dep not in earlier for dep in dependencies):
                passed = False
            ids.append(step_id)
            earlier.add(step_id)
        passed = expected["check_dependencies"] is True and passed and dependency_shape_valid
        checks.append(check(
            "plan_dependencies",
            passed,
            "Every dependency points to one unique earlier step."
            if passed else "Dependencies are malformed, duplicated, missing, or refer to a later step.",
        ))

    if not steps_valid:
        checks.append(check(
            "plan_schema",
            False,
            "Candidate must include a steps list with a non-empty label/title/task/name per step.",
        ))
    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "planning_ground_truth",
            False,
            "Provide labeled plan steps, a step limit, dependency checks, or text constraints.",
        ))
    return checks