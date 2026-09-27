"""Structured tool-call checks. Calls are inspected, never executed."""

from __future__ import annotations

from typing import Any, Mapping

from .common import Check, check, expectation, text_contract


def _tool_name(call: Any) -> str | None:
    if not isinstance(call, Mapping):
        return None
    name = call.get("name")
    if isinstance(name, str):
        return name
    function = call.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    return None


def _arguments(call: Any) -> Mapping[str, Any] | None:
    if not isinstance(call, Mapping):
        return None
    arguments = call.get("arguments")
    if arguments is None and isinstance(call.get("function"), Mapping):
        arguments = call["function"].get("arguments", {})
    return arguments if isinstance(arguments, Mapping) else None


def evaluate(case: Mapping[str, Any], candidate: Any) -> list[Check]:
    checks = text_contract(case, candidate)
    expected = expectation(case)
    if not isinstance(candidate, Mapping):
        candidate = {}
    calls = candidate.get("tool_calls", [])
    calls_valid = isinstance(calls, list) and all(
        _tool_name(call) and _arguments(call) is not None for call in calls
    )
    names = [_tool_name(call) for call in calls] if isinstance(calls, list) else []
    configured = False

    allowed = expected.get("allowed_tools")
    if allowed is not None:
        configured = True
        valid = isinstance(allowed, list) and all(isinstance(name, str) for name in allowed)
        passed = valid and calls_valid and all(name in allowed for name in names)
        checks.append(check(
            "allowed_tools",
            passed,
            "All calls use an explicitly allowed tool."
            if passed else "A tool call is malformed or not in the labeled allowlist.",
        ))

    must_call = expected.get("must_call")
    if must_call is not None:
        configured = True
        valid = isinstance(must_call, list) and all(isinstance(name, str) for name in must_call)
        passed = valid and calls_valid and all(names.count(name) >= must_call.count(name) for name in set(must_call))
        checks.append(check(
            "required_tool_calls",
            passed,
            "All required tools were called." if passed else "One or more required tool calls are missing or malformed.",
        ))

    must_not_call = expected.get("must_not_call")
    if must_not_call is not None:
        configured = True
        valid = isinstance(must_not_call, list) and all(isinstance(name, str) for name in must_not_call)
        passed = valid and calls_valid and not any(name in must_not_call for name in names)
        checks.append(check(
            "forbidden_tool_calls",
            passed,
            "No forbidden tools were called." if passed else "A forbidden or malformed tool call was present.",
        ))

    expected_calls = expected.get("expected_calls")
    if expected_calls is not None:
        configured = True
        valid = isinstance(expected_calls, list) and all(
            _tool_name(call) is not None and _arguments(call) is not None for call in expected_calls
        )
        actual = [
            {"name": _tool_name(call), "arguments": dict(_arguments(call) or {})}
            for call in calls
        ] if calls_valid else []
        gold = [
            {"name": _tool_name(call), "arguments": dict(_arguments(call) or {})}
            for call in expected_calls
        ] if valid else []
        passed = valid and calls_valid and actual == gold
        checks.append(check(
            "exact_tool_calls",
            passed,
            "Tool names, arguments, and order match the labels."
            if passed else "Tool calls differ from the labeled names, arguments, or order.",
        ))

    if not calls_valid:
        checks.append(check(
            "tool_call_schema",
            False,
            "tool_calls must be a list of calls with a name and object-valued arguments.",
        ))
    if not configured and not any(key in expected for key in (
        "exact", "required_terms", "forbidden_terms", "min_chars", "max_chars"
    )):
        checks.append(check(
            "tool_use_ground_truth",
            False,
            "Provide an allowlist, required/forbidden calls, exact calls, or text constraints.",
        ))
    return checks