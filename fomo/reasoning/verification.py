"""Deterministic checks that do not execute generated code."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VerificationResult:
    passed: bool
    issues: tuple[str, ...] = ()


def verify_result(
    result: str,
    *,
    require_python_syntax: bool = False,
    max_chars: int = 50_000,
) -> VerificationResult:
    """Check basic output invariants and optionally parse Python without running it."""
    issues: list[str] = []
    if not isinstance(result, str) or not result.strip():
        issues.append("result is empty")
    elif len(result) > max_chars:
        issues.append(f"result exceeds the {max_chars}-character limit")
    elif require_python_syntax:
        try:
            ast.parse(result)
        except SyntaxError as error:
            issues.append(f"Python syntax error on line {error.lineno}: {error.msg}")
    return VerificationResult(not issues, tuple(issues))