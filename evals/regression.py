"""Compare deterministic evaluation outcomes for matched case IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Regression:
    case_id: str
    status: str
    baseline_passed: bool | None
    candidate_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "baseline_passed": self.baseline_passed,
            "candidate_passed": self.candidate_passed,
        }


def compare_case(case_id: str, current: bool, baseline: bool | None) -> Regression:
    if baseline is None:
        status = "no_baseline"
    elif current and not baseline:
        status = "improved"
    elif baseline and not current:
        status = "regressed"
    else:
        status = "unchanged"
    return Regression(case_id, status, baseline, current)


def summarize_regressions(items: list[Regression]) -> dict[str, int]:
    return {
        status: sum(item.status == status for item in items)
        for status in ("improved", "regressed", "unchanged", "no_baseline")
    }