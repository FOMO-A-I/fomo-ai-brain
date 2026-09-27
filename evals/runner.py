"""JSONL loading, category dispatch, evaluation, and baseline comparison."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import (
    agent_coordination,
    coding,
    hallucination,
    instruction_following,
    memory,
    planning,
    reasoning,
    safety,
    tool_use,
)
from .common import Check, candidate_response
from .regression import Regression, compare_case, summarize_regressions

EVALUATORS = {
    "coding": coding.evaluate,
    "reasoning": reasoning.evaluate,
    "tool_use": tool_use.evaluate,
    "planning": planning.evaluate,
    "hallucination": hallucination.evaluate,
    "agent_coordination": agent_coordination.evaluate,
    "memory": memory.evaluate,
    "instruction_following": instruction_following.evaluate,
    "safety": safety.evaluate,
    "regression": reasoning.evaluate,
}


class EvaluationError(ValueError):
    """Invalid evaluation input, reported with file and line where possible."""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: tuple[Check, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "category": self.category,
            "passed": self.passed,
            "checks": [item.as_dict() for item in self.checks],
        }


def validate_case(case: Any, *, source: str = "case") -> Mapping[str, Any]:
    if not isinstance(case, Mapping):
        raise EvaluationError(f"{source}: each case must be a JSON object")
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise EvaluationError(f"{source}: every case needs a non-empty string id")
    category = case.get("category")
    if category not in EVALUATORS:
        raise EvaluationError(
            f"{source}: case {case_id!r} has unsupported category {category!r}; "
            f"choose one of {', '.join(EVALUATORS)}"
        )
    if not isinstance(case.get("prompt"), str) or not case["prompt"].strip():
        raise EvaluationError(f"{source}: case {case_id!r} needs a non-empty prompt")
    expect = case.get("expect", {})
    if not isinstance(expect, Mapping):
        raise EvaluationError(f"{source}: case {case_id!r} expect must be an object")
    return case


def evaluate_case(case: Mapping[str, Any], candidate: Any) -> CaseResult:
    validated = validate_case(case)
    evaluator = EVALUATORS[validated["category"]]
    checks = tuple(evaluator(validated, candidate))
    if candidate is None:
        checks = (*checks, Check(
            name="candidate_present",
            passed=False,
            detail="No candidate response was supplied for this case ID.",
        ))
    return CaseResult(
        case_id=validated["id"],
        category=validated["category"],
        passed=bool(checks) and all(item.passed for item in checks),
        checks=checks,
    )


def read_jsonl(path: str | Path, *, kind: str) -> dict[str, Mapping[str, Any]]:
    """Read arbitrary-length JSONL input, rejecting malformed or duplicate IDs."""
    source = Path(path)
    records: dict[str, Mapping[str, Any]] = {}
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"Cannot open {kind} file {source}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{source}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, Mapping):
                raise EvaluationError(f"{where}: expected a JSON object")
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id.strip():
                raise EvaluationError(f"{where}: every record needs a non-empty string id")
            if record_id in records:
                raise EvaluationError(f"{where}: duplicate id {record_id!r}")
            records[record_id] = record
    return records


def _prediction_from_record(record: Mapping[str, Any]) -> Any:
    if "candidate" in record:
        return record["candidate"]
    if "response" in record:
        return {key: value for key, value in record.items() if key != "id"}
    return None


def run_suite(
    cases: Iterable[Mapping[str, Any]],
    candidates: Mapping[str, Any] | None = None,
    baselines: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate supplied outputs against cases; no model or generated code is run."""
    candidate_map = dict(candidates or {})
    baseline_map = dict(baselines or {})
    case_results: list[CaseResult] = []
    baseline_results: dict[str, CaseResult] = {}
    regressions: list[Regression] = []
    case_ids: set[str] = set()

    for raw_case in cases:
        case = validate_case(raw_case)
        case_id = case["id"]
        if case_id in case_ids:
            raise EvaluationError(f"Duplicate case id {case_id!r}")
        case_ids.add(case_id)
        candidate = candidate_map.get(case_id, case.get("candidate"))
        result = evaluate_case(case, candidate)
        case_results.append(result)
        baseline = baseline_map.get(case_id, case.get("baseline"))
        if baseline is None:
            baseline_passed = None
        else:
            baseline_result = evaluate_case(case, baseline)
            baseline_results[case_id] = baseline_result
            baseline_passed = baseline_result.passed
        regressions.append(compare_case(case_id, result.passed, baseline_passed))

    unused_candidates = sorted(set(candidate_map) - case_ids)
    unused_baselines = sorted(set(baseline_map) - case_ids)
    counts = Counter(result.category for result in case_results)
    by_category: dict[str, dict[str, int]] = {}
    for category, total in sorted(counts.items()):
        passed = sum(result.passed for result in case_results if result.category == category)
        by_category[category] = {"total": total, "passed": passed, "failed": total - passed}

    total = len(case_results)
    passed_total = sum(result.passed for result in case_results)
    return {
        "report_type": "offline_label_check",
        "candidate_provenance": "caller-supplied response data; model/checkpoint identity is unbound",
        "responses_generated_by_evaluator": False,
        "checkpoint_binding": None,
        "summary": {
            "total": total,
            "passed": passed_total,
            "failed": total - passed_total,
            "pass_rate": passed_total / total if total else 0.0,
            "by_category": by_category,
            "regressions": summarize_regressions(regressions),
            "unused_candidate_ids": unused_candidates,
            "unused_baseline_ids": unused_baselines,
        },
        "cases": [result.as_dict() for result in case_results],
        "regression_details": [result.as_dict() for result in regressions],
        "baseline_cases": [result.as_dict() for result in baseline_results.values()],
        "limitations": [
            "Checks are deterministic proxies against supplied labels, not a general measure of intelligence.",
            "Generated code is never executed; optional execution outputs are compared as supplied data only.",
            "Claim matching is lexical and citation checks validate IDs, not semantic entailment or truth.",
            "Memory scores cover supplied IDs/facts, not actual retrieval quality, retention, privacy, or internal recall.",
            "Instruction-following scores cover literal and structural constraints, not semantic understanding.",
            "Safety checks do not detect jailbreaks or perform semantic moderation; refusal labels and action logs must be independently reviewed.",
        ],
    }


def load_candidates(path: str | Path) -> dict[str, Any]:
    records = read_jsonl(path, kind="candidate")
    candidates: dict[str, Any] = {}
    for record_id, record in records.items():
        candidate = _prediction_from_record(record)
        if candidate is None:
            raise EvaluationError(f"Candidate {record_id!r} needs a candidate or response field")
        candidates[record_id] = candidate
    return candidates


def load_cases(path: str | Path) -> list[Mapping[str, Any]]:
    records = read_jsonl(path, kind="case")
    return [validate_case(record, source=str(path)) for record in records.values()]


def load_baselines(path: str | Path) -> dict[str, Any]:
    return load_candidates(path)