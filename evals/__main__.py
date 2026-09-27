"""Command-line JSONL evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .benchmark import (
    benchmark_binding,
    ensure_report_does_not_overwrite_inputs,
    require_complete_external_candidates,
    sha256_file,
)
from .runner import (
    EvaluationError,
    load_baselines,
    load_candidates,
    load_cases,
    run_suite,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description=(
            "Run offline label checks or a strict, externally supplied "
            "checkpoint benchmark against deterministic JSONL labels."
        ),
    )
    parser.add_argument("--cases", required=True, help="JSONL case definitions with IDs, prompts, categories, and labels")
    parser.add_argument("--candidate", help="JSONL candidate responses keyed by case ID; otherwise cases must include candidate")
    parser.add_argument("--baseline", help="Optional JSONL baseline responses keyed by case ID")
    parser.add_argument("--json-out", help="Write the complete machine-readable report to this path")
    parser.add_argument("--fail-on-regression", action="store_true", help="Return failure if any matched baseline case regresses")
    parser.add_argument(
        "--checkpoint",
        help="Enable strict benchmarking for a verified checkpoint directory or registered checkpoint ID",
    )
    parser.add_argument(
        "--registry",
        default="training/checkpoints/registry.json",
        help="Read-only checkpoint registry path used when --checkpoint is an ID",
    )
    parser.add_argument(
        "--generation-settings-id",
        help="Operator-supplied identifier for the settings used to produce external responses (required in strict mode)",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Write a strict, provenance-bound report; requires --checkpoint, --candidate, --generation-settings-id, and --json-out",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        strict = args.checkpoint is not None or args.publish
        if args.publish and not args.checkpoint:
            raise EvaluationError("--publish requires --checkpoint ID or checkpoint directory")
        if strict and not args.candidate:
            raise EvaluationError("Strict benchmarking requires a complete external --candidate JSONL file")
        if strict and not args.generation_settings_id:
            raise EvaluationError("Strict benchmarking requires --generation-settings-id")
        if strict and not args.json_out:
            raise EvaluationError("Strict benchmarking requires --json-out to preserve the provenance-bound report")
        if not strict and args.generation_settings_id:
            raise EvaluationError("--generation-settings-id requires --checkpoint or --publish")
        ensure_report_does_not_overwrite_inputs(
            args.json_out,
            [args.cases, args.candidate, args.baseline, args.registry if strict else None],
        )

        case_hash_before = sha256_file(args.cases) if strict else None
        cases = load_cases(args.cases)
        case_hash_loaded = sha256_file(args.cases) if strict else None
        if strict and case_hash_before != case_hash_loaded:
            raise EvaluationError("Cases file changed while it was being loaded; retry with a stable file")

        candidate_hash_before = sha256_file(args.candidate) if strict else None
        candidates = load_candidates(args.candidate) if args.candidate else {}
        candidate_hash_loaded = sha256_file(args.candidate) if strict else None
        if strict and candidate_hash_before != candidate_hash_loaded:
            raise EvaluationError("Candidate file changed while it was being loaded; retry with a stable file")
        if strict:
            require_complete_external_candidates(cases, candidates)

        baselines = load_baselines(args.baseline) if args.baseline else {}
        report = run_suite(cases, candidates, baselines)
        summary = report["summary"]
        if summary["unused_candidate_ids"]:
            raise EvaluationError(f"Candidate IDs do not appear in cases: {summary['unused_candidate_ids']!r}")
        if summary["unused_baseline_ids"]:
            raise EvaluationError(f"Baseline IDs do not appear in cases: {summary['unused_baseline_ids']!r}")
        if strict:
            binding = benchmark_binding(
                checkpoint=args.checkpoint,
                registry_path=args.registry,
                cases_path=args.cases,
                candidate_path=args.candidate,
                generation_settings_id=args.generation_settings_id,
                case_count=len(cases),
                report_path=args.json_out,
            )
            if binding["case_file_sha256"] != case_hash_loaded:
                raise EvaluationError("Cases file changed during evaluation; refusing to issue a bound report")
            if binding["candidate_file_sha256"] != candidate_hash_loaded:
                raise EvaluationError("Candidate file changed during evaluation; refusing to issue a bound report")
            report["report_type"] = "strict_checkpoint_benchmark"
            report["benchmark"] = binding
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.json_out:
            output = Path(args.json_out)
            try:
                output.write_text(rendered + "\n", encoding="utf-8")
            except OSError as exc:
                raise EvaluationError(f"Cannot write report {output}: {exc}") from exc
        print(
            f"Cases: {summary['total']} | passed: {summary['passed']} "
            f"| failed: {summary['failed']} | pass rate: {summary['pass_rate']:.1%}"
        )
        for category, counts in summary["by_category"].items():
            print(f"  {category}: {counts['passed']}/{counts['total']} passed")
        regression = summary["regressions"]
        print(
            f"Baseline: {regression['improved']} improved, {regression['regressed']} regressed, "
            f"{regression['unchanged']} unchanged, {regression['no_baseline']} without baseline"
        )
        if args.json_out:
            print(f"JSON report: {args.json_out}")
        failed = summary["failed"] > 0
        regressed = args.fail_on_regression and regression["regressed"] > 0
        return 1 if failed or regressed else 0
    except EvaluationError as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())