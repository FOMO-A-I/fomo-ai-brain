"""Provenance helpers for strict, externally supplied checkpoint benchmarks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .runner import EvaluationError


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise EvaluationError(f"Cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def checkpoint_identity(
    checkpoint: str,
    registry_path: str | Path,
) -> dict[str, Any]:
    """Resolve a registry ID or verify a checkpoint directory without mutation."""
    try:
        # Use only the registry's read/verification functions. Do not register,
        # rewrite, or otherwise modify the checkpoint or registry.
        from training.model_registry import read_checkpoint_manifest, resolve_checkpoint
    except ImportError as exc:
        raise EvaluationError(
            "Strict benchmarking needs the local training.model_registry package; "
            "run the CLI from examples/fomo-ai-brain."
        ) from exc

    reference = Path(checkpoint).expanduser()
    try:
        if reference.is_dir():
            root = reference.resolve()
        else:
            root = resolve_checkpoint(checkpoint, registry_path, verify_files=True)
        manifest = read_checkpoint_manifest(root, verify_files=True)
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"Cannot verify checkpoint {checkpoint!r}: {exc}") from exc

    checkpoint_id = manifest.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise EvaluationError("Verified checkpoint manifest has no checkpoint_id")
    return {
        "checkpoint_id": checkpoint_id,
        "manifest_sha256": sha256_file(root / ".fomo-checkpoint.json"),
        "manifest_status": manifest.get("status"),
        "training_method": manifest.get("method"),
        "_checkpoint_root": str(root),
    }


def require_complete_external_candidates(
    cases: list[Mapping[str, Any]],
    candidates: Mapping[str, Any],
) -> None:
    """Strict mode forbids embedded predictions and requires exact ID coverage."""
    if not cases:
        raise EvaluationError("Strict benchmarking requires at least one labeled case")
    embedded = sorted(case["id"] for case in cases if "candidate" in case)
    if embedded:
        raise EvaluationError(
            "Strict benchmarking rejects candidates embedded in the cases file: "
            f"{embedded!r}; provide them only in --candidate JSONL."
        )
    case_ids = {case["id"] for case in cases}
    candidate_ids = set(candidates)
    missing = sorted(case_ids - candidate_ids)
    extra = sorted(candidate_ids - case_ids)
    if missing or extra:
        raise EvaluationError(
            "Strict benchmarking requires exactly one external candidate per case ID; "
            f"missing={missing!r}, extra={extra!r}."
        )


def benchmark_binding(
    *,
    checkpoint: str,
    registry_path: str | Path,
    cases_path: str | Path,
    candidate_path: str | Path,
    generation_settings_id: str,
    case_count: int,
    report_path: str | Path,
) -> dict[str, Any]:
    if not isinstance(generation_settings_id, str) or not generation_settings_id.strip():
        raise EvaluationError("generation settings identifier must be a non-empty string")
    verified_checkpoint = checkpoint_identity(checkpoint, registry_path)
    checkpoint_root = Path(verified_checkpoint.pop("_checkpoint_root"))
    report_file = Path(report_path).expanduser().resolve()
    if report_file == checkpoint_root or checkpoint_root in report_file.parents:
        raise EvaluationError("Report output must not be written inside the checkpoint directory")
    return {
        "report_type": "strict_checkpoint_benchmark",
        "case_file": Path(cases_path).name,
        "case_file_sha256": sha256_file(cases_path),
        "case_count": case_count,
        "candidate_file": Path(candidate_path).name,
        "candidate_file_sha256": sha256_file(candidate_path),
        "candidate_count": case_count,
        "candidate_coverage": "complete",
        "generation_settings_id": generation_settings_id.strip(),
        "checkpoint": verified_checkpoint,
        "candidate_provenance": "operator-attested external JSONL; not independently verified",
        "responses_generated_by_evaluator": False,
        "limitations": [
            "The evaluator never loads the checkpoint or generates responses.",
            "Checkpoint identity and artifact hashes are verified from the training registry manifest.",
            "The response-to-checkpoint association and generation settings identifier are self-attested, not cryptographically proven.",
            "Scores remain deterministic checks against the supplied case labels, not a semantic quality or safety guarantee.",
        ],
    }


def ensure_report_does_not_overwrite_inputs(
    report_path: str | Path | None,
    input_paths: list[str | Path | None],
) -> None:
    if not report_path:
        return
    output = Path(report_path).expanduser().resolve()
    for source in input_paths:
        if source and output == Path(source).expanduser().resolve():
            raise EvaluationError(f"Report output must not overwrite input file {source}")