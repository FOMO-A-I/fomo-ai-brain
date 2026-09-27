"""Provenance helpers for strict, externally supplied checkpoint benchmarks."""

from __future__ import annotations

import hashlib
import json
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


def generation_settings_identifier(settings: Mapping[str, Any]) -> str:
    """Stable ID over the complete generation configuration."""
    serialized = json.dumps(
        settings,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:16]
    prefix = (
        "local-transformers-v1"
        if settings.get("backend") == "LocalTransformersBackend"
        else "injected-backend-v1"
    )
    return f"{prefix}-{digest}"


def validate_candidate_provenance(
    records: Mapping[str, Mapping[str, Any]],
    cases: list[Mapping[str, Any]],
    *,
    case_file_sha256: str,
    checkpoint_id: str,
    checkpoint_manifest_sha256: str,
    generation_settings_id: str,
    operator_attested: bool,
) -> dict[str, Any]:
    """Verify generated-response bindings or require an explicit external attestation."""
    case_ids = {case["id"] for case in cases}
    if set(records) != case_ids:
        raise EvaluationError("Candidate provenance validation requires exact case ID coverage")

    marked_records = [
        record for record in records.values()
        if any(key in record for key in (
            "provenance",
            "generation_settings",
            "generation_settings_id",
            "case_file_sha256",
            "prompt_sha256",
            "checkpoint_id",
            "checkpoint_manifest_sha256",
        ))
    ]
    if not marked_records:
        if not operator_attested:
            raise EvaluationError(
                "Candidate file has no generated provenance; pass "
                "--operator-attested-candidate to accept an external candidate with "
                "explicit self-attested limitations"
            )
        return {
            "mode": "operator_attested_external",
            "candidate_provenance": "operator-attested external JSONL; not independently verified",
        }
    if len(marked_records) != len(records):
        raise EvaluationError(
            "Candidate file mixes generated-provenance records with unmarked records"
        )
    if operator_attested:
        raise EvaluationError(
            "--operator-attested-candidate cannot be used to bypass generated provenance validation"
        )

    cases_by_id = {case["id"]: case for case in cases}
    common_settings: Mapping[str, Any] | None = None
    for case_id, record in records.items():
        if set(record) != {"id", "response", "provenance"}:
            raise EvaluationError(
                f"Generated candidate {case_id!r} has unexpected or incomplete record fields"
            )
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise EvaluationError(f"Candidate {case_id!r} has malformed generated provenance")
        expected_keys = {
            "schema",
            "case_file_sha256",
            "prompt_sha256",
            "checkpoint_id",
            "checkpoint_manifest_sha256",
            "generation_settings_id",
            "generation_settings",
        }
        if set(provenance) != expected_keys:
            raise EvaluationError(
                f"Candidate {case_id!r} has incomplete or unknown generated provenance fields"
            )
        if provenance["schema"] != "fomo-generated-candidate-v1":
            raise EvaluationError(f"Candidate {case_id!r} has an unsupported provenance schema")
        if provenance["case_file_sha256"] != case_file_sha256:
            raise EvaluationError(
                f"Candidate {case_id!r} is bound to a different case file SHA-256"
            )
        prompt_digest = hashlib.sha256(
            cases_by_id[case_id]["prompt"].encode("utf-8")
        ).hexdigest()
        if provenance["prompt_sha256"] != prompt_digest:
            raise EvaluationError(
                f"Candidate {case_id!r} is bound to a different prompt SHA-256"
            )
        if provenance["checkpoint_id"] != checkpoint_id:
            raise EvaluationError(f"Candidate {case_id!r} is bound to a different checkpoint ID")
        if provenance["checkpoint_manifest_sha256"] != checkpoint_manifest_sha256:
            raise EvaluationError(
                f"Candidate {case_id!r} is bound to a different checkpoint manifest SHA-256"
            )
        if provenance["generation_settings_id"] != generation_settings_id:
            raise EvaluationError(
                f"Candidate {case_id!r} generation settings ID does not match "
                "--generation-settings-id"
            )
        settings = provenance["generation_settings"]
        if not isinstance(settings, Mapping):
            raise EvaluationError(f"Candidate {case_id!r} has malformed generation settings")
        required_settings = {
            "backend",
            "temperature",
            "do_sample",
            "max_new_tokens",
            "device_map",
            "deterministic",
            "seed",
        }
        if set(settings) != required_settings:
            raise EvaluationError(
                f"Candidate {case_id!r} has incomplete or unknown generation settings"
            )
        if (
            settings["backend"] != "LocalTransformersBackend"
            or isinstance(settings["temperature"], bool)
            or not isinstance(settings["temperature"], (int, float))
            or settings["temperature"] != 0
            or settings["do_sample"] is not False
            or isinstance(settings["max_new_tokens"], bool)
            or not isinstance(settings["max_new_tokens"], int)
            or not 1 <= settings["max_new_tokens"] <= 8192
            or not isinstance(settings["device_map"], str)
            or not settings["device_map"].strip()
            or settings["deterministic"] is not True
            or settings["seed"] is not None
        ):
            raise EvaluationError(
                f"Candidate {case_id!r} declares unsupported or non-deterministic generation settings"
            )
        if generation_settings_identifier(settings) != generation_settings_id:
            raise EvaluationError(
                f"Candidate {case_id!r} generation settings do not match their recorded ID"
            )
        if common_settings is None:
            common_settings = settings
        elif dict(settings) != dict(common_settings):
            raise EvaluationError("Candidate records declare inconsistent generation settings")
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            raise EvaluationError(
                f"Generated candidate {case_id!r} needs a non-empty response string"
            )

    return {
        "mode": "generated_local_transformers",
        "candidate_provenance": (
            "generated-response metadata matched to current case/prompt hashes, "
            "verified checkpoint manifest, and generation settings; inference itself "
            "remains self-attested"
        ),
        "generation_settings": dict(common_settings or {}),
        "generation_settings_id": generation_settings_id,
    }


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