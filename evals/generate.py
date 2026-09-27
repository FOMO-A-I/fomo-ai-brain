"""Generate bounded candidate JSONL from a verified local FOMO checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .benchmark import checkpoint_identity, generation_settings_identifier, sha256_file
from .runner import EvaluationError, load_cases

MAX_CASES = 10_000
MAX_CASES_FILE_BYTES = 100 * 1024 * 1024
MAX_PROMPT_CHARS = 100_000


class CompletionBackend(Protocol):
    def complete(self, messages: Sequence[Mapping[str, str]]) -> str: ...


def generate_candidates(
    *,
    cases_path: str | Path,
    output_path: str | Path,
    checkpoint: str,
    registry_path: str | Path = "training/checkpoints/registry.json",
    max_new_tokens: int = 1024,
    device_map: str = "auto",
    overwrite: bool = False,
    backend: CompletionBackend | None = None,
) -> dict[str, Any]:
    """Generate one response per case, atomically publishing only on success.

    ``backend`` is an injection point for unit tests. Production CLI use always
    constructs LocalTransformersBackend after verifying the checkpoint.
    """
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise EvaluationError("max_new_tokens must be an integer")
    if not 1 <= max_new_tokens <= 8192:
        raise EvaluationError("max_new_tokens must be between 1 and 8192")
    if not isinstance(device_map, str) or not device_map.strip():
        raise EvaluationError("device_map must not be empty")

    case_file = Path(cases_path).expanduser()
    output_file = Path(output_path).expanduser()
    checkpoint_info = checkpoint_identity(checkpoint, registry_path)
    checkpoint_root = Path(checkpoint_info.pop("_checkpoint_root"))
    output_resolved = output_file.resolve()
    if output_resolved == case_file.resolve():
        raise EvaluationError("Candidate output must not overwrite the cases file")
    if output_resolved == checkpoint_root or checkpoint_root in output_resolved.parents:
        raise EvaluationError("Candidate output must not be written inside the checkpoint directory")
    if output_file.exists() and not overwrite:
        raise EvaluationError(
            f"Candidate output already exists: {output_file}; pass --overwrite to replace it"
        )

    try:
        if case_file.stat().st_size > MAX_CASES_FILE_BYTES:
            raise EvaluationError(
                f"Cases file exceeds the {MAX_CASES_FILE_BYTES}-byte limit"
            )
    except OSError as exc:
        raise EvaluationError(f"Cannot inspect cases file {case_file}: {exc}") from exc
    cases_hash_before = sha256_file(case_file)
    cases = load_cases(case_file)
    if not cases:
        raise EvaluationError("Candidate generation requires at least one held-out case")
    if len(cases) > MAX_CASES:
        raise EvaluationError(f"Case count exceeds the limit of {MAX_CASES}")
    for case in cases:
        if "candidate" in case:
            raise EvaluationError(
                f"Case {case['id']!r} embeds a candidate; remove it before generating external responses"
            )
        if len(case["prompt"]) > MAX_PROMPT_CHARS:
            raise EvaluationError(
                f"Case {case['id']!r} prompt exceeds the {MAX_PROMPT_CHARS}-character limit"
            )
    if sha256_file(case_file) != cases_hash_before:
        raise EvaluationError("Cases file changed while it was being loaded; retry with a stable file")

    local_backend = backend is None
    if backend is None:
        try:
            from fomo.brain.model import LocalTransformersBackend
        except ImportError as exc:
            raise EvaluationError(
                "Cannot import LocalTransformersBackend; run from examples/fomo-ai-brain "
                "with the FOMO package available"
            ) from exc
        try:
            backend = LocalTransformersBackend(
                checkpoint_root,
                max_new_tokens=max_new_tokens,
                temperature=0,
                device_map=device_map,
            )
        except Exception as exc:
            raise EvaluationError(f"Cannot initialize local checkpoint backend: {exc}") from exc

    settings = {
        "backend": "LocalTransformersBackend" if local_backend else type(backend).__name__,
        "temperature": 0,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "device_map": device_map,
        "deterministic": local_backend,
        "seed": None,
    }
    settings_id = generation_settings_identifier(settings)
    records: list[str] = []
    for case in cases:
        try:
            response = backend.complete([
                {"role": "user", "content": case["prompt"]},
            ])
        except Exception as exc:
            raise EvaluationError(f"Model generation failed for case {case['id']!r}: {exc}") from exc
        if not isinstance(response, str) or not response.strip():
            raise EvaluationError(
                f"Model generation returned an empty or invalid response for case {case['id']!r}"
            )
        provenance = {
            "schema": "fomo-generated-candidate-v1",
            "case_file_sha256": cases_hash_before,
            "prompt_sha256": hashlib.sha256(case["prompt"].encode("utf-8")).hexdigest(),
            "checkpoint_id": checkpoint_info["checkpoint_id"],
            "checkpoint_manifest_sha256": checkpoint_info["manifest_sha256"],
            "generation_settings_id": settings_id,
            "generation_settings": settings,
        }
        records.append(json.dumps(
            {"id": case["id"], "response": response, "provenance": provenance},
            ensure_ascii=False,
            separators=(",", ":"),
        ))

    if sha256_file(case_file) != cases_hash_before:
        raise EvaluationError("Cases file changed during generation; refusing to publish candidates")
    if checkpoint_identity(str(checkpoint_root), registry_path)[
        "manifest_sha256"
    ] != checkpoint_info["manifest_sha256"]:
        raise EvaluationError("Checkpoint manifest changed during generation; refusing to publish candidates")

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EvaluationError(f"Cannot create candidate output directory {output_file.parent}: {exc}") from exc
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write("\n".join(records) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary_path, output_file)
        else:
            # Hard-link creation is atomic and fails rather than clobbering a
            # file created by another process after the initial existence check.
            os.link(temporary_path, output_file)
            temporary_path.unlink()
        temporary_path = None
    except FileExistsError as exc:
        raise EvaluationError(
            f"Candidate output already exists: {output_file}; pass --overwrite to replace it"
        ) from exc
    except OSError as exc:
        raise EvaluationError(f"Cannot atomically write candidate output {output_file}: {exc}") from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass

    return {
        "case_count": len(cases),
        "candidate_file_sha256": sha256_file(output_file),
        "generation_settings": settings,
        "generation_settings_id": settings_id,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.generate",
        description="Generate candidate responses using a verified local FOMO checkpoint.",
    )
    parser.add_argument("--cases", required=True, help="Held-out case definitions in JSONL format")
    parser.add_argument("--checkpoint", required=True, help="Verified local checkpoint directory or registry ID")
    parser.add_argument(
        "--registry",
        default="training/checkpoints/registry.json",
        help="Read-only checkpoint registry path when --checkpoint is an ID",
    )
    parser.add_argument("--output", required=True, help="Candidate response JSONL output path")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing candidate output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = generate_candidates(
            cases_path=args.cases,
            output_path=args.output,
            checkpoint=args.checkpoint,
            registry_path=args.registry,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            overwrite=args.overwrite,
        )
    except EvaluationError as exc:
        print(f"candidate generation error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Generated {result['case_count']} candidate responses | "
        f"SHA-256: {result['candidate_file_sha256']}"
    )
    print(f"Generation settings ID: {result['generation_settings_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())