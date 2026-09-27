"""Deterministically partition deduplicated records into train/evaluation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from training.common import DatasetError, canonical_hash, file_sha256, read_jsonl, write_jsonl
from training.datasets.validate import validate_dpo_record, validate_sft_record


def split_records(
    rows: list[dict], *, validation_fraction: float = 0.02, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_fraction < 1:
        raise DatasetError("validation_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise DatasetError("At least two records are needed for train/evaluation split")
    by_key: dict[str, dict] = {}
    content_keys: set[str] = set()
    for row in rows:
        payload = validate_sft_record(row) if "messages" in row else validate_dpo_record(
            row, human_preferences=False
        )
        content_key = canonical_hash(payload)
        if content_key in content_keys:
            raise DatasetError("Duplicate example content; clean duplicates before splitting")
        content_keys.add(content_key)
        key = row.get("record_id")
        if not isinstance(key, str) or not key:
            key = hashlib.sha256(
                json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
        if key in by_key:
            raise DatasetError(f"Duplicate record_id {key!r}; clean duplicates before splitting")
        by_key[key] = row
    ordered = sorted(
        by_key.items(),
        key=lambda pair: hashlib.sha256(f"{seed}:{pair[0]}".encode("utf-8")).digest(),
    )
    evaluation_count = max(1, round(len(ordered) * validation_fraction))
    evaluation_count = min(evaluation_count, len(ordered) - 1)
    evaluation_ids = {key for key, _ in ordered[:evaluation_count]}
    train = [row for key, row in ordered if key not in evaluation_ids]
    evaluation = [row for key, row in ordered if key in evaluation_ids]
    return train, evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise DatasetError(f"Refusing to overwrite non-empty output directory: {out}")
    rows = list(read_jsonl(args.input))
    train, evaluation = split_records(
        rows, validation_fraction=args.validation_fraction, seed=args.seed
    )
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "validation.jsonl", evaluation)
    manifest = {
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_records": len(train),
        "validation_records": len(evaluation),
        "input_sha256": file_sha256(args.input),
        "train_sha256": file_sha256(out / "train.jsonl"),
        "validation_sha256": file_sha256(out / "validation.jsonl"),
        "note": "Deterministic hash split; review for source, person, and task leakage.",
    }
    (out / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Split {len(train)} train / {len(evaluation)} validation records into {out}")


if __name__ == "__main__":
    main()