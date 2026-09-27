"""Normalize text and remove exact duplicate examples without inventing labels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

from training.common import DatasetError, canonical_hash, normalize_text, read_jsonl, write_jsonl
from training.datasets.validate import validate_dpo_record, validate_sft_record


def _normalized(row: Mapping[str, Any], kind: str, human_preferences: bool) -> dict[str, Any]:
    if kind == "sft":
        payload = validate_sft_record(row)
    else:
        payload = validate_dpo_record(row, human_preferences=human_preferences)
    provenance = dict(row["provenance"])
    metadata = {key: value for key, value in row.items()
                if key not in {"messages", "prompt", "chosen", "rejected", "instruction", "response", "output", "system", "provenance"}}
    normalized = {**metadata, **payload, "provenance": provenance}
    if "record_id" not in normalized:
        normalized["record_id"] = canonical_hash(payload)[:24]
    return normalized


def clean_records(
    records: list[dict[str, Any]], kind: str, *, human_preferences: bool = True
) -> tuple[list[dict[str, Any]], int]:
    if kind not in {"sft", "dpo"}:
        raise DatasetError("kind must be 'sft' or 'dpo'")
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in records:
        normalized = _normalized(row, kind, human_preferences)
        fingerprint = canonical_hash({
            key: normalized[key]
            for key in (("messages",) if kind == "sft" else ("prompt", "chosen", "rejected"))
        })
        if fingerprint not in unique:
            unique[fingerprint] = normalized
            continue
        duplicates += 1
        prior = unique[fingerprint]
        origins = prior.setdefault("duplicate_origins", [])
        origin = normalized["provenance"]
        if origin != prior["provenance"] and origin not in origins:
            origins.append(origin)
    return list(unique.values()), duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("sft", "dpo"), required=True)
    parser.add_argument("--allow-nonhuman-preferences", action="store_true")
    args = parser.parse_args()
    rows, duplicates = clean_records(
        list(read_jsonl(args.input)), args.kind,
        human_preferences=not args.allow_nonhuman_preferences,
    )
    if not rows:
        raise DatasetError("Input dataset contains no records")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} unique records; removed {duplicates} exact duplicate(s)")
    print("Normalization is not PII removal or a safety review.")


if __name__ == "__main__":
    main()