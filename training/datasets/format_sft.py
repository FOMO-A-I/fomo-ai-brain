"""Convert instruction/response JSONL into TRL conversational SFT JSONL."""

from __future__ import annotations

import argparse

from training.common import DatasetError, canonical_hash, read_jsonl, write_jsonl
from training.datasets.validate import validate_sft_record


def format_record(row: dict) -> dict:
    payload = validate_sft_record(row)
    return {
        "record_id": row.get("record_id") or canonical_hash(payload)[:24],
        "messages": payload["messages"],
        "provenance": dict(row["provenance"]),
    }


def format_file(source: str, destination: str) -> int:
    rows = (format_record(row) for row in read_jsonl(source))
    count = write_jsonl(destination, rows)
    if count == 0:
        raise DatasetError("Input dataset contains no records")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(f"Formatted {format_file(args.input, args.output)} SFT records")


if __name__ == "__main__":
    main()