"""Attach an explicitly attested dataset manifest to raw JSONL records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from training.common import DatasetError, canonical_hash, normalize_text, read_jsonl, validate_provenance, write_jsonl
from training.datasets.validate import validate_dpo_record, validate_sft_record


def load_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read dataset manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DatasetError("dataset manifest must be a JSON object")
    for key in ("dataset_id", "source", "license"):
        manifest[key] = normalize_text(manifest.get(key), f"manifest.{key}")
    if manifest.get("rights_attested") is not True:
        raise DatasetError(
            "manifest.rights_attested must be true; confirm you have the rights to use this data"
        )
    return manifest


def ingest(
    source: str | Path,
    destination: str | Path,
    manifest: Mapping[str, Any],
    *,
    confirm_rights: bool,
    kind: str = "sft",
) -> int:
    if not confirm_rights:
        raise DatasetError("Refusing ingestion without explicit --confirm-rights")
    if kind not in {"sft", "dpo"}:
        raise DatasetError("kind must be 'sft' or 'dpo'")
    meta = dict(manifest)
    for key in ("dataset_id", "source", "license"):
        meta[key] = normalize_text(meta.get(key), f"manifest.{key}")
    if meta.get("rights_attested") is not True:
        raise DatasetError("manifest.rights_attested must be true")
    count = 0
    def records():
        nonlocal count
        for row in read_jsonl(source):
            row_provenance = row.pop("provenance", {})
            if not isinstance(row_provenance, dict):
                raise DatasetError("record provenance must be an object")
            for field in ("dataset_id", "source", "license"):
                claimed = row_provenance.get(field)
                if claimed is not None and claimed != meta[field]:
                    raise DatasetError(
                        f"record provenance.{field} conflicts with the manifest"
                    )
            provenance = dict(meta)
            provenance.update({
                key: value for key, value in row_provenance.items()
                if key not in {"dataset_id", "source", "license", "rights_attested"}
            })
            provenance["rights_attested"] = True
            row["provenance"] = provenance
            row.setdefault("record_id", canonical_hash(row)[:24])
            validate_provenance(row)
            if kind == "sft":
                validate_sft_record(row)
            else:
                # Preserve whatever preference source the manifest records.
                validate_dpo_record(row, human_preferences=False)
            count += 1
            yield row
    written = write_jsonl(destination, records())
    if not written:
        raise DatasetError("Input dataset contains no records")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--kind", choices=("sft", "dpo"), default="sft")
    parser.add_argument("--manifest", required=True, help="JSON file with dataset_id/source/license/rights_attested")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--confirm-rights", action="store_true", required=True,
        help="Confirm you have documented permission to train on every input record",
    )
    args = parser.parse_args()
    count = ingest(
        args.input, args.output, load_manifest(args.manifest),
        confirm_rights=args.confirm_rights, kind=args.kind,
    )
    print(f"Ingested {count} records to {args.output}")
    print("This step records your attestation; it does not verify ownership, consent, or license terms.")


if __name__ == "__main__":
    main()