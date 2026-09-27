"""Shared, dependency-light JSONL and dataset validation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class DatasetError(ValueError):
    """Raised when input data is malformed or lacks required provenance."""


def normalize_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DatasetError(f"{field} must be a string")
    value = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    value = value.strip()
    if not value:
        raise DatasetError(f"{field} must not be empty")
    return value


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise DatasetError(f"{source}:{line_number}: each JSONL row must be an object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Atomically replace a UTF-8 JSONL file; return the number of records."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        return count
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance(
    row: Mapping[str, Any], *, human_preferences: bool = False
) -> dict[str, Any]:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        raise DatasetError("provenance object is required on every record")
    required = ("dataset_id", "source", "license")
    cleaned = dict(provenance)
    for key in required:
        cleaned[key] = normalize_text(cleaned.get(key), f"provenance.{key}")
    if provenance.get("rights_attested") is not True:
        raise DatasetError("provenance.rights_attested must be true; attest rights before training")
    cleaned["rights_attested"] = True
    if human_preferences:
        if provenance.get("preference_source") != "human":
            raise DatasetError(
                "DPO requires human preference labels by default; "
                "set preference_source='human' or explicitly opt out"
            )
        count = provenance.get("annotator_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise DatasetError("human DPO records require provenance.annotator_count >= 1")
        cleaned["annotator_count"] = count
        cleaned["preference_source"] = "human"
        cleaned["annotation_protocol"] = normalize_text(
            provenance.get("annotation_protocol"), "provenance.annotation_protocol"
        )
    return cleaned


def example_payload(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
    if kind == "sft":
        keys = ("messages",)
    elif kind == "dpo":
        keys = ("prompt", "chosen", "rejected")
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")
    try:
        return {key: row[key] for key in keys}
    except KeyError as exc:
        raise DatasetError(f"record is missing {exc.args[0]!r}") from exc


_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def find_possible_secrets(value: Any) -> list[str]:
    """Best-effort triage only; not a substitute for a privacy/security review."""
    text = json.dumps(value, ensure_ascii=False)
    labels = []
    if _SECRET_PATTERNS[0].search(text):
        labels.append("OpenAI-style API key")
    if _SECRET_PATTERNS[1].search(text):
        labels.append("GitHub token")
    if _SECRET_PATTERNS[2].search(text):
        labels.append("AWS access key")
    return labels