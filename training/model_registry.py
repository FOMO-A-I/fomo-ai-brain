"""Local registry for verified, completed fine-tuning checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from training.common import DatasetError, canonical_hash, file_sha256


MANIFEST_NAME = ".fomo-checkpoint.json"
REGISTRY_VERSION = 1


def _checkpoint_files(directory: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DatasetError(f"Checkpoint contains a symlink; refusing to register: {path}")
        if path.is_file() and path.name != MANIFEST_NAME:
            files[path.relative_to(directory).as_posix()] = file_sha256(path)
    if not files:
        raise DatasetError(f"Checkpoint directory contains no model/tokenizer files: {directory}")
    return files


def seal_checkpoint(directory: str | Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Record the completed training artifact and hash every file in it."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise DatasetError(f"Checkpoint output directory does not exist: {root}")
    if (root / MANIFEST_NAME).exists():
        raise DatasetError(f"Checkpoint already has a completion manifest: {root / MANIFEST_NAME}")
    required = (
        "method", "base_model", "base_model_license", "dataset", "examples_trained",
    )
    missing = [key for key in required if metadata.get(key) in (None, "")]
    if missing:
        raise DatasetError(f"Checkpoint metadata missing: {', '.join(missing)}")
    if metadata["method"] not in {"sft", "dpo"}:
        raise DatasetError("Checkpoint method must be 'sft' or 'dpo'")
    if metadata.get("base_model_terms_reviewed") is not True:
        raise DatasetError("Checkpoint requires an explicit reviewed base-model terms attestation")
    dataset_sha = metadata.get("dataset", {}).get("sha256") if isinstance(metadata.get("dataset"), dict) else None
    if not isinstance(dataset_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", dataset_sha):
        raise DatasetError("Checkpoint metadata must include a hashed training dataset summary")
    if not isinstance(metadata.get("base_model_license"), str) or not metadata["base_model_license"].strip():
        raise DatasetError("Checkpoint must record the base model license")
    if isinstance(metadata["examples_trained"], bool) or not isinstance(metadata["examples_trained"], int):
        raise DatasetError("examples_trained must be an integer")
    if metadata["examples_trained"] < 1:
        raise DatasetError("examples_trained must be positive")
    body = {
        "format_version": 1,
        "status": "trained",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **dict(metadata),
        "files": _checkpoint_files(root),
    }
    body["checkpoint_id"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:24]
    destination = root / MANIFEST_NAME
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=".checkpoint.", suffix=".tmp", delete=False
    ) as stream:
        temporary = stream.name
        json.dump(body, stream, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return body


def read_checkpoint_manifest(directory: str | Path, *, verify_files: bool = True) -> dict[str, Any]:
    root = Path(directory).resolve()
    manifest_path = root / MANIFEST_NAME
    if not root.is_dir() or not manifest_path.is_file():
        raise DatasetError(
            f"No completed training manifest at {manifest_path}; "
            "base/pretrained model folders are not registered FOMO checkpoints"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Invalid checkpoint manifest: {exc}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != 1
        or manifest.get("status") != "trained"
    ):
        raise DatasetError("Checkpoint manifest is not a completed supported training run")
    if (
        manifest.get("method") not in {"sft", "dpo"}
        or not isinstance(manifest.get("base_model"), str)
        or not manifest.get("base_model")
        or not isinstance(manifest.get("base_model_license"), str)
        or not manifest.get("base_model_license")
        or manifest.get("base_model_terms_reviewed") is not True
        or isinstance(manifest.get("examples_trained"), bool)
        or not isinstance(manifest.get("examples_trained"), int)
        or manifest["examples_trained"] < 1
    ):
        raise DatasetError("Checkpoint manifest lacks required training/license attestations")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", dataset["sha256"]
    ):
        raise DatasetError("Checkpoint manifest has no valid training dataset SHA-256")
    claimed_id = manifest.get("checkpoint_id")
    body = {key: value for key, value in manifest.items() if key != "checkpoint_id"}
    actual_id = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:24]
    if claimed_id != actual_id:
        raise DatasetError("Checkpoint manifest identity does not match its recorded metadata")
    if verify_files:
        expected = manifest.get("files")
        if not isinstance(expected, dict) or not expected:
            raise DatasetError("Checkpoint manifest has no artifact file hashes")
        actual = _checkpoint_files(root)
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            changed = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
            added = sorted(set(actual) - set(expected))
            raise DatasetError(
                f"Checkpoint integrity verification failed; missing={missing}, "
                f"changed={changed}, added={added}"
            )
    return manifest


def _read_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format_version": REGISTRY_VERSION, "checkpoints": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read model registry {path}: {exc}") from exc
    if (
        not isinstance(value, dict)
        or value.get("format_version") != REGISTRY_VERSION
        or not isinstance(value.get("checkpoints"), list)
    ):
        raise DatasetError(f"Unsupported or malformed model registry: {path}")
    return value


def register_checkpoint(
    checkpoint_dir: str | Path, registry_path: str | Path
) -> dict[str, Any]:
    root = Path(checkpoint_dir).resolve()
    manifest = read_checkpoint_manifest(root, verify_files=True)
    destination = Path(registry_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    deadline = time.monotonic() + 10
    lock_fd: int | None = None
    while lock_fd is None:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise DatasetError(
                    f"Could not acquire registry lock {lock_path}; "
                    "check for a stale lock and remove it only after confirming no writer is active"
                )
            time.sleep(0.05)
    try:
        registry = _read_registry(destination)
        if any(item.get("checkpoint_id") == manifest["checkpoint_id"] for item in registry["checkpoints"]):
            raise DatasetError(f"Checkpoint {manifest['checkpoint_id']} is already registered")
        entry = {
            "checkpoint_id": manifest["checkpoint_id"],
            "method": manifest["method"],
            "base_model": manifest["base_model"],
            "created_at": manifest["created_at"],
            "examples_trained": manifest["examples_trained"],
            "checkpoint_path": str(root),
            "manifest_sha256": file_sha256(root / MANIFEST_NAME),
        }
        registry["checkpoints"].append(entry)
        registry["checkpoints"].sort(key=lambda item: item["created_at"])
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            json.dump(registry, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return entry
    finally:
        os.close(lock_fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def list_checkpoints(registry_path: str | Path) -> list[dict[str, Any]]:
    return _read_registry(Path(registry_path))["checkpoints"]


def resolve_checkpoint(
    checkpoint_id: str, registry_path: str | Path, *, verify_files: bool = True
) -> Path:
    for entry in list_checkpoints(registry_path):
        if entry.get("checkpoint_id") != checkpoint_id:
            continue
        root = Path(entry["checkpoint_path"]).resolve()
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.is_file() or file_sha256(manifest_path) != entry.get("manifest_sha256"):
            raise DatasetError("Registry entry and checkpoint manifest do not match")
        manifest = read_checkpoint_manifest(root, verify_files=verify_files)
        if manifest["checkpoint_id"] != checkpoint_id:
            raise DatasetError("Registry checkpoint ID mismatch")
        return root
    raise DatasetError(f"Checkpoint ID is not registered: {checkpoint_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("add", "list"))
    parser.add_argument("--checkpoint", help="Completed checkpoint directory to register")
    parser.add_argument("--registry", default="training/checkpoints/registry.json")
    args = parser.parse_args()
    if args.action == "add":
        if not args.checkpoint:
            parser.error("--checkpoint is required for add")
        entry = register_checkpoint(args.checkpoint, args.registry)
        print(f"Registered completed checkpoint {entry['checkpoint_id']}")
        return
    entries = list_checkpoints(args.registry)
    if not entries:
        print("No completed training checkpoints are registered.")
        return
    for entry in entries:
        print(
            f"{entry['checkpoint_id']}  {entry['method']}  "
            f"{entry['examples_trained']} examples  {entry['checkpoint_path']}"
        )


if __name__ == "__main__":
    main()