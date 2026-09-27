"""Lazy optional-dependency loading and shared training preflight checks."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from training.common import (
    DatasetError, canonical_hash, example_payload, file_sha256,
    find_possible_secrets, read_jsonl,
)
from training.datasets.validate import validate_dpo_record, validate_sft_record


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading training YAML requires PyYAML; install requirements-training.txt"
        ) from exc
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DatasetError(f"Cannot read training config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise DatasetError(f"Training config must be a YAML mapping: {path}")
    return config


def validate_config(config: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - config.keys())
    if missing:
        raise DatasetError(f"Training config missing required keys: {', '.join(missing)}")
    integer_keys = {
        "seed", "max_length", "max_prompt_length", "per_device_train_batch_size",
        "per_device_eval_batch_size", "gradient_accumulation_steps", "logging_steps",
        "eval_steps", "save_steps", "save_total_limit", "lora_r", "lora_alpha",
    }
    for key, value in config.items():
        if key in integer_keys:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise DatasetError(f"Training config {key} must be a positive integer")
        if key == "num_train_epochs":
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DatasetError("Training config num_train_epochs must be positive")
        if key in {"learning_rate", "beta"} and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise DatasetError(f"Training config {key} must be positive")
        if key in {"lora_dropout", "warmup_ratio", "weight_decay"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DatasetError(f"Training config {key} must be numeric")
        if key == "lora_dropout" and not 0 <= value < 1:
            raise DatasetError("Training config lora_dropout must be in [0, 1)")
        if key == "warmup_ratio" and not 0 <= value < 1:
            raise DatasetError("Training config warmup_ratio must be in [0, 1)")
        if key == "weight_decay" and value < 0:
            raise DatasetError("Training config weight_decay must be non-negative")
        if key in {
            "bf16", "fp16", "gradient_checkpointing", "trust_remote_code",
            "load_in_4bit", "local_files_only",
        } and not isinstance(value, bool):
            raise DatasetError(f"Training config {key} must be true or false")
    if config.get("bf16") and config.get("fp16"):
        raise DatasetError("Select at most one of bf16 or fp16")
    if config.get("load_in_4bit") and not (config.get("bf16") or config.get("fp16")):
        raise DatasetError("4-bit QLoRA requires exactly one of bf16 or fp16 for compute dtype")


def apply_qlora_override(config: dict[str, Any], override: bool | None) -> dict[str, Any]:
    """Return effective config, allowing CLI opt-in/out to override YAML."""
    effective = dict(config)
    if override is not None:
        effective["load_in_4bit"] = override
    return effective


def make_4bit_quantization_config(config_class: Any, compute_dtype: Any) -> Any:
    """Construct the explicit NF4 + double-quant BitsAndBytes config."""
    return config_class(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def model_quantization_kwargs(
    enabled: bool, config_class: Any, compute_dtype: Any
) -> dict[str, Any]:
    """Return only the model kwargs needed for opted-in 4-bit loading."""
    if not enabled:
        return {}
    return {
        "quantization_config": make_4bit_quantization_config(config_class, compute_dtype),
        "device_map": "auto",
    }


def prepare_kbit_model(model: Any, prepare_function: Any, *, gradient_checkpointing: bool) -> Any:
    """Apply PEFT's supported k-bit preparation before attaching LoRA adapters."""
    return prepare_function(model, use_gradient_checkpointing=gradient_checkpointing)


def qlora_metadata(enabled: bool, compute_dtype: str | None = None) -> dict[str, Any]:
    """Stable training provenance for the base-model quantization mode."""
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": compute_dtype,
    }


def summarize_dataset(path: str | Path, kind: str, *, human_preferences: bool = True) -> dict[str, Any]:
    counts: dict[str, int] = {}
    provenance_summary: dict[str, dict[str, Any]] = {}
    preference_sources: dict[str, int] = {}
    annotation_protocols: dict[str, int] = {}
    examples = 0
    possible_secret_records = 0
    for row in read_jsonl(path):
        payload = validate_sft_record(row) if kind == "sft" else validate_dpo_record(
            row, human_preferences=human_preferences
        )
        provenance = row["provenance"]
        dataset_id = provenance["dataset_id"]
        counts[dataset_id] = counts.get(dataset_id, 0) + 1
        summary = provenance_summary.setdefault(dataset_id, {
            "source": provenance["source"],
            "license": provenance["license"],
            "rights_attested": True,
        })
        if summary["source"] != provenance["source"] or summary["license"] != provenance["license"]:
            raise DatasetError(
                f"Dataset ID {dataset_id!r} has conflicting source/license metadata"
            )
        if kind == "dpo":
            preference_source = str(provenance.get("preference_source", "unspecified"))
            preference_sources[preference_source] = preference_sources.get(preference_source, 0) + 1
            protocol = str(provenance.get("annotation_protocol", "unspecified"))
            annotation_protocols[protocol] = annotation_protocols.get(protocol, 0) + 1
        possible_secret_records += bool(find_possible_secrets(payload))
        examples += 1
    if examples == 0:
        raise DatasetError(f"Training dataset is empty: {path}")
    return {
        "filename": Path(path).name,
        "sha256": file_sha256(path),
        "examples": examples,
        "dataset_ids": counts,
        "provenance": provenance_summary,
        "possible_secret_records": possible_secret_records,
        "human_preferences_required": kind == "dpo" and human_preferences,
        **(
            {
                "preference_sources": preference_sources,
                "annotation_protocols": annotation_protocols,
            }
            if kind == "dpo" else {}
        ),
    }


def assert_disjoint_datasets(
    train_path: str | Path, validation_path: str | Path, kind: str
) -> None:
    """Reject both ID and content overlap, even when records were re-IDed."""
    train_ids: set[str] = set()
    train_content: set[str] = set()
    for row in read_jsonl(train_path):
        record_id = row.get("record_id")
        if isinstance(record_id, str) and record_id:
            if record_id in train_ids:
                raise DatasetError(f"Duplicate record_id {record_id!r} in training set")
            train_ids.add(record_id)
        fingerprint = canonical_hash(example_payload(row, kind))
        if fingerprint in train_content:
            raise DatasetError("Duplicate example content within training set; clean before training")
        train_content.add(fingerprint)
    eval_ids: set[str] = set()
    eval_content: set[str] = set()
    for row in read_jsonl(validation_path):
        record_id = row.get("record_id")
        if isinstance(record_id, str) and record_id:
            if record_id in eval_ids:
                raise DatasetError(f"Duplicate record_id {record_id!r} in validation set")
            eval_ids.add(record_id)
        fingerprint = canonical_hash(example_payload(row, kind))
        if fingerprint in train_content:
            raise DatasetError(
                "Training and validation sets contain duplicate example content; "
                "combine, deduplicate, then split before training"
            )
        if fingerprint in eval_content:
            raise DatasetError("Duplicate example content within validation set; clean before training")
        eval_content.add(fingerprint)
    overlap = train_ids & eval_ids
    if overlap:
        raise DatasetError(
            f"Training and validation sets share {len(overlap)} record_id(s); "
            "create a clean deterministic split before training"
        )


def require_training_dependencies(*, qlora: bool = False) -> dict[str, Any]:
    names = ("torch", "transformers", "datasets", "peft", "trl", "accelerate")
    modules = {}
    missing = []
    for name in names:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise RuntimeError(
            "Training extras are missing: " + ", ".join(missing)
            + ". Install requirements-training.txt in the GPU environment."
        )
    if qlora:
        try:
            modules["bitsandbytes"] = importlib.import_module("bitsandbytes")
        except ImportError as exc:
            raise RuntimeError(
                "4-bit QLoRA requires bitsandbytes. Install requirements-qlora.txt "
                "in the CUDA training environment; no training was started."
            ) from exc
    return modules


def require_cuda(torch: Any, *, bf16: bool, fp16: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Fine-tuning requires a CUDA GPU in this pipeline; no training was started. "
            "Use --dry-run to validate data without GPU, or run on a configured GPU machine."
        )
    if bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Config requests bf16, but this GPU does not support bf16.")


def load_json_dataset(
    path: str | Path, kind: str, datasets_module: Any, *, human_preferences: bool = True
) -> Any:
    """Validate and normalize JSONL rows, then exclude provenance from model inputs."""
    if kind not in {"sft", "dpo"}:
        raise DatasetError("kind must be 'sft' or 'dpo'")
    columns = ["messages"] if kind == "sft" else ["prompt", "chosen", "rejected"]
    model_rows = []
    for row in read_jsonl(path):
        # Repeat record validation at the loader boundary so the training
        # representation is derived from validated content, never raw fields.
        clean = validate_sft_record(row) if kind == "sft" else validate_dpo_record(
            row, human_preferences=human_preferences
        )
        model_rows.append({column: clean[column] for column in columns})
    if not model_rows:
        raise DatasetError(f"Training dataset is empty: {path}")
    dataset_class = getattr(datasets_module, "Dataset", None)
    if dataset_class is None or not callable(getattr(dataset_class, "from_list", None)):
        raise RuntimeError(
            "The installed datasets package must provide Dataset.from_list to normalize "
            "validated training records."
        )
    return dataset_class.from_list(model_rows)


def supported_kwargs(callable_object: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Filter config arguments for small upstream Transformers/TRL API changes."""
    import inspect

    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError):
        return values
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def checkpoint_metadata(
    *, method: str, base_model: str, base_model_revision: str | None,
    base_model_license: str, data: dict[str, Any], eval_data: dict[str, Any],
    examples_trained: int, config: dict[str, Any], extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "method": method,
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "base_model_license": base_model_license,
        "base_model_terms_reviewed": True,
        "dataset": data,
        "evaluation_dataset": eval_data,
        "examples_trained": examples_trained,
        "config": config,
        **(extra or {}),
    }