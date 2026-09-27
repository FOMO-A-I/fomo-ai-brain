"""Run LoRA supervised fine-tuning from validated, provenance-bearing JSONL.

Example (on a machine with a CUDA GPU and installed training extras):
  python -m training.sft.train --train data/train.jsonl \
      --validation data/validation.jsonl --output-dir checkpoints/fomo-sft-001 \
      --base-model org/model --base-model-license Apache-2.0 \
      --reviewed-base-model-terms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import DatasetError
from training.datasets.validate import validate_file
from training.model_registry import seal_checkpoint
from training.train_utils import (
    checkpoint_metadata,
    apply_qlora_override,
    load_json_dataset,
    load_yaml_config,
    assert_disjoint_datasets,
    require_cuda,
    require_training_dependencies,
    model_quantization_kwargs,
    prepare_kbit_model,
    qlora_metadata,
    summarize_dataset,
    supported_kwargs,
    validate_config,
)


_REQUIRED_CONFIG = {
    "seed", "max_length", "num_train_epochs", "learning_rate",
    "per_device_train_batch_size", "per_device_eval_batch_size",
    "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout",
    "bf16", "fp16", "gradient_checkpointing", "trust_remote_code",
    "load_in_4bit", "local_files_only",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Formatted training JSONL")
    parser.add_argument("--validation", required=True, help="Held-out evaluation JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True, help="Licensed Hugging Face model ID or local model path")
    parser.add_argument("--base-model-license", required=True, help="License identifier/source recorded in checkpoint")
    parser.add_argument(
        "--reviewed-base-model-terms", action="store_true", required=True,
        help="Confirm you reviewed base model license/terms for this use",
    )
    parser.add_argument("--base-model-revision", help="Pinned model revision/commit")
    qlora = parser.add_mutually_exclusive_group()
    qlora.add_argument("--qlora", dest="qlora", action="store_true", help="Opt into 4-bit NF4 QLoRA")
    qlora.add_argument("--no-qlora", dest="qlora", action="store_false", help="Use ordinary LoRA")
    parser.set_defaults(qlora=None)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument(
        "--allow-possible-secrets", action="store_true",
        help="Explicitly continue despite automated API-key pattern warnings (not recommended)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/config without importing GPU libraries")
    return parser.parse_args(argv)


def _preflight(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_config(config, _REQUIRED_CONFIG)
    if not args.base_model_license.strip():
        raise DatasetError("--base-model-license must not be empty")
    if Path(args.train).resolve() == Path(args.validation).resolve():
        raise DatasetError("Training and validation files must be different")
    train_data = summarize_dataset(args.train, "sft")
    validation_data = summarize_dataset(args.validation, "sft")
    flagged = train_data["possible_secret_records"] + validation_data["possible_secret_records"]
    if flagged and not args.allow_possible_secrets:
        raise DatasetError(
            f"Possible credential patterns found in {flagged} record(s). Review/redact the data; "
            "to override the heuristic, pass --allow-possible-secrets explicitly."
        )
    validate_file(args.train, "sft")
    validate_file(args.validation, "sft")
    assert_disjoint_datasets(args.train, args.validation, "sft")
    output = Path(args.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise DatasetError(f"Refusing to overwrite non-empty output directory: {output}")
    return train_data, validation_data


def run(argv: list[str] | None = None) -> dict[str, Any] | None:
    args = _parse_args(argv)
    config = apply_qlora_override(load_yaml_config(args.config), args.qlora)
    train_summary, validation_summary = _preflight(args, config)
    if args.dry_run:
        print(
            f"Dry run passed: {train_summary['examples']} training / "
            f"{validation_summary['examples']} validation records. "
            "No model was loaded and no checkpoint was created."
        )
        return None

    modules = require_training_dependencies(qlora=bool(config["load_in_4bit"]))
    torch = modules["torch"]
    require_cuda(torch, bf16=bool(config["bf16"]), fp16=bool(config["fp16"]))
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(int(config["seed"]))
    model_kwargs: dict[str, Any] = {
        "revision": args.base_model_revision,
        "trust_remote_code": bool(config["trust_remote_code"]),
        "local_files_only": bool(config["local_files_only"]),
        "torch_dtype": torch.bfloat16 if config["bf16"] else (
            torch.float16 if config["fp16"] else "auto"
        ),
    }
    if config["load_in_4bit"]:
        compute_dtype = torch.bfloat16 if config["bf16"] else torch.float16
        model_kwargs.update(model_quantization_kwargs(True, BitsAndBytesConfig, compute_dtype))
    if args.base_model_revision is None:
        model_kwargs.pop("revision")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if config["load_in_4bit"]:
        from peft import prepare_model_for_kbit_training
        model = prepare_kbit_model(
            model,
            prepare_model_for_kbit_training,
            gradient_checkpointing=bool(config["gradient_checkpointing"]),
        )
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.base_model_revision,
        trust_remote_code=bool(config["trust_remote_code"]),
        local_files_only=bool(config["local_files_only"]),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
    )
    train_dataset = load_json_dataset(args.train, "sft", modules["datasets"])
    eval_dataset = load_json_dataset(args.validation, "sft", modules["datasets"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    evaluation_key = "eval_strategy" if "eval_strategy" in SFTConfig.__dataclass_fields__ else "evaluation_strategy"
    training_args = {
        "output_dir": str(output),
        "seed": int(config["seed"]),
        "max_length": int(config["max_length"]),
        "max_seq_length": int(config["max_length"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "logging_steps": int(config.get("logging_steps", 10)),
        "eval_steps": int(config.get("eval_steps", 100)),
        "save_steps": int(config.get("save_steps", 100)),
        "save_total_limit": int(config.get("save_total_limit", 3)),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "bf16": bool(config["bf16"]),
        "fp16": bool(config["fp16"]),
        "logging_strategy": "steps",
        "save_strategy": "steps",
        evaluation_key: "steps",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "run_name": output.name,
        "packing": False,
    }
    training_args = supported_kwargs(SFTConfig, training_args)
    args_obj = SFTConfig(**training_args)
    trainer_kwargs = {
        "model": model,
        "args": args_obj,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": lora,
    }
    import inspect
    trainer_parameters = inspect.signature(SFTTrainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_state()
    metrics = {
        "train": train_result.metrics,
        "evaluation": eval_metrics,
    }
    (output / "fomo-training-metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    metadata = checkpoint_metadata(
        method="sft",
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        base_model_license=args.base_model_license.strip(),
        data=train_summary,
        eval_data=validation_summary,
        examples_trained=train_summary["examples"],
        config=config,
        extra={
            "adapter": "LoRA",
            "quantization": qlora_metadata(
                bool(config["load_in_4bit"]),
                "bfloat16" if config["bf16"] else ("float16" if config["fp16"] else None),
            ),
            "validation_metrics": eval_metrics,
            "libraries": {
                name: getattr(modules[name], "__version__", "unknown")
                for name in (
                    "torch", "transformers", "datasets", "peft", "trl", "accelerate",
                    *(("bitsandbytes",) if config["load_in_4bit"] else ()),
                )
            },
        },
    )
    manifest = seal_checkpoint(output, metadata)
    print(
        f"Training completed and checkpoint integrity manifest written: "
        f"{manifest['checkpoint_id']} ({output})"
    )
    print("This is a fine-tuned adapter, not a claim of benchmark quality or production readiness.")
    return manifest


if __name__ == "__main__":
    run()