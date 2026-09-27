from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

from training.dpo.train import (
    _load_sft_adapter_pair,
    _named_adapter_config_values,
    _publish_policy_adapter,
    _require_named_adapter_support,
    _validate_quantization_compatibility,
)
from training.common import DatasetError, read_jsonl, write_jsonl
from training.datasets.clean import clean_records
from training.datasets.format_sft import format_record
from training.datasets.ingest import ingest
from training.datasets.split import split_records
from training.datasets.validate import validate_dpo_record, validate_sft_record
from training.model_registry import (
    register_checkpoint,
    read_checkpoint_manifest,
    resolve_checkpoint,
    seal_checkpoint,
)
from training.train_utils import (
    apply_qlora_override,
    model_quantization_kwargs,
    prepare_kbit_model,
    qlora_metadata,
    load_json_dataset,
    require_cuda,
    validate_config,
)


PROVENANCE = {
    "dataset_id": "fomo-owned-v1",
    "source": "team-authored examples",
    "license": "Internal training permission",
    "rights_attested": True,
}
TEST_PREFERENCE_PROVENANCE = {
    **PROVENANCE,
    "source": "test-only synthetic fixture; not human annotation",
    "preference_source": "test_fixture",
}


def sft_record(index: int, *, response: str | None = None) -> dict:
    return {
        "record_id": f"sft-{index}",
        "instruction": f"Explain step {index}",
        "response": response or f"Step {index} explained.",
        "provenance": dict(PROVENANCE),
    }


def dpo_record(index: int, *, provenance: dict | None = None) -> dict:
    return {
        "record_id": f"dpo-{index}",
        "prompt": f"Question {index}",
        "chosen": f"Clear answer {index}",
        "rejected": f"Unclear answer {index}",
        "provenance": dict(provenance or TEST_PREFERENCE_PROVENANCE),
    }


class DatasetValidationTests(unittest.TestCase):
    def test_sft_loader_normalizes_both_documented_input_formats(self) -> None:
        class FakeDataset:
            def __init__(self, rows):
                self.rows = rows

            @classmethod
            def from_list(cls, rows):
                return cls(rows)

        class FakeDatasets:
            Dataset = FakeDataset

        message_record = {
            "record_id": "messages-1",
            "messages": [
                {"role": "user", "content": "Help"},
                {"role": "assistant", "content": "Certainly."},
            ],
            "provenance": dict(PROVENANCE),
        }
        instruction_record = {
            "record_id": "instruction-1",
            "system": "Be precise",
            "instruction": "Explain this",
            "response": "Here is the explanation.",
            "provenance": dict(PROVENANCE),
        }
        with tempfile.TemporaryDirectory() as directory:
            for filename, record, expected_messages in (
                (
                    "messages.jsonl",
                    message_record,
                    message_record["messages"],
                ),
                (
                    "instruction.jsonl",
                    instruction_record,
                    [
                        {"role": "system", "content": "Be precise"},
                        {"role": "user", "content": "Explain this"},
                        {"role": "assistant", "content": "Here is the explanation."},
                    ],
                ),
            ):
                path = Path(directory) / filename
                write_jsonl(path, [record])
                dataset = load_json_dataset(path, "sft", FakeDatasets)
                self.assertEqual(dataset.rows, [{"messages": expected_messages}])
                self.assertNotIn("provenance", dataset.rows[0])

    def test_instruction_response_is_formatted_as_conversation(self) -> None:
        record = format_record(sft_record(1))
        self.assertEqual(
            record["messages"],
            [
                {"role": "user", "content": "Explain step 1"},
                {"role": "assistant", "content": "Step 1 explained."},
            ],
        )
        self.assertEqual(record["provenance"]["dataset_id"], "fomo-owned-v1")

    def test_message_format_requires_user_assistant_and_final_answer(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Help"},
                {"role": "assistant", "content": "Sure"},
            ],
            "provenance": dict(PROVENANCE),
        }
        self.assertEqual(len(validate_sft_record(row)["messages"]), 2)
        row["messages"].append({"role": "user", "content": "Another question"})
        with self.assertRaisesRegex(DatasetError, "end with an assistant"):
            validate_sft_record(row)

    def test_data_without_explicit_rights_attestation_is_rejected(self) -> None:
        row = sft_record(1)
        row["provenance"]["rights_attested"] = False
        with self.assertRaisesRegex(DatasetError, "rights_attested"):
            validate_sft_record(row)

    def test_dpo_requires_human_provenance_by_default(self) -> None:
        record = dpo_record(1, provenance={**PROVENANCE, "preference_source": "model"})
        with self.assertRaisesRegex(DatasetError, "human preference"):
            validate_dpo_record(record)
        self.assertEqual(validate_dpo_record(record, human_preferences=False)["chosen"], "Clear answer 1")

    def test_dpo_rejects_equal_chosen_and_rejected(self) -> None:
        record = dpo_record(1)
        record["rejected"] = record["chosen"]
        with self.assertRaisesRegex(DatasetError, "must differ"):
            validate_dpo_record(record, human_preferences=False)

    def test_dpo_accepts_chat_prompt_ending_in_user(self) -> None:
        record = dpo_record(1)
        record["prompt"] = [
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Question"},
        ]
        record["chosen"] = [
            {"role": "assistant", "content": "Clear answer"},
        ]
        record["rejected"] = [
            {"role": "assistant", "content": "Unclear answer"},
        ]
        self.assertEqual(
            validate_dpo_record(record, human_preferences=False)["prompt"][-1]["role"], "user"
        )

    def test_cleaning_deduplicates_content_and_tracks_other_origin(self) -> None:
        duplicate = sft_record(1)
        duplicate["record_id"] = "different-id"
        duplicate["provenance"] = {
            **PROVENANCE,
            "dataset_id": "second-source",
            "source": "licensed mirror",
        }
        rows, removed = clean_records([sft_record(1), duplicate], "sft")
        self.assertEqual(removed, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["duplicate_origins"][0]["dataset_id"], "second-source")

    def test_split_is_deterministic_and_has_both_partitions(self) -> None:
        rows = [format_record(sft_record(index)) for index in range(20)]
        first_train, first_eval = split_records(rows, validation_fraction=0.2, seed=7)
        second_train, second_eval = split_records(rows, validation_fraction=0.2, seed=7)
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_eval, second_eval)
        self.assertTrue(first_train)
        self.assertTrue(first_eval)
        self.assertFalse(
            {row["record_id"] for row in first_train}
            & {row["record_id"] for row in first_eval}
        )

    def test_split_refuses_duplicate_ids_and_too_small_data(self) -> None:
        duplicate_id = [format_record(sft_record(1)), format_record(sft_record(2))]
        duplicate_id[1]["record_id"] = duplicate_id[0]["record_id"]
        with self.assertRaisesRegex(DatasetError, "Duplicate record_id"):
            split_records(duplicate_id)
        duplicate_content = [format_record(sft_record(1)), format_record(sft_record(1))]
        duplicate_content[1]["record_id"] = "independent-id"
        with self.assertRaisesRegex(DatasetError, "Duplicate example content"):
            split_records(duplicate_content)
        with self.assertRaisesRegex(DatasetError, "At least two"):
            split_records([format_record(sft_record(1))])


class IngestionTests(unittest.TestCase):
    def test_ingest_requires_rights_confirmation_and_records_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "ingested.jsonl"
            write_jsonl(source, [{"instruction": "Question", "response": "Answer"}])
            manifest = {
                "dataset_id": "dataset-1",
                "source": "team authored",
                "license": "Internal",
                "rights_attested": True,
            }
            with self.assertRaisesRegex(DatasetError, "explicit --confirm-rights"):
                ingest(source, output, manifest, confirm_rights=False)
            self.assertEqual(ingest(source, output, manifest, confirm_rights=True), 1)
            record = next(read_jsonl(output))
            self.assertEqual(record["provenance"]["license"], "Internal")
            self.assertTrue(record["provenance"]["rights_attested"])

    def test_ingest_rejects_conflicting_row_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            output = root / "ingested.jsonl"
            write_jsonl(source, [{
                "instruction": "Question",
                "response": "Answer",
                "provenance": {"license": "not the manifest license"},
            }])
            manifest = {
                "dataset_id": "dataset-1",
                "source": "team authored",
                "license": "Internal",
                "rights_attested": True,
            }
            with self.assertRaisesRegex(DatasetError, "conflicts with the manifest"):
                ingest(source, output, manifest, confirm_rights=True)


class CheckpointRegistryTests(unittest.TestCase):
    def test_registry_requires_completed_manifest_and_checks_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "adapter.safetensors").write_bytes(b"trained-adapter")
            metadata = {
                "method": "sft",
                "base_model": "owner/model",
                "base_model_license": "Apache-2.0",
                "base_model_terms_reviewed": True,
                "dataset": {"sha256": "a" * 64},
                "examples_trained": 17,
            }
            manifest = seal_checkpoint(checkpoint, metadata)
            self.assertEqual(manifest["status"], "trained")
            registry = root / "registry.json"
            entry = register_checkpoint(checkpoint, registry)
            self.assertEqual(entry["checkpoint_id"], manifest["checkpoint_id"])
            self.assertEqual(resolve_checkpoint(entry["checkpoint_id"], registry), checkpoint)
            (checkpoint / "adapter.safetensors").write_bytes(b"altered")
            with self.assertRaisesRegex(DatasetError, "integrity verification failed"):
                read_checkpoint_manifest(checkpoint)

    def test_registry_does_not_accept_unsealed_pretrained_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pretrained = root / "base-model"
            pretrained.mkdir()
            (pretrained / "model.safetensors").write_bytes(b"pretrained, not trained here")
            with self.assertRaisesRegex(DatasetError, "No completed training manifest"):
                register_checkpoint(pretrained, root / "registry.json")


class DpoReferenceSemanticsTests(unittest.TestCase):
    def test_trl_config_receives_explicit_policy_and_reference_adapter_names(self) -> None:
        @dataclass
        class SupportedDPOConfig:
            model_adapter_name: str | None = None
            ref_adapter_name: str | None = None

        values = _named_adapter_config_values(
            SupportedDPOConfig,
            {"beta": 0.1},
            policy_name="fomo_policy",
            reference_name="fomo_sft_reference",
            trl_version="test-version",
        )
        self.assertEqual(values["model_adapter_name"], "fomo_policy")
        self.assertEqual(values["ref_adapter_name"], "fomo_sft_reference")
        _require_named_adapter_support(SupportedDPOConfig, "test-version")

    def test_old_trl_config_fails_closed_without_named_reference_support(self) -> None:
        @dataclass
        class UnsupportedDPOConfig:
            beta: float = 0.1

        with self.assertRaisesRegex(RuntimeError, "implicit base-model/reference fallback"):
            _require_named_adapter_support(UnsupportedDPOConfig, "old-test-version")

    def test_policy_and_frozen_reference_load_from_the_same_sft_adapter(self) -> None:
        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, base_model, checkpoint, *, adapter_name, is_trainable):
                instance = cls()
                instance.events = [
                    ("from_pretrained", base_model, checkpoint, adapter_name, is_trainable)
                ]
                instance.peft_config = {adapter_name: object()}
                return instance

            def load_adapter(self, checkpoint, *, adapter_name, is_trainable):
                self.events.append(("load_adapter", checkpoint, adapter_name, is_trainable))
                self.peft_config[adapter_name] = object()

            def set_adapter(self, adapter_name):
                self.events.append(("set_adapter", adapter_name))

        model = _load_sft_adapter_pair(
            "base-model-object",
            "/tmp/sft-checkpoint",
            FakePeftModel,
            policy_name="fomo_policy",
            reference_name="fomo_sft_reference",
        )
        policy_load, reference_load, activate = model.events
        self.assertEqual(policy_load[0], "from_pretrained")
        self.assertEqual(policy_load[3:], ("fomo_policy", True))
        self.assertEqual(reference_load[0], "load_adapter")
        self.assertEqual(reference_load[1], policy_load[2])
        self.assertEqual(reference_load[2:], ("fomo_sft_reference", False))
        self.assertEqual(activate, ("set_adapter", "fomo_policy"))

    def test_published_dpo_checkpoint_has_only_root_policy_adapter_and_valid_manifest(self) -> None:
        class FakeTrainer:
            def save_model(self, path):
                staged = Path(path)
                for name, payload in (
                    ("fomo_policy", b"trained-policy-fixture"),
                    ("fomo_sft_reference", b"frozen-reference-fixture"),
                ):
                    adapter_dir = staged / name
                    adapter_dir.mkdir(parents=True)
                    (adapter_dir / "adapter_config.json").write_text(
                        json.dumps({"base_model_name_or_path": "owner/model"}),
                        encoding="utf-8",
                    )
                    (adapter_dir / "adapter_model.safetensors").write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "dpo-output"
            checkpoint.mkdir()
            _publish_policy_adapter(FakeTrainer(), checkpoint, "fomo_policy")
            self.assertTrue((checkpoint / "adapter_config.json").is_file())
            self.assertEqual(
                (checkpoint / "adapter_model.safetensors").read_bytes(),
                b"trained-policy-fixture",
            )
            self.assertFalse((checkpoint / "fomo_sft_reference").exists())

            seal_checkpoint(
                checkpoint,
                {
                    "method": "dpo",
                    "base_model": "owner/model",
                    "base_model_license": "test-only",
                    "base_model_terms_reviewed": True,
                    "dataset": {"sha256": "b" * 64},
                    "examples_trained": 1,
                    "policy_adapter": "fomo_policy",
                    "reference_adapter": "fomo_sft_reference",
                    "reference_policy": "frozen-copy-of-sft-adapter",
                },
            )
            manifest = read_checkpoint_manifest(checkpoint, verify_files=True)
            self.assertIn("adapter_config.json", manifest["files"])
            self.assertIn("adapter_model.safetensors", manifest["files"])
            self.assertNotIn("fomo_sft_reference/adapter_model.safetensors", manifest["files"])


class QLoRATests(unittest.TestCase):
    def test_nf4_double_quant_config_is_explicit_and_dtype_is_forwarded(self) -> None:
        calls = []

        class FakeBitsAndBytesConfig:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        result = model_quantization_kwargs(
            True, FakeBitsAndBytesConfig, "mock-bfloat16"
        )
        self.assertEqual(result["device_map"], "auto")
        self.assertEqual(calls, [{
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "mock-bfloat16",
        }])

    def test_non_qlora_lora_adds_no_quantization_model_kwargs(self) -> None:
        class UnexpectedBitsAndBytesConfig:
            def __init__(self, **kwargs):
                raise AssertionError("quantization config should not be constructed")

        self.assertEqual(
            model_quantization_kwargs(False, UnexpectedBitsAndBytesConfig, object()), {}
        )
        self.assertFalse(apply_qlora_override({"load_in_4bit": False}, None)["load_in_4bit"])
        self.assertTrue(apply_qlora_override({"load_in_4bit": False}, True)["load_in_4bit"])
        self.assertFalse(apply_qlora_override({"load_in_4bit": True}, False)["load_in_4bit"])
        self.assertEqual(qlora_metadata(False), {"enabled": False})

    def test_kbit_preparation_uses_gradient_checkpointing_setting(self) -> None:
        calls = []
        model = object()

        def prepare(received_model, **kwargs):
            calls.append((received_model, kwargs))
            return "prepared-model"

        prepared = prepare_kbit_model(model, prepare, gradient_checkpointing=True)
        self.assertEqual(prepared, "prepared-model")
        self.assertEqual(calls, [(model, {"use_gradient_checkpointing": True})])

    def test_qlora_requires_a_single_supported_compute_dtype(self) -> None:
        valid = {"bf16": True, "fp16": False, "load_in_4bit": True}
        validate_config(valid, {"bf16", "fp16", "load_in_4bit"})
        with self.assertRaisesRegex(DatasetError, "requires exactly one"):
            validate_config(
                {"bf16": False, "fp16": False, "load_in_4bit": True},
                {"bf16", "fp16", "load_in_4bit"},
            )
        with self.assertRaisesRegex(DatasetError, "at most one"):
            validate_config(
                {"bf16": True, "fp16": True, "load_in_4bit": True},
                {"bf16", "fp16", "load_in_4bit"},
            )

    def test_cuda_dtype_preflight_fails_closed_and_allows_supported_modes(self) -> None:
        class FakeCuda:
            available = True
            bf16_supported = True

            def is_available(self):
                return self.available

            def is_bf16_supported(self):
                return self.bf16_supported

        class FakeTorch:
            cuda = FakeCuda()

        FakeTorch.cuda.available = False
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
            require_cuda(FakeTorch, bf16=False, fp16=True)
        FakeTorch.cuda.available = True
        FakeTorch.cuda.bf16_supported = False
        with self.assertRaisesRegex(RuntimeError, "does not support bf16"):
            require_cuda(FakeTorch, bf16=True, fp16=False)
        require_cuda(FakeTorch, bf16=False, fp16=True)

    def test_dpo_requires_parent_sft_quantization_and_compute_dtype_match(self) -> None:
        parent = {"quantization": {
            "enabled": True,
            "bits": 4,
            "quant_type": "nf4",
            "double_quant": True,
            "compute_dtype": "bfloat16",
        }}
        config = {"load_in_4bit": True, "bf16": True, "fp16": False}
        _validate_quantization_compatibility(parent, config)
        with self.assertRaisesRegex(DatasetError, "mode must match"):
            _validate_quantization_compatibility(
                parent, {"load_in_4bit": False, "bf16": True, "fp16": False}
            )
        with self.assertRaisesRegex(DatasetError, "compute dtype must match"):
            _validate_quantization_compatibility(
                parent, {"load_in_4bit": True, "bf16": False, "fp16": True}
            )
        _validate_quantization_compatibility(
            {"quantization": {"enabled": False}},
            {"load_in_4bit": False, "bf16": True, "fp16": False},
        )


if __name__ == "__main__":
    unittest.main()
