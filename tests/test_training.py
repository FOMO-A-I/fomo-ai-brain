from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import dataclass

from training.dpo.train import (
    _load_sft_adapter_pair,
    _named_adapter_config_values,
    _require_named_adapter_support,
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


if __name__ == "__main__":
    unittest.main()