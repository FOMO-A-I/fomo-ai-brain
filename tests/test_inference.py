import os
import queue
import hashlib
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from training.model_registry import seal_checkpoint
from fomo.brain.context import ConversationContext, Message
from fomo.brain.inference import infer
from fomo.brain.model import (
    CheckpointNotConfigured,
    LocalTransformersBackend,
    ModelLoadError,
)
from fomo.brain.router import ModelRouter


class ScriptedBackend:
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.requests = []

    def complete(self, messages):
        self.requests.append(tuple(messages))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeTensor:
    shape = (1, 2)

    def to(self, device):
        return self


class FakeStreamer:
    END = object()

    def __init__(self, tokenizer, *, skip_prompt, skip_special_tokens, timeout):
        self.queue = queue.Queue()
        self.timeout = timeout

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(timeout=self.timeout)
        if item is self.END:
            raise StopIteration
        return item

    def put(self, text):
        self.queue.put(text)

    def end(self):
        self.queue.put(self.END)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return {"input_ids": FakeTensor()}


class FakeTorch:
    class inference_mode:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False


class StreamingModel:
    class Embedding:
        class Weight:
            device = "cpu"

        weight = Weight()

    def __init__(self, *, error=None):
        self.error = error
        self.generation_kwargs = None
        self.generation_thread = None

    def get_input_embeddings(self):
        return self.Embedding()

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generation_thread = threading.current_thread()
        self.generation_kwargs = kwargs
        if self.error:
            raise self.error
        streamer = kwargs["streamer"]
        streamer.put("token ")
        streamer.put("stream")
        streamer.end()


class InferenceTests(unittest.TestCase):
    def test_local_backend_requires_an_explicit_existing_checkpoint(self):
        old = os.environ.pop("FOMO_MODEL_PATH", None)
        try:
            with self.assertRaisesRegex(CheckpointNotConfigured, "Set FOMO_MODEL_PATH"):
                LocalTransformersBackend()
        finally:
            if old is not None:
                os.environ["FOMO_MODEL_PATH"] = old

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "missing"
            with self.assertRaisesRegex(CheckpointNotConfigured, "does not exist"):
                LocalTransformersBackend(checkpoint)

    def test_context_is_committed_only_after_successful_inference(self):
        context = ConversationContext("FOMO test assistant", max_turns=2)
        backend = ScriptedBackend(["answer", RuntimeError("offline")])
        self.assertEqual(infer(backend, "first", context=context), "answer")
        self.assertEqual(
            [(item.role, item.content) for item in context.messages],
            [
                ("system", "FOMO test assistant"),
                ("user", "first"),
                ("assistant", "answer"),
            ],
        )
        with self.assertRaisesRegex(RuntimeError, "offline"):
            infer(backend, "second", context=context)
        self.assertEqual(len(context.messages), 3)
        self.assertEqual(backend.requests[1][-1], Message("user", "second"))

    def test_context_limits_and_message_roles_are_enforced(self):
        context = ConversationContext("system", max_message_chars=4)
        with self.assertRaises(ValueError):
            context.add("user", "12345")
        with self.assertRaises(ValueError):
            Message("developer", "do this")

    def test_model_router_only_uses_explicit_routes(self):
        default = ScriptedBackend()
        coding = ScriptedBackend()
        router = ModelRouter(default, task_models={"coding": coding})
        self.assertIs(router.for_task("coding"), coding)
        self.assertIs(router.for_task("research"), default)
        with self.assertRaises(ValueError):
            router.for_task("")

    def _stream_backend(self, model):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        backend = LocalTransformersBackend(directory.name, max_new_tokens=7)
        tokenizer = FakeTokenizer()
        patcher = patch.object(backend, "_load", return_value=(tokenizer, model))
        patcher.start()
        self.addCleanup(patcher.stop)
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.TextIteratorStreamer = FakeStreamer
        modules = patch.dict(
            "sys.modules",
            {"transformers": fake_transformers, "torch": FakeTorch},
        )
        modules.start()
        self.addCleanup(modules.stop)
        return backend, model

    def _sealed_checkpoint(self, checkpoint, base_model, *, adapter, adapter_base=None):
        checkpoint.mkdir(parents=True)
        (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        if adapter:
            (checkpoint / "adapter_config.json").write_text(
                json.dumps(
                    {"base_model_name_or_path": str(adapter_base or base_model)}
                ),
                encoding="utf-8",
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"test-adapter-weights")
        else:
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"test-model-weights")
        seal_checkpoint(
            checkpoint,
            {
                "method": "sft",
                "base_model": str(base_model),
                "base_model_license": "test-only",
                "base_model_terms_reviewed": True,
                "dataset": {"sha256": hashlib.sha256(b"test dataset").hexdigest()},
                "examples_trained": 1,
            },
        )
        return checkpoint

    def _install_fake_loader_modules(
        self, *, tokenizer_load_entered=None, tokenizer_load_release=None
    ):
        calls = {"tokenizer": [], "model": [], "adapter": []}

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["tokenizer"].append((path, kwargs))
                if tokenizer_load_entered is not None:
                    tokenizer_load_entered.set()
                    if not tokenizer_load_release.wait(timeout=3):
                        raise TimeoutError("test did not release fake tokenizer loading")
                return FakeTokenizer()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(path, **kwargs):
                calls["model"].append((path, kwargs))
                return StreamingModel()

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base, path, **kwargs):
                calls["adapter"].append((base, path, kwargs))
                return StreamingModel()

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = FakeAutoTokenizer
        transformers.AutoModelForCausalLM = FakeAutoModel
        transformers.TextIteratorStreamer = FakeStreamer
        peft = types.ModuleType("peft")
        peft.PeftModel = FakePeftModel
        modules = patch.dict(
            "sys.modules",
            {"torch": FakeTorch, "transformers": transformers, "peft": peft},
        )
        modules.start()
        self.addCleanup(modules.stop)
        return calls

    def test_stream_yields_streamer_chunks_and_bounds_generation(self):
        model = StreamingModel()
        backend, model = self._stream_backend(model)
        chunks = list(backend.stream([Message("user", "stream this")], max_new_tokens=3))
        self.assertEqual(chunks, ["token ", "stream"])
        self.assertEqual(model.generation_kwargs["max_new_tokens"], 3)
        self.assertIs(model.generation_kwargs["streamer"].__class__, FakeStreamer)
        self.assertNotEqual(model.generation_thread, threading.current_thread())

    def test_stream_rejects_unbounded_overrides_and_propagates_worker_errors(self):
        backend, _ = self._stream_backend(StreamingModel())
        with self.assertRaisesRegex(ValueError, "configured limit"):
            list(backend.stream([Message("user", "hi")], max_new_tokens=8))

        backend, _ = self._stream_backend(StreamingModel(error=ValueError("GPU failed")))
        with self.assertRaisesRegex(ModelLoadError, "GPU failed") as raised:
            list(backend.stream([Message("user", "hi")]))
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_runtime_refuses_checkpoint_without_completed_manifest_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "untrained"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            calls = self._install_fake_loader_modules()
            backend = LocalTransformersBackend(checkpoint)
            with self.assertRaisesRegex(ModelLoadError, "No completed training manifest"):
                backend._load()
            self.assertEqual(calls["tokenizer"], [])
            self.assertEqual(calls["model"], [])

    def test_adapter_checkpoint_loads_only_matching_local_base_and_local_peft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base-model"
            base.mkdir()
            (base / "config.json").write_text("{}", encoding="utf-8")
            checkpoint = self._sealed_checkpoint(
                root / "adapter-checkpoint", base, adapter=True
            )
            calls = self._install_fake_loader_modules()
            backend = LocalTransformersBackend(checkpoint)
            tokenizer, model = backend._load()
            self.assertIsInstance(tokenizer, FakeTokenizer)
            self.assertIsInstance(model, StreamingModel)
            self.assertEqual(calls["tokenizer"][0][0], str(checkpoint.resolve()))
            self.assertEqual(calls["tokenizer"][0][1]["local_files_only"], True)
            self.assertEqual(calls["tokenizer"][0][1]["trust_remote_code"], False)
            loaded_base, base_kwargs = calls["model"][0]
            self.assertEqual(loaded_base, str(base.resolve()))
            self.assertTrue(base_kwargs["local_files_only"])
            self.assertFalse(base_kwargs["trust_remote_code"])
            self.assertEqual(calls["adapter"][0][1], str(checkpoint.resolve()))
            self.assertFalse(calls["adapter"][0][2]["is_trainable"])
            self.assertTrue(calls["adapter"][0][2]["local_files_only"])

    def test_adapter_base_mismatch_and_manifest_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base-model"
            base.mkdir()
            checkpoint = self._sealed_checkpoint(
                root / "mismatched-adapter", base, adapter=True, adapter_base="another/base"
            )
            calls = self._install_fake_loader_modules()
            with self.assertRaisesRegex(ModelLoadError, "does not match"):
                LocalTransformersBackend(checkpoint)._load()
            self.assertEqual(calls["tokenizer"], [])
            self.assertEqual(calls["model"], [])

            tampered = self._sealed_checkpoint(
                root / "tampered-adapter", base, adapter=True
            )
            (tampered / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ModelLoadError, "integrity verification failed"):
                LocalTransformersBackend(tampered)._load()

    def test_full_transformers_checkpoint_loads_without_peft(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._sealed_checkpoint(
                Path(directory) / "merged-checkpoint",
                "local/base-id",
                adapter=False,
            )
            calls = self._install_fake_loader_modules()
            backend = LocalTransformersBackend(checkpoint)
            tokenizer, model = backend._load()
            self.assertIsInstance(tokenizer, FakeTokenizer)
            self.assertIsInstance(model, StreamingModel)
            loaded_path, kwargs = calls["model"][0]
            self.assertEqual(loaded_path, str(checkpoint.resolve()))
            self.assertTrue(kwargs["local_files_only"])
            self.assertFalse(kwargs["trust_remote_code"])
            self.assertEqual(calls["adapter"], [])

    def test_streaming_loads_and_runs_a_verified_full_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._sealed_checkpoint(
                Path(directory) / "merged-checkpoint",
                "local/base-id",
                adapter=False,
            )
            calls = self._install_fake_loader_modules()
            backend = LocalTransformersBackend(checkpoint, max_new_tokens=5)
            chunks = list(backend.stream([Message("user", "stream from checkpoint")]))
            self.assertEqual(chunks, ["token ", "stream"])
            self.assertTrue(calls["tokenizer"][0][1]["local_files_only"])
            self.assertFalse(calls["tokenizer"][0][1]["trust_remote_code"])
            self.assertTrue(calls["model"][0][1]["local_files_only"])

    def test_concurrent_first_load_initializes_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self._sealed_checkpoint(
                Path(directory) / "merged-checkpoint",
                "local/base-id",
                adapter=False,
            )
            tokenizer_load_entered = threading.Event()
            tokenizer_load_release = threading.Event()
            calls = self._install_fake_loader_modules(
                tokenizer_load_entered=tokenizer_load_entered,
                tokenizer_load_release=tokenizer_load_release,
            )
            backend = LocalTransformersBackend(checkpoint)
            results = []
            errors = []
            second_call_started = threading.Event()

            def load(first=False):
                if not first:
                    second_call_started.set()
                try:
                    results.append(backend._load())
                except Exception as exc:
                    errors.append(exc)

            first_thread = threading.Thread(target=load, kwargs={"first": True})
            second_thread = threading.Thread(target=load)
            first_thread.start()
            self.assertTrue(tokenizer_load_entered.wait(timeout=2))
            second_thread.start()
            self.assertTrue(second_call_started.wait(timeout=2))
            # Give the second caller time to reach _load while the first fake
            # tokenizer initialization is deliberately blocked.
            threading.Event().wait(0.05)
            tokenizer_load_release.set()
            first_thread.join(timeout=3)
            second_thread.join(timeout=3)

            self.assertFalse(first_thread.is_alive())
            self.assertFalse(second_thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 2)
            self.assertEqual(len(calls["tokenizer"]), 1)
            self.assertEqual(len(calls["model"]), 1)
            self.assertIs(results[0][0], results[1][0])
            self.assertIs(results[0][1], results[1][1])


if __name__ == "__main__":
    unittest.main()