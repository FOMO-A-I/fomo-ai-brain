"""Lazy, local-only inference from an explicitly supplied model checkpoint.

This module deliberately has no default model and never falls back to an
unrelated hosted model.  A checkpoint must exist locally before inference.
Transformers and PyTorch are optional runtime dependencies and are imported
only when checkpoint loading or inference is requested.
"""

from __future__ import annotations

import json
import os
from queue import Empty
import threading
from pathlib import Path
from typing import Iterator, Protocol, Sequence

from .context import Message


class ModelBackend(Protocol):
    """Small interface shared by planning and agent components."""

    def complete(self, messages: Sequence[Message | dict[str, str]]) -> str:
        """Return one assistant completion for a chat-style prompt."""


class CheckpointNotConfigured(RuntimeError):
    """Raised when an explicitly configured local checkpoint is unavailable."""


class ModelLoadError(RuntimeError):
    """Raised when the configured local checkpoint cannot be loaded."""


class LocalTransformersBackend:
    """Run complete or token-streamed inference from a local checkpoint."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        device_map: str = "auto",
    ) -> None:
        configured = model_path if model_path is not None else os.getenv("FOMO_MODEL_PATH")
        if configured is None or not str(configured).strip():
            raise CheckpointNotConfigured(
                "Set FOMO_MODEL_PATH to a trained local checkpoint directory. "
                "No model is bundled and no fallback model will be selected."
            )
        path = Path(configured).expanduser()
        if not path.is_dir():
            raise CheckpointNotConfigured(
                f"FOMO checkpoint directory does not exist: {path}. "
                "Train or provide a checkpoint before starting inference."
            )
        if max_new_tokens < 1 or max_new_tokens > 8192:
            raise ValueError("max_new_tokens must be between 1 and 8192")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not device_map:
            raise ValueError("device_map must not be empty")

        self.model_path = path.resolve()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self._tokenizer = None
        self._model = None
        self._manifest = None
        self._load_lock = threading.Lock()
        self._lock = threading.Lock()

    def complete(self, messages: Sequence[Message | dict[str, str]]) -> str:
        normalized = self._normalize_messages(messages)
        tokenizer, model = self._load()

        with self._lock:
            try:
                encoded = tokenizer.apply_chat_template(
                    normalized,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                input_device = self._input_device(model)
                encoded = {key: value.to(input_device) for key, value in encoded.items()}
                input_length = int(encoded["input_ids"].shape[-1])

                import torch

                generation = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": self.temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                }
                if self.temperature > 0:
                    generation["temperature"] = self.temperature
                with torch.inference_mode():
                    output = model.generate(**encoded, **generation)
                completion = tokenizer.decode(
                    output[0][input_length:], skip_special_tokens=True
                ).strip()
            except Exception as exc:
                raise ModelLoadError(f"Local checkpoint inference failed: {exc}") from exc
        if not completion:
            raise ModelLoadError("The local checkpoint returned an empty completion.")
        return completion

    def stream(
        self,
        messages: Sequence[Message | dict[str, str]],
        *,
        max_new_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield decoded text as Transformers generates tokens.

        ``max_new_tokens`` may lower, but never raise, the backend's configured
        generation bound. Model generation runs on a worker thread because
        ``TextIteratorStreamer`` is a synchronous iterator over that worker.
        Generation failures are relayed to the consumer as ``ModelLoadError``
        with the original exception chained.
        """
        normalized = self._normalize_messages(messages)
        token_limit = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if isinstance(token_limit, bool) or not isinstance(token_limit, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= token_limit <= self.max_new_tokens:
            raise ValueError(
                f"max_new_tokens must be between 1 and the configured limit "
                f"({self.max_new_tokens})"
            )

        tokenizer, model = self._load()
        try:
            # This import remains lazy so CPU-only installs and tests do not
            # need to import Transformers at module load time.
            from transformers import TextIteratorStreamer

            encoded = tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            input_device = self._input_device(model)
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=0.1,
            )
            generation = {
                "max_new_tokens": token_limit,
                "do_sample": self.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
                "streamer": streamer,
            }
            if self.temperature > 0:
                generation["temperature"] = self.temperature
        except Exception as exc:
            if isinstance(exc, ModelLoadError):
                raise
            raise ModelLoadError(f"Could not prepare local checkpoint streaming: {exc}") from exc

        worker_error: list[Exception] = []
        worker_done = threading.Event()

        def generate() -> None:
            try:
                import torch

                # Use the same lock as complete() so one loaded model is never
                # asked to generate two sequences concurrently.
                with self._lock:
                    with torch.inference_mode():
                        model.generate(**encoded, **generation)
            except Exception as exc:
                worker_error.append(exc)
            finally:
                worker_done.set()

        worker = threading.Thread(
            target=generate,
            name="fomo-checkpoint-stream",
            daemon=True,
        )
        worker.start()
        try:
            while True:
                try:
                    chunk = next(streamer)
                except Empty:
                    if worker_done.is_set():
                        break
                    continue
                except StopIteration:
                    break
                if chunk:
                    yield chunk
            worker.join()
            if worker_error:
                raise ModelLoadError(
                    f"Local checkpoint streaming failed: {worker_error[0]}"
                ) from worker_error[0]
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Local checkpoint streaming failed: {exc}") from exc

    def _load(self):
        # Keep initialization synchronization separate from generation. This
        # covers validation and all heavyweight loads, then the cache check
        # makes later concurrent callers reuse the initialized pair.
        with self._load_lock:
            return self._load_checkpoint()

    def _load_checkpoint(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        manifest = self._read_verified_manifest()
        adapter_config_path = self.model_path / "adapter_config.json"
        is_adapter = adapter_config_path.is_file()
        base_model = None
        local_base = None
        if is_adapter:
            try:
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ModelLoadError(f"Invalid local PEFT adapter config: {exc}") from exc
            adapter_base = adapter_config.get("base_model_name_or_path")
            if not isinstance(adapter_base, str) or not adapter_base.strip():
                raise ModelLoadError("PEFT adapter config has no base_model_name_or_path.")
            base_model = manifest["base_model"]
            if self._normalize_model_reference(adapter_base) != self._normalize_model_reference(
                base_model
            ):
                raise ModelLoadError(
                    "PEFT adapter base model does not match the verified checkpoint manifest."
                )
            local_base = self._local_model_reference(base_model)
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path), local_files_only=True, trust_remote_code=False
            )
            if is_adapter:
                from peft import PeftModel

                base = AutoModelForCausalLM.from_pretrained(
                    local_base,
                    local_files_only=True,
                    trust_remote_code=False,
                    device_map=self.device_map,
                    **self._base_model_revision(manifest),
                )
                model = PeftModel.from_pretrained(
                    base,
                    str(self.model_path),
                    is_trainable=False,
                    local_files_only=True,
                )
            else:
                if not (self.model_path / "config.json").is_file():
                    raise ModelLoadError(
                        "Verified checkpoint is neither a PEFT adapter nor a full "
                        "Transformers model (missing config.json)."
                    )
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    local_files_only=True,
                    trust_remote_code=False,
                    device_map=self.device_map,
                )
            model.eval()
        except ModelLoadError:
            raise
        except ImportError as exc:
            raise ModelLoadError(
                "Local checkpoint inference requires PyTorch and Transformers; "
                "adapter checkpoints additionally require PEFT. Install the optional "
                "inference dependencies in the GPU environment."
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load local checkpoint at {self.model_path}: {exc}"
            ) from exc
        self._tokenizer, self._model = tokenizer, model
        self._manifest = manifest
        return tokenizer, model

    def _read_verified_manifest(self):
        """Validate training provenance and every checkpoint file before loading."""
        try:
            # Keep the training package optional for importing this module, but
            # require its canonical verifier before any model or tokenizer load.
            from training.model_registry import read_checkpoint_manifest

            return read_checkpoint_manifest(self.model_path, verify_files=True)
        except Exception as exc:
            raise ModelLoadError(
                f"Checkpoint validation failed for {self.model_path}: {exc}"
            ) from exc

    @staticmethod
    def _normalize_model_reference(reference: str) -> str:
        path = Path(reference).expanduser()
        if path.is_absolute() or path.exists() or reference.startswith(("./", "../", "~/")):
            return str(path.resolve())
        return reference

    @staticmethod
    def _local_model_reference(reference: str) -> str:
        """Resolve filesystem references; repo IDs can only resolve from cache."""
        path = Path(reference).expanduser()
        if path.is_absolute() or path.exists() or reference.startswith(("./", "../", "~/")):
            resolved = path.resolve()
            if not resolved.is_dir():
                raise ModelLoadError(f"PEFT base model is not available locally: {resolved}")
            return str(resolved)
        # Transformers is always passed local_files_only=True by the caller.
        return reference

    @staticmethod
    def _base_model_revision(manifest) -> dict[str, str]:
        revision = manifest.get("base_model_revision")
        if isinstance(revision, str) and revision.strip():
            return {"revision": revision}
        return {}

    @staticmethod
    def _normalize_messages(
        messages: Sequence[Message | dict[str, str]],
    ) -> list[dict[str, str]]:
        if not messages or len(messages) > 128:
            raise ValueError("inference requires between 1 and 128 messages")
        normalized: list[dict[str, str]] = []
        total_chars = 0
        for message in messages:
            if isinstance(message, Message):
                role, content = message.role, message.content
            elif isinstance(message, dict):
                role, content = message.get("role"), message.get("content")
            else:
                raise TypeError("messages must contain Message or role/content dictionaries")
            checked = Message(role, content)
            total_chars += len(checked.content)
            if total_chars > 100_000:
                raise ValueError("combined prompt exceeds the 100,000 character limit")
            normalized.append(checked.as_dict())
        return normalized

    @staticmethod
    def _input_device(model):
        # With device_map="auto", the input embedding module is the authoritative
        # location for input IDs; fall back to the model's first parameter.
        try:
            return model.get_input_embeddings().weight.device
        except (AttributeError, TypeError):
            try:
                return next(model.parameters()).device
            except (AttributeError, StopIteration) as exc:
                raise ModelLoadError("Loaded checkpoint exposes no input device.") from exc