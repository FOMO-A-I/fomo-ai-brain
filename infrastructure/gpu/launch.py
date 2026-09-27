"""Validate a real local FOMO checkpoint and GPU before starting inference."""

import os
import subprocess
import sys
from pathlib import Path

from training.common import DatasetError
from training.model_registry import read_checkpoint_manifest


def main() -> None:
    path = os.getenv("FOMO_MODEL_PATH", "")
    if not path or not Path(path).is_dir():
        raise SystemExit("Set FOMO_MODEL_PATH to an existing trained checkpoint directory")
    try:
        read_checkpoint_manifest(path, verify_files=True)
    except DatasetError as exc:
        raise SystemExit(f"Trained checkpoint verification failed: {exc}") from exc
    if not any((Path(path) / name).is_file() for name in ("config.json", "adapter_config.json")):
        raise SystemExit("Expected a full model or a verified LoRA adapter checkpoint")
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install the CUDA-compatible inference dependencies first") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU unavailable; refusing to claim GPU inference")
    print(f"CUDA ready ({torch.cuda.get_device_name(0)}); loading checkpoint from {path}", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "fomo.api.server"]))


if __name__ == "__main__":
    main()