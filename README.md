# FOMO AI Brain

FOMO AI Brain is FOMO AI's Python stack for training, evaluating, and serving local model checkpoints. The workflow is dataset validation → SFT → DPO → checkpoint → evaluation → inference.

## Codebase

| Path | Purpose |
| --- | --- |
| `training/datasets/` | Validate SFT examples and preference pairs. |
| `training/sft/`, `training/dpo/` | LoRA training with optional 4-bit NF4 QLoRA. |
| `training/model_registry.py` | Checkpoint manifests, file hashes, and registry. |
| `evals/` | Generate checkpoint responses and score held-out JSONL cases. |
| `fomo/brain/` | Local checkpoint inference and token streaming. |
| `fomo/agents/`, `fomo/reasoning/` | Bounded task planning and 36 role profiles on a shared model backend. |
| `fomo/memory/` | Scoped SQLite memory retrieval with local embeddings. |
| `fomo/tools/` | Approved-URL research and external Python sandbox adapters. |
| `fomo/api/` | Loopback JSON/SSE API with bearer-token authentication. |

## Development

Use Python 3.10+ from the repository root:

```bash
python -m unittest discover -s tests -v
```

Training data formats and SFT/DPO commands are in [training/README.md](training/README.md). Install `requirements-training.txt` with a CUDA-compatible PyTorch build for GPU training; install `requirements-qlora.txt` and use `--qlora` in both stages for the optional 4-bit path.

To generate responses for held-out cases and score them against a checkpoint, follow [evals/README.md](evals/README.md). The generator prints the settings ID required by the strict evaluation command.

## Local API

Set a local checkpoint path and an API token, then start the server:

```bash
export FOMO_MODEL_PATH=/path/to/checkpoint
export FOMO_API_TOKEN="$(openssl rand -hex 32)"
python -m fomo.api.server
```

The server listens on `127.0.0.1:8765`. For example:

```bash
curl http://127.0.0.1:8765/v1/chat \
  -H "Authorization: Bearer $FOMO_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

`POST /v1/chat/stream` streams responses; `POST /v1/tasks` runs bounded agent tasks. `GET /health` and authenticated `GET /metrics` provide service status.

## Optional capabilities

- **Memory:** Install `requirements-memory.txt`. Set `FOMO_MEMORY_DB_PATH`, `FOMO_EMBEDDING_MODEL_PATH`, and `FOMO_MEMORY_SCOPE_ID` together. Chat and tasks retrieve from that scope; authenticated `POST /v1/memory` writes entries.
- **Research:** Set `FOMO_ENABLE_WEB_RESEARCH=1`. Research tasks can retrieve URLs explicitly approved in their request.
- **Python execution:** Configure `FOMO_SANDBOX_URL` and `FOMO_SANDBOX_TOKEN` for an isolated HTTPS sandbox, then set `FOMO_ENABLE_SANDBOX_EXECUTION=1`. Execution requires an explicit task request.

See [.env.example](.env.example) for configuration names.

## FOMO token

FOMO token contract address (CA): `ExWPmvNCXPbkQG8qXe9Ddjhgi53UvLCu2pn1akzGpump`

Token information is separate from the FOMO Brain engineering roadmap. Token checkout is not live.
