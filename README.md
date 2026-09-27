# FOMO AI Brain

FOMO AI Brain is the core model and training infrastructure we are building for FOMO AI.

This repository contains our training pipeline, SFT and DPO workflows, model checkpoints, evaluation system, inference runtime, memory, reasoning and multi-agent architecture.

Our goal is to continuously train and improve FOMO-specific model checkpoints using our own datasets, preference data and evaluation results.

## How it works

Training Data → SFT → DPO → FOMO Checkpoint → Evaluation → FOMO Brain → Agents → Memory → Tools → Verification

We are building the system in separate modules so the model, training pipeline, agents, memory and tools can all evolve independently as FOMO AI grows.

The first FOMO-trained checkpoints will be added as GPU training and evaluation are completed.

## Project areas

| Area | What the code does | What is still required |
| --- | --- | --- |
| `training/datasets/` | Validate, clean and format supplied SFT/preference records | Licensed, high-quality examples and provenance |
| `training/sft/`, `training/dpo/` | GPU training entry points and configurations | GPU environment, base model license, actual training runs |
| `training/model_registry.py` | Record checkpoint metadata and evaluations | Real checkpoint artifacts and measured reports |
| `evals/` | Deterministic checks and baseline comparisons across multiple categories | Curated cases and independent human review; no fabricated scores |
| `fomo/brain/` | Local checkpoint inference and token streaming | Compatible trained model weights and tokenizer |
| `fomo/agents/`, `fomo/reasoning/` | Bounded planning, research over supplied sources, coding and verification | Quality/safety evaluation with the eventual model |
| `fomo/memory/`, `fomo/tools/` | Scoped memory and explicitly restricted tool adapters | Configure external services only as needed |
| `fomo/api/` | Local, token-protected backend API and rate guard | TLS, per-user authorization, distributed quotas and load testing before public use |
| `infrastructure/` | Development container, GPU startup checks, worker routing, SQL schema and metrics | Operator-provisioned GPU cluster and deployment |

The Python tool **does not run untrusted code locally**. Code execution needs an explicitly configured isolated external sandbox. The research agent works only with sources supplied to a task; web access is a separate restricted tool and is not silently enabled.

## Start with reproducible checks

From the repository root, with Python 3.10+:

```bash
python -m unittest discover -s tests -v
python -m evals --help
```

Offline tests use small test doubles to check behavior. They do not measure a real model or claim thousands of benchmark cases. Add your licensed examples and human preference pairs to train and evaluate actual checkpoints; see `training/` and `evals/README.md` for the file formats and commands.

After generating responses for a **held-out** case set with a real checkpoint, use strict benchmark mode. It requires complete external candidate responses and binds the report to the checkpoint manifest, case-file hash and generation settings identifier:

```bash
python -m evals --cases data/eval-cases.jsonl \
  --candidate data/checkpoint-responses.jsonl \
  --checkpoint /absolute/path/to/a-trained-checkpoint \
  --generation-settings-id reviewed-run-settings \
  --json-out data/evaluation-report.json
```

The evaluator does not generate those responses or independently certify their provenance.

## Run an actual checkpoint

On a machine with a compatible GPU, install a CUDA-compatible PyTorch build and the requirements in `requirements.txt`. Then point the runtime at a **real local model checkpoint directory**:

```bash
export FOMO_MODEL_PATH=/absolute/path/to/a-trained-checkpoint
export FOMO_API_TOKEN='a-long-random-token-kept-out-of-the-repository'
python -m fomo.api.server
```

The service binds to `127.0.0.1:8765` and rejects non-loopback binding. It accepts only a local full checkpoint or LoRA adapter with a valid `.fomo-checkpoint.json` training manifest and matching file hashes. The manifest is self-attested metadata, **not independent proof** of training quality. It loads the model locally on first inference; without a verified checkpoint, inference fails instead of selecting a base model. A trusted website backend can call `POST /v1/chat` with JSON `{"messages":[{"role":"user","content":"Hello"}]}` and `Authorization: Bearer <token>`. `POST /v1/chat/stream` sends actual model-generated chunks as server-sent events. `POST /v1/tasks` accepts `{"prompt":"...","sources":[{"title":"...","content":"..."}]}` for a bounded plan and registered agents. `/health` reports configuration and whether model weights have loaded; `/metrics` requires the token.

Example programs under `examples/` demonstrate direct Python use. A browser must never receive the server token. This is a **single-operator reference API**, not a ready-made public multi-user site; the site backend must implement its own accounts, conversation ownership, permissions and retention.

## Training and deployment boundaries

Supervised fine-tuning (SFT) requires instruction/ideal-response pairs and a licensed base model. Direct preference optimization (DPO) requires genuine chosen/rejected rankings from humans (or clearly identified external data). These are separate stages; the repository does not invent either dataset. Checkpoint quality depends on data, compute and evaluation. A real training run may require substantial GPU memory, time, and cost.

The local Transformers backend does not provide continuous batching, KV-cache scheduling, distributed inference, GPU autoscaling, moderation, or a verified jailbreak defense. A production deployment needs an optimized serving engine, real isolation and policy enforcement, operational monitoring and a security review. The `infrastructure/` directory provides building blocks, **not a pre-deployed cluster**.

The `LICENSE` file currently reserves rights; public visibility alone does not grant reuse permission. The repository owner can replace it with an explicit license of their choice.