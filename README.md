# FOMO AI Brain

FOMO AI Brain is the standalone Python **training and runtime stack** for future FOMO-specific checkpoints. It contains a data-validation pipeline, SFT and DPO entry points, checkpoint verification, offline evaluation tools, local inference, and a bounded agent runtime.

**Current status:** No trained FOMO weights, licensed training datasets, human preference labels, GPU training run, or measured checkpoint evaluation are included. The scripts and offline tests do not establish model quality. Inference requires a real, locally available checkpoint with a verified completion manifest; it does not select another model when one is missing.

The repository has no hosting-platform SDK requirement. It is not a completed trained model or a production multi-user service.

## Pipeline

Licensed training data → SFT (LoRA or opt-in 4-bit QLoRA) → human-ranked preferences → DPO → verified local checkpoint → held-out response generation → evaluation → local inference and agents

This is a sequence of runnable components, **not a claim that a training run or evaluation has taken place**. Training instructions and input contracts are in [`training/README.md`](training/README.md); evaluation instructions are in [`evals/README.md`](evals/README.md).

## Project areas

| Area | What the code does | What is still required |
| --- | --- | --- |
| `training/datasets/` | Validate, clean and format supplied SFT/preference records | Licensed, high-quality examples and provenance |
| `training/sft/`, `training/dpo/` | LoRA training entry points with optional 4-bit QLoRA | Compatible GPU, reviewed base-model license, real datasets and GPU runs |
| `training/model_registry.py` | Seal and verify operator-attested checkpoint metadata and file hashes | Real trained artifacts; a manifest alone does not prove quality |
| `evals/` | Generate responses from a verified local checkpoint and score supplied held-out cases | Actual checkpoint responses, curated cases and human review; no benchmark result is bundled |
| `fomo/brain/` | Local checkpoint inference and token streaming | Compatible trained model weights and tokenizer |
| `fomo/agents/`, `fomo/reasoning/` | Bounded planning, 36 named role profiles sharing one model backend, and model-reported verification | Actual checkpoint, role-quality testing; roles are not separately trained models |
| `fomo/memory/` | Opt-in, scoped SQLite vector retrieval in chat and tasks | A local embedding model, operator-configured storage and retention policy |
| `fomo/tools/` | Explicitly approved public-URL retrieval and an external sandbox client | Enable and authorize each tool; sandbox execution needs a real isolated service |
| `fomo/api/` | Local, token-protected backend API and rate guard | TLS, per-user authorization, distributed quotas and load testing before public use |
| `infrastructure/` | Development container, GPU startup checks, worker routing, SQL schema and metrics | Operator-provisioned GPU cluster and deployment |

Tool adapters are **not automatically enabled**. The URL provider fetches explicitly approved public URLs; it is not an unrestricted internet search engine. Python execution requires a separately configured HTTPS sandbox and never runs untrusted code locally. A model's request for a tool is not authorization to use one.

## Start with reproducible checks

From the repository root, with Python 3.10+:

```bash
python -m unittest discover -s tests -v
python -m evals --help
python -m evals.generate --help
```

Offline tests use small test doubles to check behavior. They do not measure a real model or claim thousands of benchmark cases. Add your licensed examples and human preference pairs to train and evaluate actual checkpoints; see `training/` and `evals/README.md` for the file formats and commands.

Only after training a real checkpoint, generate responses for a **held-out** case set and run strict benchmark mode. Generation loads that checkpoint; evaluation scores the saved responses and binds the report to the checkpoint manifest, case-file hash and generation settings identifier:

```bash
python -m evals.generate --cases data/eval-cases.jsonl \
  --checkpoint /absolute/path/to/a-trained-checkpoint \
  --output data/checkpoint-responses.jsonl \
  --max-new-tokens 1024

python -m evals --cases data/eval-cases.jsonl \
  --candidate data/checkpoint-responses.jsonl \
  --checkpoint /absolute/path/to/a-trained-checkpoint \
  --generation-settings-id "COPY-THE-EXACT-ID-PRINTED-BY-GENERATOR" \
  --json-out data/evaluation-report.json
```

Copy the settings ID printed by `evals.generate` into the second command. Generated responses are checked against the case-file and checkpoint hashes; externally prepared responses without that metadata require the explicit `--operator-attested-candidate` flag and remain self-attested. No result is included here. The evaluator does not independently certify model quality.

## Run an actual checkpoint

On a machine with a compatible GPU, install a CUDA-compatible PyTorch build and the requirements in `requirements.txt`. Then point the runtime at a **real local model checkpoint directory**:

```bash
export FOMO_MODEL_PATH=/absolute/path/to/a-trained-checkpoint
export FOMO_API_TOKEN='a-long-random-token-kept-out-of-the-repository'
python -m fomo.api.server
```

The service binds to `127.0.0.1:8765` and rejects non-loopback binding. It accepts only a local full checkpoint or LoRA adapter with a valid `.fomo-checkpoint.json` training manifest and matching file hashes. The manifest is self-attested metadata, **not independent proof** of training quality. It loads the model locally on first inference; without a verified checkpoint, inference fails instead of selecting a base model. A trusted website backend can call `POST /v1/chat` with JSON `{"messages":[{"role":"user","content":"Hello"}]}` and `Authorization: Bearer <token>`. `POST /v1/chat/stream` sends actual model-generated chunks as server-sent events. `POST /v1/tasks` accepts `{"prompt":"...","sources":[{"title":"...","content":"..."}]}` for a bounded plan and registered agents. `/health` reports configuration and whether model weights have loaded; `/metrics` requires the token.

Example programs under `examples/` demonstrate direct Python use. A browser must never receive the server token. This is a **single-operator reference API**, not a ready-made public multi-user site; the site backend must implement its own accounts, conversation ownership, permissions and retention.

### Optional memory and tools

Memory retrieval is disabled unless `FOMO_MEMORY_DB_PATH`, `FOMO_EMBEDDING_MODEL_PATH` and `FOMO_MEMORY_SCOPE_ID` are configured together. Install the optional dependencies in `requirements-memory.txt` and supply a compatible local embedding model. The authenticated `POST /v1/memory` endpoint explicitly writes a memory; `GET /v1/memory` lists memories in the server's fixed operator scope. Chat, streaming chat and tasks retrieve relevant memories from that scope. Requests cannot select another scope. This is **not multi-user memory authorization** or automatic long-term learning.

Tool use is separately disabled by default. Set `FOMO_ENABLE_WEB_RESEARCH=1` to permit the research agent to fetch **only caller-approved public URLs**. A task must explicitly supply `{"browse":{"urls":["https://example.org/page"],"approved_urls":["https://example.org/page"],"approved_domains":[]}}` alongside its prompt. This is restricted URL retrieval, **not a search engine or unrestricted internet browsing**. To permit the coding agent to execute Python, configure a real isolated HTTPS service using `FOMO_SANDBOX_URL` and `FOMO_SANDBOX_TOKEN`, set `FOMO_ENABLE_SANDBOX_EXECUTION=1`, and explicitly supply `{"execution":{"code":"print(1 + 1)","timeout_seconds":5}}` in the task. The repository does not provision that sandbox; it never runs submitted code in the API process. Tool-capable tasks still depend on the bounded planner selecting the corresponding agent. Do not expose either tool through an untrusted public site without an independent authorization and safety review.

## Training and deployment boundaries

Supervised fine-tuning (SFT) requires instruction/ideal-response pairs and a licensed base model. Direct preference optimization (DPO) requires genuine chosen/rejected rankings from humans (or clearly identified external data). These are separate stages; the repository does not invent either dataset. Checkpoint quality depends on data, compute and evaluation. A real training run may require substantial GPU memory, time, and cost.

The local Transformers backend does not provide continuous batching, KV-cache scheduling, distributed inference, GPU autoscaling, moderation, or a verified jailbreak defense. A production deployment needs an optimized serving engine, real isolation and policy enforcement, operational monitoring and a security review. The `infrastructure/` directory provides building blocks, **not a pre-deployed cluster**.

The `LICENSE` file currently reserves rights; public visibility alone does not grant reuse permission. The repository owner can replace it with an explicit license of their choice.