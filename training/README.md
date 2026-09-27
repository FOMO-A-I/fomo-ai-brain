# Train an actual FOMO checkpoint

These scripts require **real, licensed data and an external CUDA machine**. No
data, preference labels, or model weights are included. A successful dry run
validates only input and configuration; it does not train a model.

## Data contracts

Each line is a UTF-8 JSON object. Every record needs a `provenance` object with
nonempty `dataset_id`, `source`, `license`, and `rights_attested: true`. Rights
attestation is an operator claim, not a legal review. Avoid personal data and
remove likely secrets before training.

- **SFT:** `instruction` and `response`, or `messages` with roles, ending in
  an ideal `assistant` response. A separate held-out validation split is
  required.
- **DPO:** `prompt`, `chosen` and `rejected` responses. By default provenance
  must also include `preference_source: "human"`, `annotator_count >= 1` and
  `annotation_protocol`. Do not mark synthetic rankings as human.

Validate real JSONL files from the repository root:

```bash
python -m training.datasets.validate --input data/sft-train.jsonl --kind sft
python -m training.datasets.validate --input data/dpo-train.jsonl --kind dpo
```

## Supervised fine-tuning

Choose a base model whose license permits your intended use. Review its terms
and record the actual license identifier. Install `requirements-training.txt`
in a dedicated GPU environment using a compatible CUDA PyTorch build. Check
the configuration in `training/sft/config.yaml` against available VRAM.

```bash
python -m training.sft.train \
  --train data/sft-train.jsonl \
  --validation data/sft-validation.jsonl \
  --output-dir training/checkpoints/fomo-sft-001 \
  --base-model /models/licensed-base-model \
  --base-model-license 'REVIEWED-LICENSE-ID' \
  --reviewed-base-model-terms \
  --dry-run
```

Remove `--dry-run` only on the GPU machine when the data and terms have been
reviewed. The output is a LoRA adapter plus a completion manifest, **not** a
new foundation model. Training and validation sets must be disjoint.

## Preference optimization

After SFT, collect genuine preference pairs, keep a separate validation split,
and run:

```bash
python -m training.dpo.train \
  --train data/dpo-train.jsonl \
  --validation data/dpo-validation.jsonl \
  --sft-checkpoint training/checkpoints/fomo-sft-001 \
  --output-dir training/checkpoints/fomo-dpo-001 \
  --dry-run
```

Remove `--dry-run` on the GPU machine. DPO rejects an absent or altered SFT
checkpoint. This is DPO, **not PPO-based RLHF or a separately trained reward
model**. Those would be separate research and infrastructure work.

## Optional 4-bit QLoRA

For memory-constrained CUDA training, install the optional
`requirements-qlora.txt` alongside the training requirements, with a
CUDA-compatible PyTorch build. Add `--qlora` to **both** SFT and DPO commands
above (or set `load_in_4bit: true` in both configurations). The scripts use NF4
quantization, double quantization, mixed-precision compute and LoRA adapters.
DPO requires the same quantization mode and compute dtype as its verified SFT
checkpoint. Set `bf16` or `fp16` to match the GPU, and review sequence length,
batch size and base-model size before training. `--dry-run` checks inputs but
does not load the quantized model.

This is an optional path intended to make some workloads feasible on 32 GB
GPUs, **not a guarantee** that any particular base model or context length
fits. No 4-bit GPU run has been completed in this repository.

Register a completed checkpoint:

```bash
python -m training.model_registry add \
  --checkpoint training/checkpoints/fomo-dpo-001 \
  --registry training/checkpoints/registry.json
```

Run the evaluation suite against actual saved model responses before serving a
candidate. No benchmark result is claimed without real cases and inference.