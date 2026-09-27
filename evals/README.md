# Offline evaluation framework

The runner evaluates **responses already produced elsewhere**. It does not call
a model, execute candidate code, or claim that a small hand-labeled suite proves
general intelligence. It supports any number of JSONL cases; coverage is
limited by the real labels supplied.

Run from `examples/fomo-ai-brain/`:

```bash
python -m evals --cases cases.jsonl --candidate current.jsonl \
  --baseline previous.jsonl --json-out report.json
```

The process exits `0` if every candidate case passes, `1` if any case fails,
and `2` for invalid input. Add `--fail-on-regression` to also fail on a
candidate case that passed on the baseline but now fails.

## Strict checkpoint benchmark

Use `--checkpoint ID_OR_DIRECTORY` to opt into strict benchmarking. The
candidate responses must be a separate `--candidate` JSONL file covering every
case ID exactly once; case-embedded candidates are rejected. Strict mode also
requires `--generation-settings-id` and `--json-out`:

```bash
python -m evals --cases cases.jsonl \
  --candidate fomo-responses.jsonl \
  --checkpoint 0123456789abcdef01234567 \
  --registry training/checkpoints/registry.json \
  --generation-settings-id "vllm-temp0-top_p1-max_tokens1024-v1" \
  --json-out benchmark-report.json
```

`--checkpoint` accepts a registered checkpoint ID (resolved read-only using
`training.model_registry`) or a checkpoint directory with a valid
`.fomo-checkpoint.json` manifest. Manifest and checkpoint files are verified;
the report records the manifest checkpoint ID and manifest SHA-256. `--publish`
is an explicit publish-report intent flag and additionally requires
`--checkpoint` and `--json-out`; it does not upload or sign anything.

The strict report also binds the **raw bytes** of the case JSONL by SHA-256,
records the supplied generation-settings identifier, and hashes the external
candidate JSONL. These provenance values make later comparison/review
repeatable, but they do not prove that the candidate outputs came from that
checkpoint or used those settings. The checkpoint/output association and
settings identifier are operator self-attestations: the evaluator never loads
a model, runs inference, generates responses, signs reports, or verifies a
generation server's logs.

The default mode remains `offline_label_check`. It remains compatible with
case-embedded responses and optional candidate files, and reports explicitly
state that there is no checkpoint binding. Do not describe an offline label
check as a measured checkpoint benchmark.

## JSONL contract

Each case has a unique `id`, a non-empty `prompt`, one supported `category`,
and labeled checks in `expect`. Candidate and baseline files are optional
JSONL files keyed by the same `id`; a record has either a `response` string
plus structured candidate fields, or a `candidate` value. Without a
`--candidate` file, each case may embed its candidate.

Example:

```json
{"id":"sum-1","category":"reasoning","prompt":"What is 2 + 2?","expect":{"numeric_answer":4}}
```

Candidate JSONL:

```json
{"id":"sum-1","response":"4"}
```

Supported categories and additional labels:

- `coding`: text constraints (`required_terms`, `forbidden_terms`, `exact`,
  `min_chars`, `max_chars`) and/or `tests`, a list of `{ "id", "expected" }`.
  Candidate `execution_results` may provide `{ "id", "actual" }`; these are
  compared as reported data only, never run or independently attested.
- `reasoning`: the text constraints above, or `numeric_answer` and optional
  non-negative absolute `tolerance`. The response must be one number.
- `tool_use`: candidate `tool_calls`; checks `allowed_tools`, `must_call`,
  `must_not_call`, and/or ordered `expected_calls`. A call is
  `{ "name": "...", "arguments": {} }`. No tool is invoked.
- `planning`: candidate `steps` with a `label` (or `title`, `task`, `name`);
  checks `required_steps`, ordered `expected_steps`, `max_steps`, and
  `check_dependencies: true`. Dependency-checked steps need unique `id`s and
  `depends_on` lists pointing only to earlier steps.
- `hallucination`: exact normalized `gold_claims` matching candidate `claims`,
  candidate `citations` constrained to labeled `evidence_ids`, and optional
  boolean `must_abstain` matching candidate `abstained`. This is a
  human-label-based grounding proxy; it does not prove claim meaning or truth.
- `agent_coordination`: candidate `agents` with `name` and `status`; checks
  `required_agents`, `all_agents_succeeded: true`, and ordered
  `expected_handoffs` records (`from`, `to`).
- `memory`: checks candidate `memory_ids` against `relevant_memory_ids` with
  optional `min_precision`/`min_recall` thresholds (both default to exact
  set-match), `memories` against an exact `expected_memories` ID/value map, and
  excluded IDs via `forbidden_memory_ids`. These checks score supplied
  retrieval/retention outputs; they do not validate a memory store, relevance
  semantics, privacy policy, or what the model actually recalled internally.
- `instruction_following`: adds `min_words`, `max_words`, ordered
  `required_order` phrases, exact `items`, required/exact `structured` fields,
  and non-empty `required_sections`. These are literal/count/structure
  constraints—not a semantic judge of whether an instruction was understood.
- `safety`: `must_not_disclose` performs a case-insensitive literal substring
  check; `must_refuse` compares an independently reviewed candidate field
  `safety_review.refused`; `required_refusal_terms` checks literal phrases;
  and `forbidden_actions`/`allowed_actions` compare `observed_actions` supplied
  by an external tool audit. These checks do **not** detect jailbreaks,
  classify harmful intent, or provide semantic moderation. Review labels and
  audit records must be generated independently; a model's own self-report is
  not reliable safety evidence.
- `regression`: uses the numeric-answer/text checks from `reasoning`; the
  separate `--baseline` report classifies every matched case as improved,
  regressed, or unchanged.

Every case must provide at least one relevant labeled check. Missing labels
fail explicitly rather than treating an unconstrained response as correct.
Inputs are checked for duplicate IDs, malformed JSON, and unmatched outputs.
The report contains per-check evidence, per-category pass/fail counts, and
baseline deltas; all metrics are deterministic and only as trustworthy as
their labels.