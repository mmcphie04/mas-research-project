# Internal Representations for Evaluating Answer Revisions in Multi-Agent Debate

Harvard CS independent study, Fall 2026. Advisor: Prof. Mengyu Wang.

Can linear probes on a language model's internal activations tell a **beneficial** answer revision
from a **harmful** one during multi-agent debate, well enough to improve final panel accuracy?

See [`docs/PLAN.md`](docs/PLAN.md) for the full design and schedule, and
[`docs/RISKS.md`](docs/RISKS.md) for the risk register.

## Design in one paragraph

Three homogeneous Gemma agents answer an MMLU-Pro question independently (round 0), then each sees
the other two agents' rationales and revises (round 1). One round only. A forced-format readout
pass extracts a canonical activation state and a forced-choice answer distribution before and after
peer exposure. Revisions are labelled beneficial / harmful / unhelpful. Because there is exactly one
round, the gating intervention picks between two answers that were **already generated** — so all
policy evaluation is offline and needs no GPU.

## Quick start

```bash
uv sync --extra dev                    # laptop: analysis + tier-0
uv run pytest -q                       # ~130 tests, about a minute, no GPU needed

uv run mad run    --config configs/base.yaml configs/smoke_stub.yaml
uv run mad report --run runs/smoke-stub
uv run mad power  --n 300 600 900 1500 --auroc 0.65

# the resumable multi-condition driver used on Colab (docs/COLAB.md), here on the stub
uv run python scripts/run_experiment.py --config configs/base.yaml configs/smoke_stub.yaml \
    --out-root runs --name demo --dataset synthetic --n-questions 20 --chunk 10
uv run python scripts/summarize_run.py --out-root runs --name demo
```

## The three backend tiers

The model sits behind a config value, so code correctness is decoupled from GPU availability.

| Tier | `model.backend` | What it is | Speed | What it validates |
|---|---|---|---|---|
| 0 | `stub` | No model. Planted ground truth: a closed-form B/H/U distribution and an activation signal of configurable strength. | ms | Schemas, storage, and the entire analysis layer against a known answer |
| 1 | `hf` | A real but tiny model on CPU. Defaults to `google/gemma-3-270m-it` (18 layers x 640). | ~8 s/question | Prompt rendering, readout token indexing, letter logits, activation shapes |
| 2 | `vllm` / `hf` | `google/gemma-3-12b-it` on a GPU. | hours | The actual experiment |

```bash
uv sync --extra dev --extra hf
uv run mad run --config configs/base.yaml configs/smoke_tiny.yaml     # tier 1, CPU
```

Gemma is gated: accept the licence on the model page and run `huggingface-cli login`. The token
lands in `~/.cache/huggingface/token` and `transformers` reads it automatically -- it never needs to
appear in a config file. Without it, set `model.repo_id` to the ungated
`HuggingFaceTB/SmolLM2-135M-Instruct`.

Confirmed shapes, which the storage arithmetic depends on:

| model | layers | d_model | bytes/state |
|---|---|---|---|
| `google/gemma-3-270m-it` | 18 | 640 | 24 KB |
| `google/gemma-2-9b-it` | 42 | 3584 | 308 KB |
| `google/gemma-3-12b-it` | 48 | 3840 | 376 KB |

Two Gemma-specific things the backend handles: vision-capable Gemma 3 checkpoints (`gemma3`, 4b and
up) nest their shape under `config.text_config` and only load via the image-text class, and Gemma's
chat template forbids a system role while requiring strictly alternating user/assistant turns -- so
the system prompt is merged into the first user turn and the agent's own round-0 response is carried
as a real assistant turn.

### Tier 0 is the instrument, not a placeholder

A 135M model is at chance on MMLU-Pro and produces no realistic revision distribution, so tier 1
validates plumbing but not logic. Tier 0 plants a distribution whose answer is known in closed form,
which is what lets the labeller, panel vote, tie-break, four oracles and McNemar code be verified
before any GPU time is spent. Two knobs matter:

* `probe_signal` — instance-level correctness signal, recoverable within a question.
* `difficulty_leak` — question-level signal only. Produces a high **marginal** AUROC and a
  **conditional** AUROC at chance: the difficulty-detector shortcut, planted on purpose so the
  diagnostic that catches it can itself be tested. `configs/smoke_stub_difficulty.yaml`.

## Layout

```
src/mad/            collection: config, provenance, data, permutation, prompts, backends, store, debate
src/mad/analysis/   offline: labels, oracles, probes, calibration, baselines, gating, stats, power_sim, report
configs/            base + smoke (stub / difficulty / tiny) + pilot
tests/              tier-0 ground-truth assertions and the backend conformance suite
runs/<run_id>/      transcripts, activation shards, index, manifest.json, completion.json
```

## Four design decisions worth knowing before reading the code

1. **There is no non-degenerate "revision quality" label.** Among revised events every binary
   partition of {beneficial, harmful, unhelpful} reduces to a single-state correctness label. The
   study is therefore a *representation × target grid*: correctness is the label, and the question
   is whether the pre→post transition predicts it better than either single state alone.
2. **`oracle_full` is not a panel upper bound.** Rejecting an unhelpful revision restores an
   agent's original wrong answer, which can consolidate a wrong plurality the revision had split.
   The true ceiling (`oracle_panel_optimal`) searches accept subsets per question; with three agents
   that is at most eight combinations. Gate G5 uses it. See `tests/test_oracles.py`.
3. **The panel tie-break is neutral by default.** With ten options and a weak model, three-way
   splits are common, so breaking ties by confidence would quietly turn the panel metric into a
   confidence baseline. `round0_then_confidence` is available as a reported variant.
4. **Generation and activation extraction are separate passes.** Activations can be re-extracted at
   new layers or positions without regenerating a token.

## Storage

Shards are layer-major, `(n_layers+1, n_states, d_model)` in fp16, so reading one layer is
contiguous — ~138 MB for 18,000 states at d=3840, against ~6.8 GB for the full tensor.

**Fixed token positions only.** All layers at a few canonical positions is affordable; one layer at
every token position of every transcript would be ~1.8 TB.
