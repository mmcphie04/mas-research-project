# Milestone A — a correctness probe on real activations

**Run 2026-09-10.** `runs/probe-arcc`, reproduce with:

```bash
uv run mad collect --config configs/base.yaml configs/probe_arc.yaml \
  --variants 3 --set run_id=probe-arcc data.dataset_revision=ARC-Challenge data.n_questions=400
uv run python scripts/probe_real.py runs/probe-arcc
```

## Setup

| | |
|---|---|
| model | `google/gemma-3-1b-it`, fp32, CPU (26 layers, d=1152) |
| benchmark | ARC-Challenge test, 4 options, 400 questions |
| states | 1,200 = 400 questions x 3 permutation variants |
| extraction | residual stream at the token after `Final answer: (`; no generation |
| model accuracy | 0.50 — chosen for maximum label variance |
| probe | L2 logistic on the 1,152-dim vector, C by grouped 5-fold CV on question id |
| throughput | 3.56 s/state (was 11.31 before `torch.set_num_threads`) |

This is **not** the experiment. No debate, no peers, no revisions, no gating. It answers one
question: does a linear probe on real activations beat the model's own forced-choice logit
confidence at predicting whether its answer is correct?

## Result: yes, by a small but significant margin

| signal | marginal AUROC (95% CI) | conditional |
|---|---|---|
| logit confidence | 0.6506 [0.6131, 0.6888] | 0.6236 |
| **linear probe (layer 26)** | **0.6978 [0.6546, 0.7352]** | **0.6593** |
| mass-mean probe | 0.6791 [0.6402, 0.7163] | 0.6291 |
| probe + confidence combined | 0.6956 [0.6533, 0.7332] | 0.6566 |

**Δ (probe − confidence) = +0.0472 [+0.0130, +0.0803]**, cluster bootstrap over 400 questions.
The interval excludes zero.

At 100 questions the same effect was +0.057 [−0.018, +0.135] — not significant. The effect size
barely moved; only the interval shrank. A useful calibration on how much data these comparisons need.

### Confidence adds nothing on top of the probe

Probe alone 0.6978, probe + confidence 0.6956 — identical within noise. The probe appears to
subsume the logit-confidence signal rather than complement it. That is a stronger statement than
"the probe wins," and it is the version worth testing at scale.

### It is not merely reading the answer logits

The winning layer is the last one, where the probe sees exactly what the logits are computed from,
so the comparison could be trivial. It is not — the advantage survives well before the output:

| layer | depth | probe AUROC | Δ vs confidence | |
|---|---|---|---|---|
| 15 | 58% | 0.6663 | +0.0156 [−0.0239, +0.0526] | not significant |
| 18 | 69% | 0.6789 | +0.0283 [−0.0077, +0.0654] | not significant |
| **20** | **77%** | **0.6885** | **+0.0379 [+0.0032, +0.0724]** | **beats confidence** |
| 23 | 88% | 0.6930 | +0.0424 [+0.0094, +0.0769] | beats confidence |
| 26 | 100% | 0.6978 | +0.0472 [+0.0145, +0.0817] | beats confidence |

Significance starts around 77% depth. The advantage does grow monotonically toward the output,
so part of it is proximity to the decision — but not all of it.

### Calibration is where the probe is far ahead

| | ECE |
|---|---|
| logit confidence | **0.4674** |
| probe, raw | 0.0602 |
| probe, Platt-scaled | 0.0520 |

Roughly an 8x difference. This may matter more than the AUROC gap: confidence-weighted voting and
the two-probe gating rule both compare *probabilities*, so a well-ranked but badly calibrated
signal is much less useful than the AUROC alone suggests.

## Controls

- **Label permutation:** 0.4858 ± 0.0343 over 5 shuffles. At chance, as required.
- **Difficulty-detector guard:** conditional AUROC 0.6593 against marginal 0.6978. A probe that had
  only learned which items are hard would sit at 0.5 conditionally.
- **Positional leakage:** probe AUROC for predicting "gold is in slot 0" = 0.5243, essentially none.
  Correctness AUROC within each slot separately: 0.616 / 0.679 / 0.747 / 0.784.

## Position bias is severe, and worth carrying into the main design

Accuracy by gold option slot, 300-state run: **0.702 / 0.457 / 0.587 / 0.225** — a 48-point spread.
**51% of questions changed correctness purely from reordering the options.**

Concretely, on one item ("heat was applied to a substance… which state of matter did it start as?",
gold = Solid) the model answered correctly only when Solid sat in slot A, and was 90–99% confident
on all three orderings including both wrong ones.

Consequences for the main experiment: recording permutations is load-bearing, not hygiene; a single
permutation per question understates answer instability badly; and the permutation-replicate run
currently listed third in the cut order deserves promotion.

## Limits — what this does not show

- 1B model, not 12B. Small-model activations are not evidence about a larger model's.
- Direct answering with no chain-of-thought. Closer to Kadavath et al.'s P(IK) than to the debate
  setting, where the readout follows a generated rationale.
- ARC-Challenge, 4 options — not MMLU-Pro, 10 options.
- Correctness, not revision quality. Says nothing yet about beating the AUC 0.79 behavioural
  baseline for predicting harmful revisions (docs/RELATED_WORK.md §1).

## What it does establish

1. The estimation machinery is sound on real data end to end: grouped CV, cluster bootstrap,
   permutation control, conditional AUROC, layer sweep.
2. Correctness is linearly decodable from Gemma activations, with a clean monotone depth profile.
3. The probe-vs-logit-confidence gap is real but **small (~0.05 AUROC)** — the incremental-value
   claim will need careful statistics, not a large effect.
4. The calibration advantage is large and may be the more durable contribution.
5. ~400 question groups is roughly the floor for resolving an effect of this size.
