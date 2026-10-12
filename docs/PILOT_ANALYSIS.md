# Pilot analysis: gemma-3-12b-it, 120 pilot-pool questions (CPU only, 2026-10-12)

Descriptive and exploratory. The pilot pool is excluded from the main test set, and nothing here
tests the activation hypothesis. Data: `runs/colab/` (laptop; review export
`artifacts/pilot12b/data/`). Collection commit `5ff52b3`; see docs/PILOT_LOG.md for lineage.

## 1. Answer outcomes (`scripts/outcomes.py` → `artifacts/pilot12b/outcomes/`)

Panel answer = plurality of three canonical answers, ties by round-0 plurality then agent order.
Counts out of 120 questions; intervals resample questions (10,000 draws).

| | self-reconsideration | debate | answer-only | misleading peer |
|---|---|---|---|---|
| original vote (shared round 0) | 69 | 69 | 69 | 69 |
| final vote | 58 | 67 | 69 | 66 |
| final − original | −0.092 [−0.183, 0.000] | −0.017 [−0.075, +0.042] | 0.000 [−0.058, +0.058] | −0.025 [−0.083, +0.025] |
| questions gained / lost | 11 / 22 | 6 / 8 | 7 / 7 | 4 / 7 |
| vote of all six answers | 72 | 71 | 70 | 72 |
| panel-optimal ceiling | 81 | 78 | 78 | 78 |
| new correct answers (none correct in round 0) | 9 | 1 | 2 | 2 |

A correct first answer was present on 82 of 120 questions. Direct comparisons of final votes:
debate − answer-only −0.017 [−0.075, +0.042] (5 vs 7 questions only one of them got right);
debate − self-reconsideration +0.075 [−0.017, +0.167]; answer-only − self-reconsideration
+0.092 [−0.008, +0.192]; debate − misleading +0.008 [−0.058, +0.075].

**Sensitivity (labelled; the full population is primary):** 18 first answers on 11 questions hit
the 768-token cap (verified from rows; none fell back for another reason). Excluding those 11
questions leaves 109: original vote 65; final self 53, debate 62, answer-only 64, misleading 61
-- the same pattern. All three capped self-reconsideration revisions are on those 11 questions.

**Where revisions happen** (beneficial / harmful / unhelpful revision events; small cells are
noisy):
- Debate revisions occur almost only when first answers split 2-1 (28 / 27 / 30); with unanimous
  first answers there were 4 revisions in all.
- Self-reconsideration's harm is concentrated in unanimous questions: 55 of its 79 harmful
  revisions abandon an answer all three agents gave; the misleading peer breaks unanimous panels
  33 times.
- Debate mostly redistributes answers already present: 31 of its 32 beneficial revisions are on
  questions where some agent was already correct, and 101 of 110 debate revisions adopt a shown
  peer's answer. 9 harmful debate revisions came from an agent that was the only one with the
  correct first answer.
- Self-reconsideration produced 16 beneficial revisions on questions with no correct first answer
  (debate: 1).
- First-answer accuracy differs by chunk -- 0.656 on questions 1-60 (session 2), 0.467 on 61-120
  (session 3) -- different questions, so this is question difficulty, not a session effect.
  In both halves answer-only and debate end above self-reconsideration (revised-answer accuracy
  0.672 / 0.644 / 0.494 and 0.483 / 0.478 / 0.383); the misleading peer is above
  self-reconsideration in the first half and below it in the second.
- Pivotal question ids and a 3-gain / 3-loss debate case study: `outcomes.json`,
  `case_study.md`.

## 2. Exploratory probe analysis (`scripts/reanalysis.py --mode cv` → `artifacts/pilot12b/analysis/`)

**Procedure.** Debate condition: 360 agent events on 120 questions, 110 revisions (32
beneficial, 32 harmful, 46 unhelpful). Five outer folds grouped by question, one seed. Each
family predicts two targets -- *first answer correct* and *revised answer correct* -- from what is
available at a given time:
- **pre**: round 0 only. Output features (option probabilities, margin, entropy, response
  log-likelihood) at the readout position, before the stated answer and at the last generated
  token; activations at the round-0 readout position.
- **exposure**: adds the round-1 prompt with the peers' answers, before any round-1 generation
  (activations at that prompt's readout position).
- **post**: adds the revised answer's state.

Families: `output_linear` (L2 logistic on the output features), `output_hgb` (gradient boosting,
same features), `activation` (L2 logistic on one layer of {12, 24, 31, 38, 44, 48}),
`combined` (that layer plus the output features). Layer and hyperparameters by grouped 3-fold CV
log-loss on the training folds only; monotone Platt calibration fitted out of fold; gate
thresholds tuned on the training folds. Intervals: 10,000 question-level bootstrap draws.

**AUROC, out of fold** -- all 360 events, with the 110 revision events in parentheses (their
intervals are about ±0.13):

| time | target | output_linear | output_hgb | activation | combined |
|---|---|---|---|---|---|
| pre | first answer correct | 0.703 [0.62, 0.78] (0.59) | 0.675 (0.49) | 0.581 [0.49, 0.67] (0.48) | 0.596 (0.49) |
| pre | revised answer correct | 0.734 (0.55) | 0.720 (0.55) | 0.527 (0.51) | 0.547 (0.51) |
| exposure | first answer correct | 0.752 (0.63) | 0.740 (0.61) | 0.727 (0.52) | 0.731 (0.53) |
| exposure | revised answer correct | 0.766 (0.58) | 0.764 (0.60) | 0.750 (0.70) | 0.752 (0.67) |
| post | first answer correct | 0.765 (0.55) | 0.702 (0.50) | 0.658 (0.36) | 0.670 (0.36) |
| post | revised answer correct | 0.792 (0.68) | 0.741 (0.63) | 0.696 (0.66) | 0.706 (0.66) |

- Every family's out-of-fold log-loss is below the constant training-rate baseline (0.697);
  the activation family at pre only barely (0.675 and 0.686).
- Activation − output_linear, all events: pre −0.12 [−0.21, −0.03] (first answer) and
  −0.21 [−0.29, −0.13] (revised answer); exposure −0.03 [−0.09, +0.04] and −0.02 [−0.08, +0.05];
  post −0.11 [−0.17, −0.04] and −0.10 [−0.16, −0.03]. On revision events the largest
  differences are exposure, revised answer, +0.12 [−0.06, +0.30], and post, first answer,
  −0.20 [−0.35, −0.05].
- 28 of 30 activation fits chose C = 1e-4, the strongest regularization in the grid
  {1e-4, …, 1e-1} (288 training events, 3,840 dimensions); at pre the activation scores are nearly
  constant (SD 0.05–0.08). A wider grid or a reduced representation was not tried.
- Post-time first-answer correctness on revision events is *below* 0.5 for activation and
  combined (0.356 [0.24, 0.50]). A simpler explanation than anything about the
  representation: 250 of the 360 training events are unrevised, where the two labels are identical,
  so a model trained on all events can learn "this state looks correct". On a revision with one
  correct side the labels are opposite, so that signal inverts. The within-question AUROCs below 0.5
  for revised-answer correctness at pre (0.31–0.35, 48 pairs) plausibly have the same cause.

**Gates** (panel correct out of 120; accept all = debate's final vote 67; reject all = original
vote 69; panel-optimal 78):

| time | rule | output_linear | output_hgb | activation | combined |
|---|---|---|---|---|---|
| pre | difference | 68 | 65 | 64 | 64 |
| pre | protect | 64 | 61 | 63 | 63 |
| exposure | difference | 67 | 65 | 66 | 65 |
| exposure | protect | 64 | 62 | 63 | 63 |
| post | difference | 67 | 69 | 68 | 68 |
| post | protect | 63 | 65 | 64 | 64 |

No gate beats keeping the first answers; the best ties it (post, output_hgb, difference rule).
Seven of the twelve protect gates are below reject-all with intervals excluding zero.
Activation − output_linear on the primary gate: −0.033 [−0.075, 0.000] (pre),
−0.008 [−0.058, +0.042] (exposure), +0.008 [−0.025, +0.050] (post).

**Transfer** (debate-trained models applied unchanged): on self-reconsideration (181 revisions;
accept all 58, reject all 69) every gate is at or below reject-all, and the accurate-looking ones
accept few revisions (post combined difference gate: 68, accepting 11 of 181). The output models'
revised-answer AUROC on self-reconsideration revisions falls to 0.45–0.49 at pre and exposure
(activation 0.52 and 0.61). On answer-only (103 revisions; accept all = reject all = 69), gates
score 65–70.

**Selecting among first answers** (`scripts/selection_baselines.py` →
`artifacts/pilot12b/selection/`; no round-1 generation is needed). With pre predictions (round-0
information only): original vote 69; vote of all six answers 71 (3 gained, 1 lost); vote of the
three first answers weighted by predicted correctness 70–71 for every family (1–2 gained, none
lost); the single most confident first answer 68–69. With exposure predictions (round-1 prompt,
no generation), output_hgb's most confident first answer reaches 73 (+0.033 [+0.008, +0.067]);
that is one of 24 rule × family × time comparisons, unadjusted.
A cheaper explanation for the weighted vote's gains: the plain vote breaks the 9 three-way
first-answer splits by agent order, and agent 0 was correct on none of them (the correct answer
was present on 5). A random tie-break would expect 5/3 ≈ 1.7 of those right. The weighted vote's
changed decisions at pre were all three-way splits but one (a 2-1 overturned), and its +1 to +2 is
within that.

**Method change, before and after.** These numbers use log-loss selection and monotone
calibration (preregistration deviation log, 2026-10-12). The first runs used mean per-fold AUROC
for selection, which picked near-constant models: pre output_linear first-answer AUROC was 0.534;
it is 0.703 now. Other cells moved by −0.08 to +0.06 (e.g. pre activation revised answer
0.595 → 0.527; exposure activation revised answer on revision events 0.641 → 0.698). No gate beat
the original vote under either method. Old outputs: `artifacts/pilot12b/superseded/`.

**Scope.** One seed, 120 questions, 32 beneficial and 32 harmful debate revisions; the method
was fixed after looking at these runs, and the pilot pool is excluded from the main test set.
Nothing here tests the activation hypothesis. As measured: over all states, output features
predict correctness as well as or better than a single activation layer; on revision events,
no predictor separates beneficial from harmful well enough for a gate to beat keeping first
answers at this n; the original vote is 9 questions below the panel-optimal ceiling.
