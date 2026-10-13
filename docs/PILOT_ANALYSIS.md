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

## 3. Probe search (`scripts/probe_search.py` → `artifacts/pilot12b/probe_search/`)

Requested after §2: try reduced-dimension probes and keep trying until a probe reliably predicts,
or the activations look uninformative. Exploratory and on the pilot pool only; how the search
could fool itself is part of the result.

**What changed from §2.**
- One correctness probe is trained on every answer state: 360 round-0 states plus 1,440 revised
  states from the four conditions (1,800 states, folds by question), not 360 debate events.
- Reduced dimensions: a mean-difference direction, PCA (8–256) then logistic regression, and
  shrinkage LDA (dropped: slow, no better). Optional transforms: *letter* (subtract the training
  mean of states with the same answer letter -- the readout is near-deterministic, mean P(own
  letter) 0.995, so its state is dominated by which letter comes next) and *panel* (subtract the
  panel's mean, label-free).
- Metrics that never compare scores from different fold models: per-fold AUROC; **within-panel**
  AUROC (pairs of the three agents answering the same prompt); **revision AUROC** (score after −
  score before, beneficial vs harmful, within each condition, averaged). An earlier pooled "pair
  accuracy" was dropped: a probe that merely scores revised states lower gets 0.60 because 196 of
  the 325 one-sided revisions are harmful.
- A stronger output-only comparator: letter-invariant output summaries, verbalized confidence,
  per-letter log-probabilities, **panel agreement** (how many other agents gave my answer), shown-
  peer agreement, whether the agent revised, and **option text** (length, relative length,
  longest-option flag, positions). Logistic (`out+text_lr`) and boosted (`out+text_hgb`).

**Measurements** (out of fold; intervals: 2,000 question-level bootstrap draws).

1. *A broad search overfits.* Screening 11,766 configurations (49 layers × 4 positions × 4
   transforms × 5 methods × 3 CV seeds) gives best means of about 0.78 (all states), 0.77
   (within panel) and 0.80 (revisions). Nested CV over 896 single-layer candidates -- selection
   redone inside each outer training fold -- gives, on the outer questions, 0.64–0.73, 0.62–0.71
   and 0.59–0.69 (two outer seeds), while the chosen candidates' inner estimates were 0.77–0.85.
   With 120 questions, picking one layer from hundreds mostly picks noise.
2. *Layer ensembles fix this.* Averaging fold-standardized scores over a band of layers needs no
   layer choice. Nested CV over 48 ensembles (4 positions × raw/letter × mean-difference/PCA16 ×
   layers 12–24, 21–33, 15–39), three selection criteria × three outer seeds: within panel 0.70–0.73
   and revisions 0.69–0.74 averaged over seeds (single runs 0.66–0.76 and 0.63–0.81);
   the selections converge on the position before the stated answer letter, letter-centered,
   PCA16, layers 21–33.
3. *A recipe declared before those runs* -- readout position, letter-centered, mean difference,
   layers 15–39 averaged (chosen a priori for the template-fixed position and the simplest method;
   the band avoids picking a peak) -- on six CV partitions:

   | | all states | within panel | round-0 panels | revisions | debate revisions |
   |---|---|---|---|---|---|
   | activation (declared recipe) | 0.74–0.77 (lower ≥ 0.69) | 0.73–0.78 (≥ 0.63) | 0.60–0.73 | 0.71–0.75 (≥ 0.59) | 0.68–0.73 (≥ 0.51) |
   | out+text_lr | 0.78–0.82 | 0.68–0.73 | 0.56–0.67 | 0.66–0.70 | 0.55–0.61 |
   | out+text_hgb | 0.74–0.80 | 0.68–0.72 | 0.52–0.75 | 0.66–0.73 | 0.59–0.71 |

   Round-0 panels have only 62 mixed-label pairs (intervals about ±0.15).
4. *Controls.*
   - **Option text alone** (no model state): within panel 0.62–0.65, round-0 panels 0.69–0.71,
     revisions 0.61–0.63 -- the correct option is the longest one for 43% of correct states vs
     25% of incorrect ones (a known MMLU-Pro artifact).
   - Regressing the option-text features out of the activations (training rows only) leaves the
     probe at 0.72–0.73 within panel and 0.68–0.69 on revisions; on 90 within-panel pairs matched
     for answer length (same longest-option status, relative length within 0.5 SD) the probe
     scores 0.78–0.79, out+text_lr 0.70–0.71, out+text_hgb 0.70–0.78.
   - Regressing out **all** output-only features: all-state AUROC falls to 0.54–0.58 and within
     panel to 0.58–0.60, but revisions stay at 0.67–0.68 (lower bounds 0.58–0.59; debate
     0.72–0.73, lower bounds 0.60–0.61).
   - Leave one condition out (its revised states never trained on): revision AUROC on it is
     unchanged -- debate 0.69/0.69, 0.73/0.72, 0.71/0.72 (trained with / held out); answer-only,
     self-reconsideration and misleading-peer likewise within 0.03.
   - Layer profiles are smooth (rising to layers ~21–30, then a gentle decline); layer 0 scores
     exactly 0.50. At the last generated token, layer 3 already reaches 0.72 (all states) -- that
     position reads the surface of the answer text -- whereas the readout position, a fixed
     template token, is near chance in early layers and gains with depth.
5. *Beyond outputs.*
   - Stacked logistic model (activation score + output features + option text) vs out+text_lr:
     within panel +0.01 to +0.03 (lower bounds ≈ 0.00), revisions +0.01 to +0.02.
   - Equal-weight sum of the fold-standardized activation and out+text_lr scores (no fitting;
     **post hoc** -- proposed after seeing that the stacker dilutes the revision signal) vs
     out+text_lr: within panel +0.05 to +0.09 (lower bounds −0.00 to +0.04; above zero in four of
     six partitions), revisions +0.05 to +0.08 (lower bounds −0.02 to +0.01). Against
     out+text_hgb the same sum gains +0.00 to +0.06 within panel and +0.03 to +0.05 on revisions,
     all intervals including zero (six partitions).
6. *Panel decisions* (debate; rules fixed before scoring, no thresholds; six partitions; original
   vote 69 of 120). Best of all six answers by score: activation 71–75 (vs original: +0.017 to
   +0.050; lower bounds −0.017 to +0.008, above zero once), out+text_lr 68–70, out+text_hgb
   69–72. Highest-scoring first answer: activation 69–74, outputs 68–72. Gate revisions by
   score(after) > score(before): every scorer 68–71. (`combination.json`.)

**Interpretation, with the cheaper explanations.**
- The activations reliably contain information about whether an answer is correct: a simple,
  low-variance probe ranks states, agents within a panel, and the two sides of a revision at
  AUROC ≈ 0.72–0.77, stable over six partitions, transferring to an unseen condition, and not
  explained by answer length. "No useful information" is ruled out for this pilot.
- Whether that information goes *beyond* output-only information is not established at 120
  questions. For overall correctness the outputs (with panel agreement) are as good or better;
  for ranking agents within a panel and for the direction of a revision, adding activations helps
  by roughly +0.03 to +0.09 AUROC depending on the comparator, with intervals at or just above
  zero. The residual after removing all linear output information still ranks revisions at about
  0.67, which is the strongest single hint that something non-output is there.
- Within-panel pairs always involve two different chosen options, so a probe that scored the
  *option* (how plausible the model finds it) rather than the *agent's state* would look the same;
  correctness labels cannot separate the two. For gating, either serves.
- Post-revision states attend to the peers' answers; linear agreement is in the comparator and
  is regressed out above, but nonlinear uses of peer context remain possible.
- The declared recipe was fixed after the screen (a mild selection); the nested estimate over 48
  ensembles (item 2) is the honest bound for the search.

**For the main run** (decisions for the preregistration, not made here): if this is to be the
primary probe, specify it now -- pooled all-state correctness probe, layer-band ensemble,
letter-centered, at the before-answer or readout position -- with within-panel and revision
AUROC against an output comparator that includes panel agreement and option text. At 120
questions the interval on such a difference is about ±0.04 to ±0.07; it shrinks roughly as
1/√n, to about ±0.02 to ±0.03 at 600 questions, enough to resolve a +0.05 difference if it is real. Panel accuracy
remains underpowered.
