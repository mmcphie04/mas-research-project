# Probe audit: corrected evaluation of the pilot probe result (2026-10-13)

Requested review of `7787b68` (the exploratory probe search, docs/PILOT_ANALYSIS.md §3). CPU only,
on the 12B pilot archive (120 questions; collection commit `5ff52b3`). Analysis code: `3857a72`
(after a second correction round, §6; clean tree); frozen scorers for the validation tranche: SHA-256
`8e5785fe…` at `3857a72`. The first corrected run (`3e0eea0`) is kept under
`artifacts/pilot12b/superseded/probe_validation_3e0eea0/`. Code:
(`mad.analysis.candidate_scoring`, `scripts/probe_validate.py`, corrections in
`scripts/probe_search.py`). Exports: `artifacts/pilot12b/probe_validation/` (generated tables in
`summary.md`). No GPU work was run.

**Scope of every number below.** 120 questions from the pilot pool. "Partitions" are six
re-splits of those same questions into five folds (seeds 0, 1, 2, 10, 11, 12) — sensitivity to
the split, not replications. Intervals resample questions (2,000 draws for ranking metrics,
10,000 for decisions) and are conditional on the fitted scores: they do not account for the
screen and earlier choices that produced the two recipes, which were selected after inspecting
this pilot. Nothing here is a confirmatory test.

## 1. What was corrected

| | defect at `7787b68` | correction | test |
|---|---|---|---|
| 1a | Layer ensembles (`ensemble_config`, `ens_scores`) and the `combine` sum standardized each held-out fold by its own mean and SD, so a score depended on the evaluation batch (its conditions, phases, other questions). | Fitted scorers: every statistic from training questions only. Normalizers from **cross-fitted reference scores** inside the training set (k = 5 groups of training questions; a model fitted without group r scores group r); the scorer applied to evaluation rows is refitted on all training rows. Platt calibration on the same references. `combine` retired. | `test_scores_do_not_depend_on_the_evaluation_batch` (ensemble, learned and equal-weight combinations, first-answer selector), `test_normalizers_ignore_evaluation_rows` |
| 1a | `Data.covs` imputed missing covariates with whole-dataset means before residualization. | Imputation values, and which columns get missingness indicators, from training rows only (both in the new library and in `probe_search.transform`). | `test_imputation_and_indicators_are_fitted_on_training_rows` |
| 1b | `rev_auc` pooled `s_post − s_pre` across fold models within a condition (reproducer: per-fold AUROC 1.0, pooled 0.75, and 1.0 after rescaling fold 0). Repeated in the bootstrap and leave-condition-out code. | Helpful vs harmful differences compared only inside one (condition, fold) cell; cells pooled by their pair counts. Cell counts and one-class cells exported. Same rule in the bootstrap, selection, outer evaluation and leave-condition-out. The cross-fold value is kept only as a labelled diagnostic in `probe_search`. | `test_revision_auc_is_invariant_to_rescaling_one_fold` (both modules) |
| 1c | `out+text_lr` used `of`+`ot` only (no per-letter log-probabilities) with C = 1; HGB used all three sets. | Kept for continuity, plus `out_lr[full]`: all output features, C chosen from {0.001, 0.01, 0.1, 1} by grouped inner-CV log-loss on training rows; the activation-augmented logistic model uses the same features plus one activation column and the same search. Exact column names exported (`params_*.json`, `frozen_scorers.json`). | `test_combined_model_matches_output_model_features_and_tuning` |
| 1d | — (held) | Question-grouped folds and training-only letter centering kept; inner/reference folds are drawn over training questions with explicit masks; the loader rejects duplicate state keys and checks round 0 appears once per (question, agent); the cache builder checks every condition shares round 0 (same first answers, same round-0 files). | `test_duplicate_state_keys_are_rejected` |

The C grid was extended from {0.01, 0.1, 1} after a smoke run picked the 0.01 edge in every fold
— a change that strengthens the output comparator and is applied identically to the combined
model.

**Information set.** Both recipes read states *after* the agent wrote its answer: the readout
prompt contains the response, the before-answer token is inside it, and letter centering uses
the chosen letter. They are retrospective candidate scorers, not pre-generation gates.

## 2. Before and after

Earlier statements (docs/PILOT_ANALYSIS.md §3 at `7787b68`) against the corrected measurement.
R1 = `readout_last:letter:meandiff:15-39` (layers 15, 18, …, 39); R2 =
`before_stated_answer:letter:pca16:21-33` (layers 21, 24, …, 33; 109 of 120 questions have every
debate state at that position). Medians over the six partitions (range).

| # | earlier | corrected (natural debate unless stated) | verdict |
|---|---|---|---|
| 1 | Activations "reliably" predict correctness: all-state AUROC 0.74–0.77 | Revised states: R1 0.754 (0.74–0.79), R2 0.763. Round-0 states: R1 0.652 (0.63–0.68), R2 0.672. Matched output model: 0.813 / 0.769; activation − output on round-0 states −0.116 (R1; interval below 0 in 4/6 partitions). | Above chance survives. Not better than outputs for single-state correctness. |
| 2 | Within-panel 0.73–0.78 | Pooled over all conditions' panels, that number mixed populations. Debate: round-0 panels R1 0.637 / R2 0.670 (62 / 50 pairs); revised panels 0.688 / 0.726 (48 / 42 pairs); six candidates per question 0.619 / 0.672 (252 / 206 pairs, 33 / 27 questions). | Narrower than stated; round-0 panel ranking ≈ output model. |
| 3 | Revision AUROC 0.71–0.75; debate 0.68–0.73 | Fold-stratified, debate: R1 0.716 (0.69–0.73), R2 0.782 (0.70–0.82); 32 helpful / 32 harmful revisions (R2: 27 / 29), 221 / 166 within-cell pairs. | Survives. |
| 4 | Equal-weight activation + output sum: +0.05 to +0.09 within panel, +0.05 to +0.08 on revisions | Against the same weak comparator (`out_lr[of+ot]`): revised panel +0.167 (interval above 0 in 4/6), revisions +0.116 (3/6). Against the matched, tuned `out_lr[full]`: revised panel **+0.125 (3/6)**, revisions +0.056 (0/6), six candidates +0.004, round-0 panel 0.000. | Survives only for ranking within the revised debate panel. Much of the earlier size reflected a weak comparator. |
| 5 | Stacked model adds +0.01 to +0.03 | Cross-fitted learned combination with matched features and tuning: in the folds inspected the activation coefficient ranks 4th–7th of 115, but its scores correlate 0.998 with the output model's and every panel decision is identical. | Withdrawn as stated: the learned combination gives little practical improvement (its activation coefficient is not small, but the combined scores track the output model's). |
| 6 | Best of six by activation 71–75 vs 68–72 by outputs, 69 original vote | Selection inside a question is unchanged by normalization: R1 71–75. Pooled-six vote 71; matched output selector 69–72. Activation − pooled six: wins/losses 4/0, 0/0, 2/1, 2/0, 3/0, 2/1. Activation − matched output: 6/2, 2/2, 4/4, 4/3, 6/3, 6/3. R2 (109 questions): 66–68 vs pooled six 66. | Counts survive; the gain over pooled-six voting is 0–4 questions. |
| 7 | Residual after removing all linear output information still ranks revisions at 0.67–0.68 | `act_rout` (R1, residualize then letter-center): debate revisions 0.704 (0.65–0.79). After the **complete** transform its held-out score is not linearly predictable from the outputs (R² −0.22, largest single-feature \|r\| 0.28, range 0.19–0.59). For comparison, the ordinary activation score has held-out R² 0.82 from the output features. | Measurement survives, interpretation narrowed: one tested linear predictor (least squares on all output features, fitted on training references) explained none of the held-out variance (R² −0.22; fold range −1.93 to −0.03). That is one predictor's failure, with wide fold-to-fold variation, not proof that the activations carry information unavailable from the outputs. |
| 8 | Leave-one-condition-out: unchanged | Fold-stratified, debate R1 0.716 → 0.711, R2 0.782 → 0.782 (trained with → held out); other conditions within −0.02 to +0.05. | Survives. |
| 9 | Option text alone within-panel 0.62–0.65 | Within-panel pairs come from one fitted model; unaffected. | Stands. |
| 10 | Broad nested search: inner 0.77–0.85, outer 0.59–0.73 | Within-panel figures unaffected; its revision figures used the cross-fold metric. | Within-panel stands; revision figures superseded. |

## 3. Corrected comparison (natural debate primary)

Medians over partitions (range) — `summary.md` has every scorer, population and count.

| scorer | round-0 panel | revised panel | six candidates | helpful vs harmful |
|---|---|---|---|---|
| R1 activation | 0.637 (0.58–0.66) | 0.688 (0.62–0.71) | 0.619 (0.59–0.63) | 0.716 (0.69–0.73) |
| R2 activation (109 q) | 0.670 (0.64–0.74) | 0.726 (0.69–0.76) | 0.672 (0.64–0.68) | 0.782 (0.70–0.82) |
| out_lr[full] (R1 population) | 0.653 (0.60–0.71) | 0.573 (0.50–0.62) | 0.645 (0.63–0.69) | 0.646 (0.62–0.71) |
| out_hgb[full] | 0.613 (0.58–0.73) | 0.625 (0.60–0.65) | 0.633 (0.59–0.69) | 0.660 (0.59–0.70) |
| eq[R1 + out_lr_full] | 0.653 (0.63–0.66) | 0.698 (0.67–0.75) | 0.655 (0.64–0.67) | 0.713 (0.68–0.74) |
| comb_lr[full + R1] | 0.653 (0.60–0.69) | 0.604 (0.54–0.67) | 0.643 (0.62–0.70) | 0.665 (0.63–0.71) |

Paired differences against `out_lr[full]` (median; partitions whose interval excludes 0):
R1 activation: revised panel +0.115 (0/6), revisions +0.061 (0/6), six −0.048, round-0 panel
−0.032. R2: revised panel +0.179 (0/6), revisions +0.123 (0/6), six +0.015, round-0 panel +0.040.
None of these four differences has an interval excluding 0 in any partition at this n.

- **Other conditions (secondary).** The activation advantage on revisions is largest with the
  misleading peer (R2 +0.165, interval above 0 in 4/6; R1 +0.082) and reversed for
  self-reconsideration (R1 −0.115, R2 −0.053). Answer-only resembles debate.
- **Wrong-to-wrong changes** (46 in debate; 38 for R2's eligible questions), which a
  helpful-vs-harmful AUROC leaves out: the activation score rises on 21% of them (R1; 0.20–0.24
  over partitions) and 9% (R2), the matched output score on 35%. So a zero-threshold activation
  gate blocks most of them: it accepts 9–11 of 46 (R1) and 0–5 of 38 (R2).
- **R2 eligibility:** 18 debate states lack a before-answer token (capped responses with no
  stated answer), leaving 109 of 120 questions with all six states. Every comparator is
  recomputed on those 109. The all-question fallback uses the matched output score for the
  absent states.
- **Calibration** (debate revised states, Platt on training references): Brier R1 0.202, matched
  output 0.184, constant 0.248; ECE 0.07–0.08. Standardized scores are not probabilities; only
  the Platt outputs are reported as such.
- **Alignment diagnostic:** held-out z SD 0.98 (0.80–1.09) for single scorers — the refitted
  model's scale matches its training references.

## 4. The practical comparison

Natural debate, R1, all 120 questions (`decisions_R1.csv.gz` has every selected answer).
Ties use the preregistered tie-break (round-0 plurality, then agent order). As a
label-independent sensitivity, the exact expected count under a uniformly random tie-break is
also given. That expectation is the limit of seeded random tie-breaks; no earlier seeded version
existed in the code.

| rule | correct (6 partitions) |
|---|---|
| original vote / revised vote / pooled-six vote | 69 / 67 / 71 (expected under a uniformly random tie-break: 70.7 / 67.0 / 70.2) |
| best first answer: activation / matched output / equal-weight | 69–73 / 69–73 / 70–71 |
| best of six: activation / matched output / equal-weight | 71–75 / 69–72 / 71–72 |
| gate (accept iff score rises): activation / matched output | 70 (all) / 69–72 |
| one validation-tuned activation gate (secondary) | 68–72 (fold thresholds −1.4 to +1.4, mostly negative: unstable) |

- **Where the gains come from.** Best-of-six by activation beats the same scorer's best first
  answer by 0–2 questions (wins/losses 2/0, 2/0, 2/0, 2/0, 1/0, 1/1). 74 of its 120 picks are
  revised states, but only one is an option absent from round 0 (not correct). Debate added a
  correct option missing from round 0 on **1 of 120** questions. Choosing among debate's six
  candidates and among answer-only's six gives the same answer on 118–119 of 120 questions,
  although their revised answers differ on 39. The apparent gain is better selection among
  round-0 options, not something debate contributed.
- **Gate behaviour** (partition 0): the activation gate accepts 12 of 32 helpful revisions,
  blocks 26 of 32 harmful ones and 36 of 46 wrong-to-wrong changes; the matched output gate
  accepts 22 helpful, blocks 24 harmful and 33 wrong-to-wrong. Neither is consistently above
  pooled-six voting (median difference −1 question for the activation gate, 0 for the output gate).
- **R2** (109 eligible questions): original 65, pooled six 66, best of six by activation 66–68,
  matched output 64–67, activation gate 64–65. With the all-question fallback (output score where
  the before-answer state is absent), R2 best of six is 71–73 on 120.
- **Reconsideration pools (secondary).** Six-candidate coverage: debate 83/120, misleading 84,
  answer-only 84, self-reconsideration 91 (9 questions where only a reconsidered answer is
  correct). Best of six by R1 activation: self 70–74, misleading 71–74 vs pooled six 72 in both.

## 5. What survives, stated narrowly

- An activation score from a fixed recipe ranks answer states by correctness above chance on
  held-out pilot questions, and it transfers to a condition left out of training.
- Against a strong, matched output model, its advantage is confined to ranking among an agent's
  revised states within a debate panel and to the direction of a revision (+0.06 to +0.18
  AUROC, intervals including 0 in every partition at this n). For overall correctness and for
  ranking the six candidates of a question it is not better, and a learned combination gives
  little practical improvement over the output model.
- In end-to-end decisions, selecting by activation score matches or slightly exceeds pooled-six
  voting (0–4 questions of 120) and the matched output selector (median +2), and almost all of
  that comes from choosing among answers already present in round 0.
- Cheaper explanations remain open: two answers in a panel are always different options, so a
  score of the *option's* plausibility looks the same as a score of the agent's state; and the
  revision advantage concentrates where a peer endorses a wrong answer, which could reflect
  endorsement sensitivity of the output features rather than knowledge in the activations.

Next step: docs/NEXT_STEP_MEMO.md. GPU specification: docs/VALIDATION_JOB.md.

## 6. Second correction round (`3857a72`, 2026-10-13)

Requested after review of `f7a8f4d`; recipes, seeds and populations unchanged.

- **A. Frozen application on any dataset size.** Feature columns are now validated by schema
  (names only). The old check built the combined scorer's columns from the saved, pilot-length
  training references, which fails whenever the new dataset has a different number of states. The
  planned 200 × 9 = 1,800 states only coincides with the pilot's 120 × 15. Evaluation activation
  columns always come from the frozen scorers applied to the new states. `freeze` strips the
  training-length arrays, so prediction cannot depend on training rows. End-to-end test
  (`tests/test_frozen_apply.py`): stub archives through the real cache builder, frozen on 12
  questions; the training cache is then deleted, and the scorers are applied to 8 questions (only
  `debate` and `independent`, absent before-answer states) and to a complete-chunk partial archive.
- **B. Validation endpoints** in `probe_validate.py apply` (docs/preregistration.md §10).
- **C. Training-only inner preprocessing.** Imputation and standardization are now fitted inside
  each tuning and reference fold, and C is chosen on those folds by log-loss, identically for the
  output-only and activation-augmented models. The combined scorer's internal reference
  predictions are now pure: for each reference fold the activation column of the other rows comes
  from an activation scorer refitted without that fold, so the held-out fold's labels cannot reach
  its training features through the upstream probe. (Outer-test predictions were never affected;
  this concerned the internal references used for normalization and calibration.)
- **Effect of C on the bounded comparison:** ranking metrics and decisions are identical for every
  scorer, because the final models use the same C on the same rows, except R2's learned
  combination (round-0 panel median 0.660 → 0.640; one partition's best-of-six 67 → 66). Only the
  normalizers and Platt calibrations moved.
- **D.** Rows 5 and 7 above and §5 reworded (learned combination: little practical improvement;
  residual score: what the tested linear predictor explained, not a proof of unique information).

