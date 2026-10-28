# Probe training audit, action choice, and transcript review (2026-10-28)

CPU only; no GPU work was run for this report. **Everything here is exploratory.** The 120 pilot
questions trained the probes, and the 200 validation questions were analyzed before this work
(docs/VALIDATION_RESULTS.md, docs/R1_INVESTIGATION.md). Nothing below confirms anything, and no
estimator is chosen as a winner.

Code and outputs:

| part | code | outputs |
|---|---|---|
| 1. probe audit | `scripts/probe_audit.py` (`cv`, `validation`, `curves`, `stability`, `paired`) | `artifacts/val12b/probe_audit/` |
| 2. action choice | `scripts/action_choice.py` | `artifacts/val12b/action_choice/` |
| 3. transcripts | `scripts/transcript_sample.py`, plus hand coding | `artifacts/val12b/transcript_review/` |

The outputs hold no activations. `validation_scores.csv.gz` holds only one scalar score per
state per estimator.

Bug found and fixed along the way: `evaluate_scores` raised a KeyError when asked to evaluate a
subset of questions. Rows outside the subset were already masked; only the index lookup failed.
It now returns −1 for those rows. Results on the full question set are unchanged.

## Summary

1. **Probe audit.** No estimator reliably beats the matched output-only model on the 200
   validation questions.
   - Across 27 paired comparisons with the output model (9 activation estimators × 3 training
     sets):
     - six-candidate ranking (AUROC) differences run from −0.10 to +0.05;
     - revision-ranking differences run from −0.12 to +0.11;
     - every 95% interval includes zero.
   - The scores are stable from one training sample to another (correlation 0.79–0.96 between
     refits). The thing that hurts debate selection is a *systematic* shift that depends on the
     peers' context, not random instability.
   - More training questions helps the logistic-type estimators. Compression (PCA) does not
     help, and rescaling the PCA components adds noise.
   - One consistent but post-hoc pattern: picking among the three *first* answers. It is
     described in §1.4 with its caveats.
2. **Action choice.** The 200 questions cannot support a learned stop / independent / debate
   policy.
   - Only **10 of 200** questions get a different final answer depending on the action.
   - Always adding three independent answers gives 120 correct, against 116 for stopping.
   - Learned policies also reach 120 on every seed.
   - Picking the best action with hindsight reaches only 122.
3. **Transcripts.** Revisions mostly follow the peers, and checking is rare.
   - Over all 600 debate revisions:
     - an agent switches in 43 of 46 cases where both visible peers disagree with it (16 helpful,
       15 harmful);
     - it switches in 58 of 89 cases where the peers are split;
     - 489 of 600 revisions contain praise or deference wording.
   - In a hand-coded sample of 24 revisions:
     - an identifiable decisive point of disagreement was engaged in 3 of 10 cases;
     - 8 of the 12 calculation questions accepted a wrong number without recomputing it.
4. **Smallest next GPU experiment:** the fixed-candidate context readout (§4). It needs readout
   only, no generation, and under 0.5 GPU-hour.

---

## 1. Probe training audit

### 1.1 What was compared

**Fixed representation.** The audit uses R1's position (`readout_last`, letter-centered and
standardized) at layers 21, 27 and 33. Each layer gets its own score; the scores are normalized
using training-only references and averaged. No other layers or positions were searched.

**Estimators.** Each is fitted per layer.

- `meandiff`: the current R1 estimator, mean(correct) − mean(incorrect).
- `lr`: L2 logistic regression with C ∈ {1e-4, 1e-3, 1e-2, 1e-1}.
- `pca16/64/128+lr`: PCA, then each component rescaled to unit variance, then logistic
  regression.
- `pca64raw+lr`: PCA without rescaling. It tests whether rescaling amplifies noise.
- `pairwise`: logistic regression on within-question differences (correct minus incorrect),
  with no intercept and every question given equal total weight.
- `pca64+pairwise`: the same on 64 rescaled components.
- `wq_avgdiff`: the average within-question difference from docs/R1_INVESTIGATION.md. This is
  a direction, not a fitted ranking model, and is kept separate from `pairwise`.

**Matched output-only baselines.** The same estimators (`meandiff`, `lr`, `pairwise`,
`wq_avgdiff`) on the full output features (`of_` + `ol_` + `ot_`).

**Fitting rules.**

- C is chosen by grouped 3-fold inner cross-validation over **training questions only**. All
  states of a question stay in the same fold. Log-loss picks C for `lr`; pairwise log-loss picks
  it for `pairwise`.
- Letter centering, standardization, PCA and imputation are fitted on training rows only.
- Platt calibration is fitted on the inner out-of-fold scores.

**Training sets.**

| training set | pilot conditions included |
|---|---|
| originals | round 0 |
| originals+debate | round 0 + debate |
| all_pilot_conditions | adds answer-only debate, misleading-peer and self-reconsideration |

**Informative questions** (those with at least one correct and one incorrect state):

| set | informative questions | per CV training split |
|---|---|---|
| originals | 31 of 120 | 23, 28, 25, 22, 26 |
| originals+debate | 33 of 120 | 24, 30, 26, 24, 28 |
| all_pilot_conditions | 74 of 120 | 61, 56, 59, 60, 60 |

On validation, 44 / 49 / 68 / 71 of 200 questions are informative for originals / +debate /
+independent / all states.

**Evaluations.**

- **(a) Pilot cross-validation:** 5 grouped folds; held-out vs in-sample performance on the
  training questions (`cv.json`).
- **(b) Validation:** fit on the whole pilot, score the 200 validation questions
  (`validation.json`, `validation_scores.csv.gz`).
- **(c) Paired comparison:** each activation estimator minus the matched output estimator, same
  training set, 2,000 shared question-bootstrap draws (`paired.json`).
- **(d) Learning curves:** 24 / 48 / 72 / 96 / 120 random pilot questions (all conditions),
  5 repeats each, evaluated on validation (`learning_curves.csv`).
- **(e) Stability:** 10 refits per estimator on bootstrap resamples of the pilot questions,
  scored on validation (`stability.json`).

### 1.2 Train vs unseen performance

**Training fit vs held-out performance.** Mean six-candidate AUROC on the debate population; the
first number is in-sample on training questions, the second is held-out in pilot CV:

| estimator | training fit → held-out (pilot CV) | validation |
|---|---|---|
| meandiff | 0.61 → 0.60 | 0.50–0.57 |
| lr | 0.74–0.78 → 0.62–0.66 | 0.56–0.61 |
| pca16+lr | 0.65–0.71 → 0.65–0.68 | 0.53–0.56 |
| pca64+lr | 0.75–0.81 → 0.53–0.64 | 0.57–0.62 |
| pca128+lr | 0.85–0.89 → 0.54–0.60 | 0.59–0.63 |
| pca64raw+lr | 0.70–0.73 → 0.62–0.66 | 0.55–0.62 |
| pairwise | 0.81–0.87 → 0.58–0.63 | 0.52–0.55 |
| out:lr | 0.70–0.74 → 0.58–0.68 | 0.54–0.58 |

**What the table shows.**

- **The gap between training fit and held-out performance grows with flexibility.** The mean
  difference has almost no gap; 128 PCA components have the largest. For PCA-128 the
  revised-state AUROC is 0.96–0.98 in-sample against 0.69–0.75 held-out.
- **The best pilot-CV setting did not hold up on validation.** pca16+lr on originals had the
  highest pilot CV result (six 0.68, revision 0.81) but scored 0.56 / 0.61 on validation. This
  is the reason not to pick an estimator by its top exploratory score.
- **The six-candidate intervals on validation are wide:** about ±0.08 to ±0.10 around each
  point.

**Paired against out:lr on validation** (`paired.json`; 27 comparisons per baseline):

- **Six-candidate AUROC:** −0.10 to +0.05. No interval excludes zero.
- **Revision AUROC:** −0.12 to +0.11. No interval excludes zero.
- **Best-of-six correct:** −1 to +7 questions. No interval excludes zero. Only 14–24 of 200
  questions pick differently between the two scorers.
- **Against out:pairwise:** 1 of 27 six-candidate intervals excludes zero, about what chance
  predicts for 27 tests.

### 1.3 Stability, and instability vs systematic shift

These come from 10 bootstrap refits per estimator (`stability.json`).

**Validation score correlation between refits** (median):

| estimator | correlation |
|---|---|
| meandiff | 0.96 |
| wq_avgdiff | 0.95 |
| out:lr | 0.90 |
| pca64raw+lr | 0.90 |
| pca16+lr | 0.89 |
| lr | 0.88 |
| pairwise | 0.88 |
| pca64+lr (rescaled) | 0.79 |
| out:pairwise | 0.79 |

**Selected-answer stability.**

- The best-of-six pick agrees with its most common pick across refits on 94–97% of debate
  questions.
- The number correct still moves by 4–6 questions on debate across refits of the *same*
  estimator (for example pca64+lr 123–129), and by up to 11 on independent (pca64+lr 119–130).
  That is as large as most differences *between* estimators.

**Does rescaling the PCA components amplify noise?** Yes, by these measures. Comparing
pca64+lr (rescaled) with pca64raw+lr:

- **Refit correlation:** 0.79 vs 0.90.
- **Pilot-CV gap between training fit and held-out:** larger with rescaling (0.75–0.81 → 0.53–0.64
  vs 0.70–0.73 → 0.62–0.66).
- **Per-state score SD across refits:** 0.34 vs 0.20.

Rescaling gives low-variance components the same weight as high-variance ones. The validation
point estimates differ by less than the noise.

**Systematic shift vs instability.** For each refit I took the mean revised-minus-first score
change on debate. Across the 10 refits it is:

| estimator | unanimous questions | disputed questions | SD across refits |
|---|---|---|---|
| meandiff | +0.43 | −0.75 | 0.06–0.08 |
| lr | +0.62 | −0.66 | 0.09 |
| pca16+lr | +0.70 | −0.80 | 0.11–0.13 |
| pca64+lr | +0.32 | −0.49 | 0.10–0.12 |
| pairwise | +1.00 | −0.42 | 0.14–0.19 |
| wq_avgdiff | +0.34 | −0.78 | 0.16–0.17 |
| out:lr | +0.31 | −0.16 | 0.07 |

- **The shift is systematic.** It is 2–10 times larger than its variation across refits, and
  every activation estimator has it. The pairwise and within-question estimators have it too.
- **It is not just instability.** The per-state change varies far more across states (SD
  0.85–0.96) than across refits (0.11–0.34).
- **Independent answers show no shift** (|mean| ≤ 0.04 for every activation estimator). The
  output models show a smaller disputed shift (−0.11 to −0.17).

This agrees with docs/R1_INVESTIGATION.md: it is the peer context that moves the scores, not the
choice of estimator.

**Reliability.** Platt-calibrated scores on the 600 revised debate states, compared with a
constant at the training base rate (Brier 0.240):

| estimator | training set | Brier | ECE |
|---|---|---|---|
| activation (all estimators) | originals | 0.22–0.23 | mostly 0.14–0.15 |
| activation (all estimators) | includes debate | 0.18–0.20 | 0.04–0.09 |
| out:lr | any | 0.21 | 0.07–0.11 |

pca64+lr is the exception among originals-trained models, with ECE 0.06.

A model trained only on first answers is miscalibrated on revised answers, which fits the
context shift. Including debate states in training fixes the calibration. It does not fix the
ranking.

### 1.4 The one consistent pattern: choosing among the three first answers

**What was measured.** "Best first" picks the highest-scored of the three *first* answers, so
there is no peer context. The three counts below are for the originals, originals+debate and
all-conditions training sets:

| scorers | correct of 200 |
|---|---|
| activation scorers | meandiff 125/128/126, lr 124/126/127, pca64raw+lr 125/127/126, wq_avgdiff 128/127/122 |
| output scorers | 118–124 |
| plurality vote of the first three (the "stop" action) | 116 |

**Paired against out:lr.**

- The activation estimator is ahead in 24 of 27 comparisons, by −2 to +9 questions.
- 2 of the 27 intervals exclude zero, for example meandiff on originals+debate: +9 [+2, +17].

**Why this is not evidence that activations help.**

- **Preregistered result:** the frozen, preregistered version of this comparison found R1 122
  vs matched output 121 (docs/VALIDATION_RESULTS.md).
- **Post hoc:** the estimators here were refit after the validation data had been seen.
- **Correlated comparisons:** the 27 comparisons are strongly correlated, since they share
  training data, layers and questions. They are not 27 independent findings.
- **A cheaper explanation:** the output-only `lr` may be a weak baseline for this one rule
  (out:pairwise reaches 124 on the all-conditions set).

What it does suggest is a hypothesis: activations may help when no peer context shifts them. A
test of that needs new questions and one estimator fixed in advance.

### 1.5 What looks most promising: compression, regularization or more data?

- **More data** shows the clearest trend. lr and pca64+lr improve steadily as training
  questions go from 24 to 120 (16 → 74 informative):
  - validation six-candidate AUROC: 0.52 → 0.59 and 0.52 → 0.60;
  - revision AUROC: 0.54 → 0.66;
  - variation across repeats falls;
  - over the same range, out:lr stays flat (0.56–0.59).
  - The mean-difference and within-question directions do not improve (about 0.50). They seem
    limited by the context shift rather than by data.
  - Caveats: the gain from 96 to 120 questions is small, and there is one evaluation set.
- **Regularization:** logistic regression chose C = 1e-4, the strongest value on the grid, in
  every fold. The best setting may lie beyond the grid. A wider grid (1e-6 to 1e-4) is cheap
  to try on CPU.
- **Compression** does not help. PCA 16/64/128 does no better than plain, heavily regularized
  logistic regression, and rescaled PCA is the least stable choice.
- **Caution:** none of these changes addresses the systematic context shift, and that shift,
  not estimator noise, is what blocks debate selection.

### 1.6 Not tried

- Other layers or positions. They were held fixed by design to keep the comparison bounded.
- Nonlinear probes.
- Per-question normalization at training time, beyond the label-free correction in the R1
  investigation.
- Any estimator with C below 1e-4.

---

## 2. Choosing the next action (200 validation questions)

### 2.1 Setup

**Actions after the three first answers.**

| action | final answer |
|---|---|
| stop | vote of the 3 first answers |
| independent | add 3 independent answers; vote over all 6 |
| debate | add one debate round; vote over all 6 |
| debate, revised only | sensitivity check: vote over the 3 revised answers |

**Fixed selection rule.** Plurality vote. Ties go to the round-0 plurality, then to agent order.

**Policies** use only information available before the action:

- the visible round-0 features: distinct answers, top count, stated confidence (mean and min),
  generation and stated-answer probabilities, length, cap hits, mean log-probability;
- then those plus the frozen R1 scores of the round-0 states (mean, max, spread, and the
  plurality answer's score).

**How learned policies are fitted.** One logistic model per action predicts P(correct). Fitting
uses 10-fold question-level cross-validation, repeated over 5 seeds.

### 2.2 Results (`action_choice.json`)

| policy | correct | vs stop [95% CI] | fixes / new errors vs stop | extra answers |
|---|---|---|---|---|
| always stop | 116 | — | — | 0 |
| always independent | 120 | +0.020 [−0.005, +0.050] | 6 / 2 | 600 |
| always debate | 116 | +0.000 [−0.020, +0.020] | 2 / 2 | 600 |
| debate, revised answers only | 119 | — | — | 600 |
| unanimous → stop, else independent | 120 | +0.020 [−0.005, +0.050] | 6 / 2 | 210 |
| unanimous → stop, else debate | 116 | +0.000 | 2 / 2 | 210 |
| mean stated confidence ≥ 95 → stop, else independent | 119 | +0.015 [0.000, +0.035] | 3 / 0 | 96 |
| mean stated confidence ≥ 90 → stop, else independent | 118 | +0.010 | 2 / 0 | 51 |
| learned, visible features (5 seeds) | 120 every seed | +0.020 | 6 / 2 | 513–534 |
| learned, visible + internal (5 seeds) | 120 every seed | +0.020 | 6 / 2 | 504–531 |
| *hindsight best action (optimistic reference only)* | *122* | *+0.030* | *6 / 0* | *18* |

**What the table shows.**

- **Learned policies** choose stop on only 11–16% of questions, so they act mostly like
  "always add answers". Adding the internal features changes only the split between
  independent and debate.
- **Distinguishing questions.** Only **10 of 200** questions get a different answer under
  different actions:
  - independent vs stop: 6 gained, 2 lost;
  - debate vs stop: 2 gained, 2 lost;
  - independent wins 6 of the questions where it and debate differ; debate wins 2.
- **By initial agreement (correct under stop / independent / debate):**
  - unanimous first answers: 97/130, identical under all three actions. A pooled vote cannot
    change them, because a 3–3 tie goes to the round-0 answer;
  - two against one: 18 / 19 / 19 of 51;
  - three-way split: 1 / 4 / 0 of 19.

**Is the sample large enough? No.**

- Ten distinguishing questions cannot support a learned choice among three actions.
- Even hindsight action choice gains only 6 questions.
- Every learned policy lands on the same 120 as the simple rule "unanimous → stop, else add
  independent answers", which uses 210 extra answers instead of about 520.
- **Smallest supportable statement:** for disputed first answers, independent sampling was at
  least as good as debate here (net +4 vs net 0 against stopping), at the same cost. The interval for
  independent vs stop includes zero.
- **What it would take:** a real test needs many more disputed questions. Roughly 1 in 20
  questions distinguishes the actions, so about 1,000+ questions would be needed to collect
  around 50 distinguishing ones.

---

## 3. Transcript review

### 3.1 Sample and method

**Sampling.** Reproducible: seed 2026, 6 per category, from the validation debate run and the
capped responses. Category sizes in the population:

| category | population |
|---|---|
| helpful changes (wrong → right) | 53 |
| harmful changes (right → wrong) | 46 |
| unchanged, correct | 307 |
| unchanged, wrong | 124 |
| truncated at the 768-token cap (any state) | 56 |

**What `sample.md` shows.** Each case shows the agent's first answer, the **full** peer text
the agent saw, and its revision.

**Hand coding.** All 30 cases were coded by reading them. Debate cases are in `coding.csv` and
truncated cases in `coding_truncated.csv`. Each row has a note explaining its codes.

**Codebook.**

| code | definition |
|---|---|
| actual check | The response does something new: recomputes a number, substitutes an option back, or tests a claim. Restating a peer's number, or saying it is "correct", does not count. "Partial" means part of the decisive step was checked; "repeat" means the same computation was redone with the same method. |
| asserts correctness without check | Says "the calculation is correct" or "my reasoning was sound" without a check in the same response. |
| accepted unchecked wrong number | A calculation item where the response relies on a number that is wrong (verified by hand) without recomputing it. |
| decisive point present | The responses shown disagree on a specific, checkable step or fact that decides the answer, such as units, a volume, or what an option is about. Recall and interpretation disagreements without such a step are coded 0. |
| answer follows own reasoning | Taken at face value, the response's own reasoning leads to the option it chose. |

**Population counts.** `population_stats.json` gives counts over all 600 revisions that need no
hand coding. Its wording markers measure wording, not behavior.

### 3.2 Counts

**Hand-coded sample** (24 debate revisions; denominators shown):

| pattern | count | where |
|---|---|---|
| changed answer with no new reason (repeats a peer's argument) | 9 / 12 changes | 3 of 6 helpful, all 6 harmful |
| changed to a peer's answer | 12 / 12 changes | 9 to a single peer's answer, 3 to both peers' answer |
| praise or deference toward peers | 18 / 24 | |
| restates a peer's argument | 21 / 24 | |
| actual check performed (calculation items) | 6 / 12 | 2 full, 3 partial, 1 repeat |
| ...of which the check was correct | 4 / 6 | |
| accepted a wrong number unchecked (calculation items) | 8 / 12 | |
| asserts correctness without checking (all items) | 11 / 24 | |
| decisive point present / engaged | 10 / 24 present, 3 / 10 engaged | |
| final answer not supported by its own reasoning | 4 / 24 | harmful-1, unchanged_wrong-1, -4, -5 |
| misstates what a peer said | 2 / 24 | |
| item suspected defective | 2 / 24 | also the item behind truncated-4 |

**All 600 debate revisions** (data-derived):

- **Switching follows the peers.** 169 revisions changed answer, and 150 of those moved to an
  answer a peer had visibly given. Switch rate by what the visible peers said:

  | visible peers | switched | outcome of the switches |
  |---|---|---|
  | both disagree with the agent | 43 / 46 | 16 helpful, 15 harmful, 12 wrong → wrong |
  | split (one agrees with the agent) | 58 / 89 | 28 helpful, 17 harmful, 13 wrong → wrong |
  | both agree with the agent | 4 / 376 | |

- **Praise or deference wording** appears in 489 / 600 revisions. It appears about equally
  often in helpful (42/53), harmful (34/46), unchanged-correct (258/307) and unchanged-wrong
  (98/124) revisions, so the wording does not distinguish them.
- **Written-out arithmetic** appears in 82 of 171 revisions on calculation subjects (math,
  physics, chemistry, engineering). This is an upper bound on checking: writing "= 25,000" is
  not checking it.
- **Agreement raises confidence even when nothing new is said.** When both peers agree with the
  agent, its stated confidence rises by 2.6 points on average and rises in 73% of 372 cases.
- **Capped peers.** In 45 / 600 revisions a peer had hit the token cap, so the agent saw that
  peer's text cut off with no answer line.

### 3.3 Failure patterns, with examples

**Copying an option.**

- unchanged_wrong-4: all three agents compute (−1, 3, 7), which matches no option. The agent
  writes: "As both panelists selected (C), I'll follow suit." A one-line back-substitution
  (2c₁+3c₂+4c₃ = (1,3,5)) gives D, the correct answer.
- harmful-3: the agent picks 25,000 because it is "close to one of the answer choices". The
  peer's arithmetic behind that number is invented: 10 × 0.25 / 0.000316 ≈ 7,900.

**Praising peers.** The typical phrases are "Panel member 2 raises a compelling point" and "The
panel's reasoning reinforces my initial assessment". These appear before both helpful and
harmful switches. harmful-4 calls a peer's point "a crucial point I initially overlooked", but
its own first answer had already weighed that option and rejected it.

**Repeating arguments.**

- All six unchanged-correct cases restate the agreeing peers and raise their confidence from 95
  to 98 or 100.
- harmful-2 and harmful-6 reverse a correct answer by restating a peer's interpretation, with no
  new evidence.

**Failing to check calculations.** These are the most consequential cases.

- unchanged_wrong-1: the agent's own first answer had the right equations and x = 1.5, then
  evaluated Kp = 2.25 / (0.25 × 2.25) as 1 instead of 4. Option J is 4. The revision never
  looks at that step again.
- unchanged_wrong-3: a peer had the right volume, 0.15 cm³. The agent keeps its own 1.5 cm³,
  writing "My initial calculation ... was correct", and misquotes the peer's result.
- helpful-1 lands on the right option through a peer's arithmetic that is off by about 10×. The
  agent's correct-sounding unit argument hides this.

**Overlooking a decisive disagreement.** A decisive point was engaged in only 3 of 10 cases.

- harmful-4: option E is about red-blood-*cell* turnover, but the question asks about cell
  *compounds*. The difference is in the option text, and no agent mentions it.
- helpful-6: all three agents label three options as "correct" in a single-answer question, and
  no one resolves which two are false.

**Truncated responses** (`coding_truncated.csv`, 6 cases):

- 5 / 6 restart the calculation several times.
- 5 / 6 throw away their own result because it does not exactly match an option. truncated-1
  computed the right value, 8.76 × 10⁶, twice and rejected it because the nearest option reads
  8.02 × 10⁶.
- Only 1 / 6 (truncated-6) was making coherent progress and simply ran out of tokens.
- The cap mostly affects first answers and independent answers, which have no word limit (29 / 600
  and 25 / 600). Debate revisions, which have a ~150-word limit, hit it in only 2 / 600.

**Actual checking vs careful-sounding language.** Real checks were rare, and they help only when
done:

- helpful-2 completes a capped peer's Boyle's-law step.
- unchanged_wrong-5 tests all ten options and correctly finds none fits. It then answers D
  anyway, an option its own check had just rejected.

More often the careful-sounding phrases ("the calculations are correct", "the reasoning is
sound", "acknowledging the discrepancy") come *instead of* a check: 11 / 24 responses assert
correctness without checking.

### 3.4 Limits of this review

- **One coder, not blinded.** I knew each case's category while coding.
- **Small sample.** 24 debate cases plus 6 truncated ones. The rates in §3.2 are descriptions
  of this sample, not estimates with useful precision.
- **Wording is not behavior.** The population counts are based on wording.
- **Defective items.** Two items appear defective (no option satisfies the congruences; two
  logically equivalent formulas), and a third looks ambiguous. Their "wrong" labels say little
  about reasoning.

### 3.5 Recommended prompt and discussion-structure changes

These are recommendations only. **They were not implemented or tested.** Each targets a pattern
counted above.

1. **Make the disagreement the task** (decisive point engaged 3 / 10). Before giving a final
   answer, the agent must:
   - name the single step where the answers it saw diverge;
   - redo that step itself;
   - state the result.

   Replace "Reconsider the question in light of these responses" with that instruction.
2. **Allow a switch only when the agent names an error** (9 / 12 changes gave no new reason; 16
   helpful vs 15 harmful switches when both peers disagreed). The agent may change only by
   quoting the specific error in its own previous reasoning. Otherwise it keeps its answer.
3. **Require a check against the options for calculation items** (8 / 12 accepted a wrong
   number; unchanged_wrong-4 and unchanged_wrong-5). The agent should:
   - substitute the chosen option back, or recompute the key quantity;
   - report the computed value next to the chosen letter;
   - when no option matches, say so and pick the nearest option *once*, rather than restarting
     (5 / 6 truncated responses looped for this reason).
4. **Remove the social framing** (praise wording in 489 / 600; switching follows the peer count).
   - Present other answers as anonymous "candidate solutions", not "panel members".
   - Consider showing the reasoning without the answer letters, or showing only the disputed
     step.
   - Do not invite agreement ("You may keep your answer or change it" reads as permission to
     defer).
5. **Fix capping before discussion** (45 / 600 revisions saw a cut-off peer with no answer). For
   first answers, either add the same length guidance as the revision round, or require the
   answer line first and the reasoning after. Never show a peer's truncated text without saying
   it was cut off.
6. **Ask for confidence before the agent sees peers, or drop it** (confidence rises in 73% of
   unanimous cases with no new content). As currently asked, revised confidence mostly
   measures agreement.
7. **Discussion structure:**
   - **Only debate disputed questions.** Under pooled voting, unanimous panels never changed
     the final answer, so debating them adds cost without benefit.
   - **For disputed questions, consider a verifier step instead of symmetric debate.** One agent
     is given the two competing derivations and asked only to find the first wrong step.
   - Independent sampling did at least as well as debate here (§2). Any changed prompt should
     therefore be compared with independent sampling, not only with the current debate.

---

## 4. Smallest next GPU experiment

**The fixed-candidate context readout** from docs/R1_INVESTIGATION.md: readout only, no
generation, under 0.5 GPU-hour on the 200 validation questions.

**Design.** Teacher-force each existing first and revised response into three contexts and read
R1's position (and the audit's lr layers) in each:

1. no peers;
2. peers who agree with it;
3. peers who disagree with it.

**Why this experiment.**

- **It tests the only robust finding in Part 1:** the systematic, context-dependent score shift.
  If the shift appears with the candidate text held fixed, the shift is caused by the context.
- **It shows whether a neutral scoring context helps.** Scoring all six candidates in the "no
  peers" context would show whether that restores within-question ranking. It would also give
  the §1.4 hypothesis (activations help without peer context) a mechanistic check that needs no
  new generation.
- **The alternatives are not supported yet.** The data does not support an action-choice
  experiment (§2). A prompt-revision pilot (§3.5) needs new generation and a comparison with
  independent sampling, so it is larger.

**Not run; requires your authorization.** Its decisions should be written down before running:
the contexts, the layers, and the metrics, which are the shift by context, and six-candidate
AUROC and best-of-six in the neutral context.

**If you later want a confirmation of §1.4 on fresh questions** (pool positions 321+), it needs
only round-0 generation. Fix the estimator now *on principle*, not by score: L2 logistic
regression on standardized features with the C grid extended below 1e-4. Its gap between
training fit and held-out performance was smaller than the PCA and pairwise variants, and it
improved with more data. It should be compared with the matched output model on best-of-first-
three.
