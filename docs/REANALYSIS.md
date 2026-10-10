# Reanalysis under the second review's methodology

> **Correction (2026-10-09).** Section 4 of the earlier version of this document was wrong in two
> ways. (1) The gate threshold τ was chosen by maximising accuracy on the **held-out questions' own
> labels**, and the text said it was "tuned on training questions", which was false. Every number
> in the old gating table, including "gating is worth about +4.6 panel points", was therefore
> optimistically biased; they are withdrawn. (2) The "8 independent splits" were random 70/30
> resamples of the same 350 questions, so their test sets overlapped: they are not independent
> replications, and the "±" figures are spreads across splits, not confidence intervals. Section 4
> below is replaced with a reanalysis in which every choice is made inside training questions, and
> the conclusion is reworded. Sections 1–3 used legitimate within-training selection for layer and
> regularisation; their numbers stand, with the "±" caveat.
>
> **Second correction (2026-10-09, review of b56c836).** Section 4's procedure still chose τ on
> raw out-of-fold probabilities and then applied it to calibrated held-out probabilities, and it
> maximised agent accuracy although the endpoint is panel accuracy. Section 5 reruns the analysis
> with both fixed; section 4 is kept unchanged as the superseded version (export:
> `artifacts/review/reanalysis_v1_b56c836/`). The corrections were specified by the review before
> this rerun; none was chosen by looking at its results.

All 1,050 events (350 questions x 3 agents) from `runs/night2-debate`, 401 of them revisions.
8 repeated question-grouped outer splits (70/30). Layer (from 6 depths) and C (from 1e-4..1e-1)
chosen by 3-fold grouped CV **inside training rows on every repetition**. Base rate for
"proposed answer correct" = 0.361.

## 1. Activations are not shown to beat output-only evidence

Target: is the proposed answer correct. Mean ± sd over 8 overlapping splits (spread, not an
interval).

| family | representation | AUROC | Brier | ECE | sd of predictions |
|---|---|---|---|---|---|
| constant | base rate | 0.5000 ± 0.000 | 0.2308 | **0.0406** | **0.0000** |
| activation | post | 0.6067 ± 0.021 | 0.2264 | 0.0814 | 0.132 |
| activation | pre | 0.6329 ± 0.042 | 0.2271 | 0.1200 | 0.207 |
| activation | pre_post | 0.6527 ± 0.034 | 0.2215 | 0.1154 | 0.213 |
| activation | pre_diff | 0.6552 ± 0.041 | 0.2205 | 0.1036 | 0.214 |
| output_only | output | 0.6415 ± 0.042 | 0.2184 | 0.0797 | 0.144 |
| combined | act + output | **0.6700 ± 0.040** | **0.2092** | 0.0825 | 0.178 |

Activations (0.653) against a fair output-only model (0.642) differ by 0.011, against a
between-split spread of ~0.04. That is no demonstrated difference. Combining gains 0.029 over output alone,
which is within one sd and should not be called a result without more data.

`pre_post` and `pre_diff` agree to 0.003, as expected: they are the same information
reparameterised.

## 2. The calibration claim is dead, exactly as the review predicted

**The constant base-rate predictor has the best ECE of anything tested** (0.0406), while ranking
at exactly chance and emitting zero variance. My earlier "probe ECE 0.022 vs raw confidence
0.757" comparison was worthless: it pitted a calibrated predictor against an uncalibrated one,
and low ECE by itself does not distinguish an informative predictor from one that repeats the
base rate. Retracted.

## 3. On the deployment population the probe is at chance

| subset | n per split (approx) | AUROC |
|---|---|---|
| all events | 315 | 0.6527 ± 0.034 |
| initially correct | 118 | 0.6837 ± 0.062 |
| initially incorrect | 197 | 0.5420 ± 0.064 |
| **actually revised** | **120** | **0.5323 ± 0.065** |

The all-event figure is carried by the easy cases. On revision events — the only population a
gate ever sees — eight overlapping splits put the probe at **0.532**, indistinguishable from
chance. This is consistent with the earlier 107-event null, now on all 401 revisions; the splits
reuse the same questions, so they are not independent confirmations.

The `initially correct` number (0.684) is inflated by an easier task: two thirds of initially
correct agents never revise, so predicting their final correctness largely means predicting
whether they revise at all.

## 4. Gating, redone with every choice made on training questions (superseded by §5)

`scripts/reanalysis.py` (corrected), outputs in `artifacts/review/reanalysis/`. Outer loop:
grouped 5-fold over questions, so every question is held out exactly once per seed. Inside each
training fold: layer and regularisation (activation models), regularisation or tree settings
(output models), Platt calibration, and τ are all chosen from out-of-fold predictions on training
questions only; the held-out questions are then scored once. A regression test flips every
held-out label and checks that no choice and no prediction changes. Three seeds = three fold
assignments of the **same** 350 questions: a stability check, not three experiments.

Gate rule for a revised answer: accept iff P(new correct) − P(old correct) > τ.

Output-only features (the baseline): full answer-choice distributions before and after, P(old)
and P(new) under each, margins, entropies, agreement of old and new answers with the other agents'
first answers, whether the answer changed, response lengths, categorical answer positions — fitted
as L2 logistic regression and as gradient-boosted trees. Activation: [pre, post] states at one
layer. Combined: both.

**Panel accuracy on held-out questions** (seed 0 / 1 / 2):

| gate | panel accuracy | agent accuracy |
|---|---|---|
| accept every revision | 0.377 | 0.361 |
| reject every revision | 0.371 | 0.381 |
| output-only, logistic | 0.383 / 0.391 / 0.406 | 0.375 / 0.383 / 0.381 |
| output-only, trees | 0.383 / 0.374 / 0.371 | 0.373 / 0.373 / 0.378 |
| activation | 0.406 / 0.391 / 0.406 | 0.388 / 0.384 / 0.388 |
| combined | 0.391 / 0.394 / 0.394 | 0.390 / 0.386 / 0.379 |
| agent-optimal (accept only wrong→right; not a panel bound) | 0.480 | 0.477 |
| panel-optimal (best of ≤8 accept patterns per question) | 0.480 | 0.431 |

**Paired differences in panel accuracy**, mean and 95% interval from resampling questions:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| activation − accept all | +0.029 [−0.006, +0.063] | +0.014 [−0.023, +0.051] | +0.029 [−0.003, +0.060] |
| combined − accept all | +0.014 [−0.023, +0.049] | +0.017 [−0.017, +0.051] | +0.017 [−0.011, +0.049] |
| output logistic − accept all | +0.006 [−0.034, +0.040] | +0.014 [−0.023, +0.049] | +0.029 [−0.006, +0.066] |
| activation − output logistic | +0.023 [−0.003, +0.049] | +0.000 [−0.029, +0.029] | +0.000 [−0.023, +0.020] |
| combined − output logistic | +0.009 [−0.017, +0.037] | +0.003 [−0.026, +0.031] | −0.011 [−0.034, +0.009] |
| activation − output trees | +0.023 [+0.000, +0.046] | +0.017 [−0.011, +0.046] | +0.034 [+0.009, +0.060] |

**Ranking (AUROC), seed 0 / 1 / 2:**

| model | P(new correct), all | P(new correct), revised only | P(new correct), within question | P(old correct), all | P(old correct), within question |
|---|---|---|---|---|---|
| output logistic | 0.644 / 0.643 / 0.634 | 0.526 / 0.500 / 0.440 | 0.633 / 0.566 / 0.592 | 0.668 / 0.663 / 0.655 | 0.628 / 0.622 / 0.595 |
| output trees | 0.642 / 0.642 / 0.612 | 0.523 / 0.546 / 0.482 | 0.653 / 0.602 / 0.628 | 0.642 / 0.619 / 0.628 | 0.595 / 0.662 / 0.601 |
| activation | 0.668 / 0.679 / 0.647 | 0.538 / 0.567 / 0.505 | 0.556 / 0.587 / 0.587 | 0.708 / 0.701 / 0.704 | 0.649 / 0.676 / 0.669 |
| combined | 0.678 / 0.665 / 0.662 | 0.541 / 0.534 / 0.519 | 0.607 / 0.577 / 0.612 | 0.709 / 0.700 / 0.705 | 0.622 / 0.649 / 0.669 |

Brier score for P(new correct): 0.208–0.217 for every model, against 0.232–0.235 for a constant
predictor that always outputs the training-fold base rate. That constant's calibration error is
about 0 — as good as any model's — which is again why calibration error alone is not evidence
of information.

What these measurements show, literally:

1. **No learned gate is shown to beat accepting every revision.** Every interval against
   accept-all includes zero, in every seed. The old "+4.6 points" came from choosing τ on the test
   labels. Point estimates are 0 to 3 points, and the intervals are about ±3 points wide at 350
   questions.
2. **Activations are not shown to add to the output-only logistic baseline**, alone or combined:
   every interval includes zero. The activation model's point estimate is +2.3 points in one seed
   and 0.0 in the other two.
3. Activation beats the tree baseline more often, with two of three intervals excluding or
   touching zero. The trees are the weaker baseline here (lower panel accuracy than logistic in two
   seeds), so beating them is not evidence of added value; the comparison that matters is against
   the stronger baseline.
4. **On revised answers, every model ranks P(new correct) near chance** (0.44–0.57).
5. The activation model's higher AUROC for P(new correct) over all events (0.65–0.68 vs
   0.63–0.64) does not hold within questions (0.56–0.59 vs 0.57–0.63). The cheaper explanation for
   that marginal edge is question-level information such as difficulty.
6. For P(old correct), activations rank higher than output features both overall (0.70 vs 0.66)
   and within questions (0.65–0.68 vs 0.60–0.63) in all three seeds. The three seeds share
   questions, so this consistency is stability, not replication, and no gate advantage followed
   from it here.

## 5. Gating with the threshold on the evaluated scale and the panel objective (current)

`scripts/reanalysis.py` at commit `6b6fa42`; outputs in `artifacts/review/reanalysis_v2/`
(per-answer raw and calibrated probabilities, per-question panel answers under every gate, every
chosen parameter and calibrator coefficient). Same data and outer folds as §4: grouped 5-fold
over the 350 questions, three fold assignments of the same questions.

What changed from §4:
1. τ is chosen on **calibrated** out-of-fold scores (each inner fold calibrated by a Platt fit on
   the others), the same construction used on held-out questions.
2. τ maximises **panel accuracy** under the tie-break actually applied (`round0_then_agent0`;
   the run's config named an unimplemented rule). The agent-accuracy τ is reported too.
3. Accept-all (τ = −∞) and reject-all (τ = +∞) are always candidates.
4. A fifth family: a cross-fitted logistic combination of the output-logistic and activation
   scores (`stacked`).
5. Output features come from the shared adapter. For this old run they are §4's plus verbalized
   confidence (with a missingness indicator) and stated-versus-readout disagreement indicators,
   before and after; columns for fields the run never recorded (fallback, cap hits, shown-peer
   answers) are constant zero. Feature names are in `results.json`.

**Panel accuracy on held-out questions, panel-objective τ** (seed 0 / 1 / 2):

| gate | panel accuracy | agent accuracy |
|---|---|---|
| accept every revision | 0.377 | 0.361 |
| reject every revision | 0.371 | 0.381 |
| output-only, logistic | 0.414 / 0.406 / 0.391 | 0.398 / 0.391 / 0.380 |
| output-only, trees | 0.394 / 0.389 / 0.386 | 0.386 / 0.382 / 0.383 |
| activation | 0.406 / 0.423 / 0.414 | 0.392 / 0.404 / 0.396 |
| combined | 0.400 / 0.391 / 0.414 | 0.391 / 0.390 / 0.395 |
| stacked | 0.403 / 0.420 / 0.411 | 0.391 / 0.403 / 0.387 |
| agent-optimal (not a panel bound) | 0.480 | 0.477 |
| panel-optimal (best of ≤8 patterns) | 0.480 | 0.431 |

**Paired differences in panel accuracy**, 95% intervals resampling questions:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| output logistic − accept all | +0.037 [+0.011, +0.066] | +0.029 [−0.006, +0.063] | +0.014 [−0.017, +0.046] |
| activation − accept all | +0.029 [−0.003, +0.060] | +0.046 [+0.011, +0.080] | +0.037 [+0.009, +0.066] |
| stacked − accept all | +0.026 [−0.009, +0.060] | +0.043 [+0.014, +0.071] | +0.034 [+0.003, +0.066] |
| combined − accept all | +0.023 [−0.006, +0.051] | +0.014 [−0.014, +0.043] | +0.037 [+0.009, +0.066] |
| activation − output logistic | −0.009 [−0.031, +0.014] | +0.017 [−0.011, +0.046] | +0.023 [+0.000, +0.046] |
| combined − output logistic | −0.014 [−0.037, +0.009] | −0.014 [−0.040, +0.011] | +0.023 [+0.003, +0.046] |
| stacked − output logistic | −0.011 [−0.037, +0.011] | +0.014 [−0.009, +0.037] | +0.020 [+0.000, +0.040] |

**Ranking differences (AUROC), minus output logistic**, 95% intervals resampling questions:

| comparison | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| activation, P(old correct), all answers | +0.043 [+0.004, +0.083] | +0.037 [−0.006, +0.078] | +0.041 [+0.002, +0.079] |
| stacked, P(old correct), all answers | +0.044 [+0.013, +0.077] | +0.045 [+0.012, +0.077] | +0.046 [+0.014, +0.077] |
| stacked, P(new correct), all answers | +0.042 [+0.006, +0.079] | +0.043 [+0.012, +0.076] | +0.017 [−0.010, +0.044] |
| activation, P(new correct), revisions only | +0.008 [−0.082, +0.098] | +0.069 [−0.023, +0.159] | +0.068 [−0.004, +0.141] |
| stacked, P(new correct), revisions only | +0.011 [−0.061, +0.082] | +0.048 [−0.008, +0.106] | +0.046 [−0.005, +0.098] |

On revisions alone, every family's AUROC for P(new correct) is 0.44–0.57 in every seed.
Within-question AUROC for P(old correct): activation 0.649 / 0.676 / 0.669, stacked
0.635 / 0.669 / 0.689, output logistic 0.622 / 0.635 / 0.615.

What these measurements show, literally:

1. **Gates now show gains over accepting every revision**: point estimates of +1.4 to +4.6 panel
   points for the logistic, activation, combined and stacked gates, with intervals excluding zero
   in one to two of the three fold assignments for each. The trees gate does not. Compared with
   §4, the change comes from the corrected procedure (calibrated-scale τ, panel objective). The
   three fold assignments reuse the same 350 questions, so their agreement in direction is
   stability, not three confirmations.
2. **Activations are not shown to raise panel accuracy over the output-logistic gate.** The
   activation, combined and stacked gates are within about ±2 points of it, and each interval
   includes or touches zero.
3. **Activations rank whether the agent's FIRST answer was correct better than output features
   do**, by about 0.04 AUROC: for the stacked combination the interval excludes zero in all three
   fold assignments; for activations alone, in two of three. The within-question AUROCs show the
   same ordering (no interval was computed for that difference), so question difficulty alone
   does not appear to explain it.
4. **Whether a revision is correct remains near chance for every predictor** on revisions alone;
   activation-minus-output differences there are positive in all seeds but every interval
   includes zero.

## Answer to the core research question, at this scale

> After an agent sees peer responses and proposes a new answer, do its internal activations
> provide information beyond its answer probabilities and observable debate features about
> whether to accept the new answer?

**No demonstrated incremental activation benefit to panel accuracy under the tested 1B
protocol.** With the corrected procedure (§5), gates built from output features alone already
improve on accepting every revision by a few panel points in some fold assignments, and adding
activations does not measurably improve on that gate. Activations do add a modest, consistent
improvement in ranking whether the agent's first answer was correct (~0.04 AUROC), which did not
translate into a measurable panel gain at this sample size. Whether a proposed revision is
correct is not predicted above chance by anything tested. All of this is at 350 questions with
intervals of about ±3 panel points: effects of a few points are not ruled out in either direction.

Scope of that claim: gemma-3-1b-it, ARC-Challenge, 350 questions, generations truncated at 64
tokens so agents exchanged incomplete reasoning. Untested: a competent model producing complete
responses. The pre-revision `peer_exposure` state was tested afterwards on a separate 1B run
(`docs/PEER_EXPOSURE.md`) and did not change the picture.
