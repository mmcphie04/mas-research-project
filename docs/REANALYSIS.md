# Reanalysis under the second review's methodology

All 1,050 events (350 questions x 3 agents) from `runs/night2-debate`, 401 of them revisions.
8 repeated question-grouped outer splits (70/30). Layer (from 6 depths) and C (from 1e-4..1e-1)
chosen by 3-fold grouped CV **inside training rows on every repetition**. Base rate for
"proposed answer correct" = 0.361.

## 1. Activations do not beat output-only evidence

Target: is the proposed answer correct. Mean ± sd over 8 splits.

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
between-split sd of ~0.04. That is not a difference. Combining gains 0.029 over output alone,
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
gate ever sees — eight independent splits put the probe at **0.532**, indistinguishable from
chance. This is no longer a small-sample story: the earlier 107-event null replicates across
repeated splits with 2.6x the data.

The `initially correct` number (0.684) is inflated by an easier task: two thirds of initially
correct agents never revise, so predicting their final correctness largely means predicting
whether they revise at all.

## 4. Gating works — but output features do it just as well

τ tuned on training questions, evaluated on held-out. Mean ± sd over 8 splits.

| policy | agent accuracy | panel accuracy |
|---|---|---|
| accept every revision | 0.3575 ± 0.035 | 0.3738 ± 0.038 |
| reject every revision | 0.3738 ± 0.047 | 0.3619 ± 0.049 |
| **output-only old-vs-new** | 0.4071 ± 0.039 | **0.4202 ± 0.041** |
| activation old-vs-new | 0.4036 ± 0.043 | 0.4155 ± 0.042 |
| combined | 0.4024 ± 0.045 | 0.4190 ± 0.044 |
| oracle (upper bound) | 0.4702 ± 0.037 | 0.4798 ± 0.044 |

Two things here, and they point in opposite directions.

**Gating is worth about +4.6 panel points** over accepting every revision, and captures roughly
44% of the oracle's headroom. My earlier report that "gating changes nothing" was wrong — it
used a single-probe threshold rather than the two-predictor old-versus-new margin, which is the
correct decision rule.

**The activations contribute nothing to it.** Output-only gating (0.4202) matches activation
gating (0.4155) and the combined model (0.4190), all inside one standard deviation.

## Answer to the core research question, at this scale

> After an agent sees peer responses and proposes a new answer, do its internal activations
> provide information beyond its answer probabilities and observable debate features about
> whether to accept the new answer?

**On this evidence, no.** Eight repeated splits, a fairly-fitted output baseline including the
recovered full answer-choice distribution, and the actual decision metric all agree.

Scope of that claim: gemma-3-1b-it, ARC-Challenge, 350 questions, generations truncated at 64
tokens so agents exchanged incomplete reasoning. Two things remain genuinely untested — a
competent model producing complete responses, and the pre-revision `peer_exposure` state, which
is measured before the answer is committed rather than after.
