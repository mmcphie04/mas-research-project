# Preregistration — DRAFT, not yet committed

**Status:** draft, revised 2026-10-09 against the code at the reviewed commit. Not in force until
the author and advisor edit it, date it, and commit it **after the pilot and before any
main-pool question is generated.** Bracketed items `[...]` are open decisions; each bracket gives
the current code default. Once dated, changes go in the deviation log, not the text.

Every rule here names the code that implements it. If they disagree, the code is wrong.

## 1. Question

When an agent in a one-round, three-agent debate proposes a revised answer, can we tell whether
accepting the revision helps — and do the model's internal activations add to what its output
probabilities and the observable debate already tell us, well enough to change panel accuracy?

## 2. Materials

- **Model:** `google/gemma-3-12b-it`, bf16, Hugging Face backend; model commit, tokenizer, chat
  template hash, special tokens and option-letter ids are fixed in the job file at the first
  session and checked on every resume (`mad/job.py`).
- **Data and estimand:** MMLU-Pro test split in one fixed master order — subjects interleaved
  round-robin; within a subject, by a seeded hash of the id (seed 7) (`mad.job.master_order`).
  The first 120 are the **pilot pool**; the **main pool** is the rest. A main run of N questions
  is the first N of the main pool, so it holds roughly equal numbers per subject. **Every
  accuracy is therefore a subject-balanced estimate, not an estimate for MMLU-Pro's natural
  subject mix.**
- **Main sample size:** `N = [900]`, set from the pilot's measured cost (docs/BUDGET.md).

## 3. Protocol (fixed by the job file and prompt hashes)

- Three agents, one persona, sampled independently (temperature 0.7, top-p 0.95), token cap
  `[768]`. Prompts ask for about 150 words of reasoning ending in `Answer: (X)` and
  `Confidence: NN`; round 1 asks for the same lines (`mad/prompts.py`).
- Round 0 once per question, saved, and shared by every condition and every child job.
- **Conditions (main):** `debate` (both real peers' responses, confidence lines removed, random
  recorded order), `self_reconsider` (no peer content), `[debate_answer_only]` (each peer's stated
  letter **and option text**, no reasoning; a peer with no valid stated answer shows "No final
  answer was given."). `misleading_one` (one fabricated mildly worded peer for a uniformly random
  wrong option + one random real peer) is run in the pilot and is secondary.
  `[If throughput requires: debate_answer_only on a matched subset — the first [450] main
  questions — as a child job reusing the parent's saved round 0.]`
- **Stated answer** (`mad/parsing.py`): the last answer field of the agent's own response under a
  line-anchored grammar (`Answer: (X)` strict; `Answer: X` alone on the line, or `Answer: (X)`
  / `X)` followed by text, permissive). A letter outside the options makes the response invalid;
  earlier fields are not used.
- **Canonical answer:** the stated answer; if invalid, the forced-readout answer, flagged
  `readout_fallback`. **Peers are never shown a fallback**: what a peer shows is what it stated.
- **Panel answer:** plurality of three canonical answers; ties by round-0 plurality, then the
  lowest agent id (`round0_then_agent0`, validated at config load).

## 4. Measurements and predictors

- **Saved:** see docs/GPU_MEASUREMENTS.md.
- **Targets:** for each agent after round 1, `y_old` = its first answer is correct, `y_new` = its
  revised answer is correct. Beneficial / harmful / unhelpful revisions are descriptive only.
- **Information times** (`mad/analysis/features.py`): `post` (primary): everything up to and
  including the revised response. `exposure` and `pre` (secondary): only what exists before the
  revision / before round 1. A predictor never uses information from a later time.
- **Output-only features:** full letter distributions at every available phase; probability of
  the old and new answers under each; margins; entropies; verbalized confidence with a
  missingness indicator; response lengths; response log-likelihood (mean, min); fallback,
  token-cap and stated-vs-readout-disagreement indicators; agreement of the old/new answer with
  the other agents' first answers and with what the agent was shown; whether it revised;
  categorical answer positions. Never: gold, correctness, which peer was fabricated, the
  misleading target. Feature names are written into every results file.
- **Families:** output-only L2 logistic (`output_linear`); output-only gradient-boosted trees,
  depth 3 (`output_hgb`); activations at one layer, L2 logistic (`activation`); activations +
  output features (`combined`); a cross-fitted logistic combination of the `output_linear` and
  `activation` scores (`stacked`).
- **Activation position:** `[readout_last]` primary. `generation_last`, `stated_answer`,
  `before_stated_answer` are secondary. Activation phases follow the information time (post:
  pre + post states). Candidate layers: 25/50/65/80/92/100% depth.

## 5. Analysis (`scripts/reanalysis.py --mode split`)

- **Split** (`scripts/make_split.py`): main-pool questions into train / validation / test
  `[37.5 / 12.5 / 50%]`, stratified by subject, seed `[...]`, by question (all agents,
  conditions and child jobs of a question on the same side). The manifest records the job
  identity hash. Frozen before any main-pool outcome is examined.
- **Fitting:** layer, regularisation and tree settings by grouped cross-validated AUROC within
  train. Models refit on all of train.
- **Calibration and threshold:** Platt calibrators fitted on validation; the gate accepts a
  revision iff calibrated P(new) − calibrated P(old) > τ; τ chosen on validation over a grid that
  includes −∞ (accept every revision) and +∞ (reject every revision), maximising **panel
  accuracy** under the declared tie-break (agent-accuracy τ reported as a secondary rule). The
  test set is scored once, with everything frozen.
- **Primary outcome — `[choose before the main run]`:**
  - *Option 1 (current draft):* paired difference in test panel accuracy, `combined` gate minus
    `output_linear` gate, in `debate`; 95% interval from resampling test questions. At N = 900
    (450 test) only a ~4–5 point difference is detectable at 80% power unless the gates rarely
    disagree (docs/BUDGET.md).
  - *Option 2 (better powered):* paired difference in held-out AUROC for P(new correct),
    `combined` minus `output_linear`, on all agent answers and on revisions, in `debate`;
    intervals resample questions. Panel accuracy co-primary, reported with its interval.
- **Secondary:** every family against `output_linear` and `output_hgb`; every gate against
  accept-all and reject-all; beneficial revisions lost vs harmful revisions blocked; AUROC
  overall, on revisions and within question (difficulty guard); calibration (Brier, ECE, against a
  constant base-rate predictor); the `exposure` and `pre` information times; the other
  positions; the panel-optimal ceiling (best of ≤ 8 accept patterns per question) and the
  agent-optimal policy (accept only wrong→right), labelled as such; transfer of gates fixed on
  `debate` to `self_reconsider` and `debate_answer_only` on the same held-out questions
  (observational, not causal).
- **Decision rule for the primary outcome:** lower bound of the 95% interval > 0 → activations add
  measurable value here; interval within ±`[2 panel points | 0.02 AUROC]` → no practically
  meaningful added value; otherwise inconclusive at this sample size.
- **Intervals:** every agent-level interval resamples questions (agents of one question are
  correlated); no row-wise DeLong. Repeated CV seeds (exploratory route,
  `scripts/reanalysis.py --mode cv`) are sensitivity checks, not replications.

## 6. Data handling

- **Primary population:** every main-pool question with complete states at the primary position
  (`readout_last` exists for every state). `readout_fallback` rows stay in it.
- **Missingness:** analyses at `stated_answer` / `before_stated_answer` drop every question in which
  any agent lacks that state, and report how many (`questions_dropped_missing_states`). No
  position is ever substituted for another (`mad/analysis/states.py`).
- **Sensitivity:** primary analysis repeated without questions containing a `readout_fallback` or
  a capped response. These are sensitivity analyses; the primary population is not changed post
  hoc.
- Incomplete chunks: questions in a chunk that did not complete are excluded and listed.

## 7. Pilot (pilot pool only) and what it may change

Four conditions on the 120 pilot questions (`scripts/pilot_report.py`): round-0 accuracy;
first-answer disagreement; revision rates; beneficial / harmful / unhelpful counts with distinct
questions; vote, accept-all and panel-optimal accuracy and headroom above the better baseline;
stated-answer parse status; cap-hit rate; stated-vs-readout disagreement; measured seconds per
question by stage; disk per question; worst-case memory; power sensitivity; a fixed random
transcript sample and every malformed answer, read by hand.

Aims, investigated rather than applied mechanically: parsed answers ≥ `[98%]`; cap hits ≤
`[2–5%]`; first answers not unanimous in a useful share of questions; a non-trivial number of
both beneficial and harmful revisions. High initial accuracy alone is not disqualifying.

**The pilot may change:** token cap, brevity wording (a prompt change means a new job and a new
pilot), batch sizes, N, which optional conditions run in full or on a subset. **It may not
change:** the primary outcome, decision rule, predictor families, threshold objective, or the
retained activation positions.

## 8. Sample size

`N` and the split are set from the pilot's measured seconds per question and the power table
(docs/BUDGET.md), within a total of `[10–20]` A100-hours, before any main-pool outcome is seen.

## Deviation log

| date | change | reason | made before or after seeing main-pool results |
|---|---|---|---|
