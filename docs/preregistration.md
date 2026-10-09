# Preregistration — DRAFT, not yet committed

**Status:** draft written 2026-10-09 from the review of commit 9e45160. Not in force until the
author and advisor edit it, date it, and commit it **after the pilot and before any main-pool
question is generated.** Every bracketed item `[...]` is a decision still open. Once dated,
changes go in the deviation log at the end, not into the text.

## 1. Question

After an agent in a one-round, three-agent debate proposes a revised answer, do its internal
activations tell us whether to accept the revision **beyond what its output probabilities and
the observable debate already tell us**, well enough to raise the panel's final accuracy?

## 2. Materials

- **Model:** `google/gemma-3-12b-it`, bf16, Hugging Face backend. The model commit, tokenizer,
  chat template hash, special tokens and option-letter token ids are recorded in the job file at
  the first session and checked on every resume.
- **Data:** MMLU-Pro test split. Every question is placed once in a fixed master order (subjects
  interleaved; within a subject, order by a seeded hash of the question id; seed 7). The first
  120 questions of that order are the **pilot pool**; the **main pool** is the rest. Pilot
  questions never enter the main analysis.
- **Main sample size:** the first `[N]` questions of the main pool, `N` set from pilot estimates
  by the rule in section 8.

## 3. Protocol (frozen by the job file and prompt hashes)

- Three agents, same model and persona, sampled independently (temperature 0.7, top-p 0.95).
- Round 0: each agent answers alone. Round 1: each agent sees a condition-specific message and
  may revise. One round only.
- Prompts ask for concise reasoning (about 150 words) ending in `Answer: (X)` and
  `Confidence: NN`; round 1 asks for exactly the same lines. Token cap `[768]`.
- **Conditions:**
  - `debate` (primary): both real peers' round-0 responses, confidence line removed, in a random
    recorded order.
  - `self_reconsider` (control): no peer content.
  - `misleading_one` (secondary): one fabricated, mildly worded peer for a wrong option drawn
    uniformly at random (never the correct answer or the agent's own), plus one real peer chosen
    at random, in a random order. Target, slot and source are recorded.
  - `[debate_answer_only]` (optional): peers' stated letters only, no reasoning.
- **Canonical answer:** the letter in the agent's last `Answer: (X)` line. If none parses, the
  forced-readout answer is used and the row is flagged `readout_fallback`. The forced readout
  (`... Final answer: (` → distribution over letters) is recorded for every state regardless.
- **Panel answer:** plurality of the three final answers; ties broken by the round-0 plurality,
  then by the lowest agent id (`round0_then_agent0`).
- **Revision labels** (description only): beneficial = wrong→right, harmful = right→wrong,
  unhelpful = wrong→different wrong. Among revisions these reduce to correctness of the old or new
  answer, so every model predicts correctness.

## 4. Predictors

All predictors estimate, for each agent after round 1, P(old answer correct) and P(new answer
correct). The gate accepts a revision iff P(new) − P(old) > τ.

- **Output-only (the baseline to beat):** full letter distributions from the forced readout
  before and after revision; probability of the old and of the new answer under each; margins;
  entropies; agreement of the old and new answers with the other agents' round-0 answers;
  whether the answer changed; response lengths; categorical answer positions. Fitted twice: L2
  logistic regression, and gradient-boosted trees (depth 3).
- **Activation:** residual-stream states at `[position: readout_last | generation_last |
  stated_answer]` before and after revision, L2 logistic regression.
- **Combined (primary):** activation features plus all output-only features, one L2 logistic
  regression.
- `[Position and candidate layers chosen on the pilot pool and fixed here.]`

## 5. Analysis

- **Split:** main-pool questions into training `[70%]` and test `[30%]`, stratified by subject,
  seed `[...]`, grouped by question (all agents and conditions of a question on the same side).
- **Every choice** — layer, regularisation, tree settings, τ, calibration — is made inside the
  training questions by grouped 5-fold cross-validation, using out-of-fold predictions for τ and
  Platt calibration. The test questions are scored once. Code: `scripts/reanalysis.py`, whose
  test suite checks that flipping every held-out label changes no choice and no prediction.
- **Primary outcome:** paired difference in test-set panel accuracy, combined gate minus the
  output-only logistic gate, in the `debate` condition. 95% interval from `[10,000]` bootstrap
  resamples of test questions.
- **Secondary outcomes:** combined minus the output-only boosted-tree gate; the same differences
  for the activation-only gate; agent-level accuracy
  differences; AUROC on revised events for P(new correct), activation vs output-only, paired
  DeLong; within-question AUROC (guard against a difficulty detector); every gate against
  accept-all and reject-all; the panel-optimal ceiling (best of at most eight accept patterns per
  question) and the agent-optimal policy, labelled as such; all of the above in `misleading_one`.
- **Decision rule for the primary outcome:**
  - lower bound of the 95% interval > 0 → activations add measurable value in this setting;
  - interval inside ±`[2]` panel points → no practically meaningful added value;
  - otherwise → inconclusive at this sample size.
- Repeated splits, if any, are reported as stability checks, never as independent replications;
  a standard deviation across overlapping splits is not a confidence interval.

## 6. Data handling

- A question enters the analysis only if every state for it exists in the condition analysed.
  Missing questions are listed with the reason (failed chunk, truncation) and counted.
- `readout_fallback` rows stay in the primary analysis; a sensitivity analysis drops every
  question containing one.

## 7. Pilot (pilot pool only) and what it may change

Run all three conditions on the 120 pilot questions. Report: round-0 accuracy; share of
questions whose first answers are not unanimous; revision rate; beneficial/harmful/unhelpful
counts with distinct questions; panel-optimal minus accept-all; stated-answer parse rate;
cap-hit rate; stated-vs-readout disagreement; seconds per question; allocated and reserved GPU
memory peaks under the stress pass.

**Go conditions:** stated answer parsed ≥ `[95%]`; cap hit ≤ `[5%]`; round-0 accuracy between
0.25 and 0.70; first answers not unanimous ≥ `[30%]`; revision rate ≥ `[10%]`; panel-optimal
minus accept-all ≥ `[4]` points.

**The pilot may change:** token cap, brevity wording, batch sizes, which activation positions are
kept, candidate layers, `N`. **It may not change** the primary outcome, the decision rule, or the
predictor families. Any prompt change starts a new job and a new pilot.

## 8. Sample size

The primary outcome is a paired difference in panel accuracy, so power depends on how many test
questions the two gates decide differently. From the pilot, estimate the share of questions with
discordant panel outcomes between gates `[d]`. With `n_test` test questions there are about
`d × n_test` discordant ones; detecting a `[3]`-point difference at 80% power needs about
`[...]` of them. Set `N = n_test / 0.3`, capped by the compute budget (about `[...]` GPU-hours
at the pilot's measured seconds per question).

## Deviation log

| date | change | reason | made before or after seeing main-pool results |
|---|---|---|---|
