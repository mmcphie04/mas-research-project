# Preregistration — DRAFT, not yet committed

**Status:** draft, revised 2026-10-10 against the code at the reviewed commit. Not in force until
the author and advisor confirm the checklist (§8), date it, and commit it **after the pilot and
before any main-pool question is generated.** Values in this draft are proposals to confirm;
the only items left open are N (from the pilot, by the §9 rule) and the remaining budget.
Once dated, changes go in the deviation log, not the text.

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
- **Main sample size:** N set from the pilot's measured cost by the rule in §9 (planning
  figure: 900, docs/BUDGET.md).

## 3. Protocol (fixed by the job file and prompt hashes)

- Three agents, one persona, sampled independently (temperature 0.7, top-p 0.95), token cap
  768 (the pilot may lower it; §7). Prompts ask for about 150 words of reasoning ending in `Answer: (X)` and
  `Confidence: NN`; round 1 asks for the same lines (`mad/prompts.py`).
- Round 0 once per question, saved, and shared by every condition and every child job.
- **Conditions (main):** `debate` (both real peers' responses, confidence lines removed, random
  recorded order), `self_reconsider` (no peer content); and, as a child job on the first 450 main questions
  (or in full if the pilot shows it affordable), `debate_answer_only` (each peer's stated
  letter **and option text**, no reasoning; a peer with no valid stated answer shows "No final
  answer was given."). `misleading_one` (one fabricated mildly worded peer for a uniformly random
  wrong option + one random real peer) is run in the pilot and is secondary.
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
- **Activation position:** `readout_last` primary. `generation_last`, `stated_answer`,
  `before_stated_answer` are secondary. Activation phases follow the information time (post:
  pre + post states). Candidate layers: 25/50/65/80/92/100% depth.

## 5. Analysis (`scripts/reanalysis.py --mode split`)

- **Split** (`scripts/make_split.py`, selection mode): made **before any main-pool question is
  generated**, from the same deterministic selection the driver makes; train / validation / test
  **37.5 / 12.5 / 50%**, stratified by subject, **seed 20261010**, by question (all agents,
  conditions and child jobs of a question on the same side). The main job is started with
  `--split-manifest ... --require-split`: the driver refuses, before loading the model, a split
  that does not cover exactly its questions; records the assignment hash in the job identity and
  keeps the assignment itself as `<job>-split.json`; children inherit it. The analysis refuses a
  manifest whose hash, split names, partitions or membership are wrong, or which the run did not
  record (`load_split`).
- **Fitting:** probe layer, regularisation and tree settings by grouped cross-validated
  **log-loss** within train (a proper scoring rule; mean per-fold AUROC is blind to scale and,
  on the pilot, chose near-constant models whose pooled predictions ranked below chance);
  models refit on all of train. No choice uses test questions.
- **Calibration and threshold:** monotone Platt calibrators fitted on validation (a calibrator
  whose fitted slope is not positive outputs the base rate instead of reversing the ranking); the gate accepts a
  revision iff calibrated P(new) − calibrated P(old) > τ ("difference" rule); τ chosen on
  validation over a grid that includes −∞ (accept every revision) and +∞ (reject every
  revision), maximising **panel accuracy** under the **`round0_then_agent0`** tie-break. The test
  set is scored once, with everything frozen.
- **Output controls, identical for every activation comparison:** the output-only features
  (§4) plus the option distributions saved just **before the agent's own answer letter** and at
  its **last generated token** (`--output-positions before_stated_answer generation_last`), with
  explicit missingness and information-time restrictions (at `pre`, only round-0 values; at
  `exposure`, nothing from the revised response). The confidence read at `Final answer: (` partly
  reflects repeating an answer already written; the before-letter distribution is the model's
  confidence before committing. The readout-only output baseline is kept as an ablation
  (`output_readout_only`).
- **Primary outcome (one):** in `debate`, the paired difference in **held-out panel accuracy**
  between the `combined` gate (activations + the output controls above) and the `output_linear`
  gate (the output controls alone, L2 logistic), at activation position `readout_last`,
  information time `post`. 95% interval from 10,000 bootstrap resamples of test questions.
  **Power, stated in advance:** with ~450 test questions only a ~4–5 point difference is
  detectable at 80% power unless the two gates rarely disagree (docs/BUDGET.md); the interval is
  reported whatever its width.
- **Decision rule for the primary outcome** (δ = **2 panel points**):
  lower bound > 0 → activations add measurable panel accuracy here;
  upper bound < +δ → no practically meaningful advantage (the interval excludes one);
  otherwise → inconclusive at this sample size. A non-significant difference is never reported
  as equivalence unless the upper bound is below +δ.
- **Predictive gains are not panel gains.** Better ranking or calibration of correctness (AUROC,
  Brier) and better panel accuracy under a selected threshold are reported separately and can
  disagree (they did on the 1B data: docs/REANALYSIS.md §6).
- **Secondary outcomes (all in `debate` unless stated; reported, never substituted for the
  primary):**
  1. predictive: AUROC of `combined`, `activation` and `stacked` minus `output_linear`, for P(old
     correct) and P(new correct) separately, on all answers, on revisions only, and within
     question (difficulty guard); calibration (Brier, log-loss, ECE) against a constant predictor
     that always outputs the training base rate;
  2. the **pre-only protection gate** (`--time pre`, rule `protect`): keep the first answer when its
     predicted correctness from round-0 information exceeds a validation-chosen threshold;
     compared with accept-all, reject-all and the primary gates -- distinct from predicting that
     a replacement is right;
  3. information stages: `pre`, `exposure`, `post`, and `post_all` (all three stages' states)
     against `post` (pre + post);
  4. the readout-only output ablation; the trees output family; other activation positions;
  5. beneficial revisions lost vs harmful revisions blocked, per gate; the panel-optimal ceiling
     (best of ≤ 8 accept patterns per question) and the agent-optimal policy, labelled as such;
  6. transfer of gates fixed on `debate` to `self_reconsider` and `debate_answer_only` on the same
     held-out questions (observational, not causal); `misleading_one` (pilot only);
  7. optional candidate-evaluation diagnostic (`scripts/run_candidates.py`): any prompt check uses
     pilot or training questions only (`--check`); the diagnostic sample is drawn explicitly per
     split from the registered split (`--n-per-split`); probes and the fitted output comparator
     train on train + validation and are scored once on test.
- **Intervals:** every agent-level interval resamples questions (agents of one question are
  correlated); no row-wise DeLong. Repeated CV seeds (exploratory route,
  `scripts/reanalysis.py --mode cv`) are sensitivity checks, not replications.
- **Frozen analysis commands** (`$R` the archive root):

```bash
A="--mode split --split-manifest $R/main12b-split.json --output-positions before_stated_answer generation_last"
F="--families output_linear output_hgb output_readout_only activation combined stacked"
# primary (and secondary 1, 4, 5, 6)
python scripts/reanalysis.py --run $R/main12b-debate $A $F --time post --position readout_last \
    --transfer-runs $R/main12b-self_reconsider --out $R/analysis/primary
# secondary 2 and 3
python scripts/reanalysis.py --run $R/main12b-debate $A $F --time pre      --out $R/analysis/pre
python scripts/reanalysis.py --run $R/main12b-debate $A $F --time exposure --out $R/analysis/exposure
python scripts/reanalysis.py --run $R/main12b-debate $A $F --time post_all --out $R/analysis/post_all
# secondary 4: other activation positions, same output controls
python scripts/reanalysis.py --run $R/main12b-debate $A $F --time post --position before_stated_answer \
    --out $R/analysis/pos_before_letter
```

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

Aims, investigated rather than applied mechanically: parsed answers ≥ 98%; cap hits ≤ 2–5%; first answers not unanimous in a useful share of questions; a non-trivial number of
both beneficial and harmful revisions. High initial accuracy alone is not disqualifying.

**The pilot may change:** token cap, brevity wording (a prompt change means a new job and a new
pilot), batch sizes, N, which optional conditions run in full or on a subset. **It may not
change:** the primary outcome, decision rule, predictor families, threshold objective, or the
retained activation positions.

## 8. Freeze checklist (confirmed and dated before the main run)

| item | value in this draft | confirmed |
|---|---|---|
| primary outcome and comparator | `combined` vs `output_linear` (strong output controls), held-out panel accuracy, `debate`, `readout_last`, `post` | `[ ]` |
| threshold rule, objective, tie-break | difference rule; panel accuracy; `round0_then_agent0` | `[ ]` |
| output controls for every comparison | readout features + `before_stated_answer` + `generation_last` distributions | `[ ]` |
| split fractions and seed | 37.5 / 12.5 / 50; seed 20261010 | `[ ]` |
| decision rule tolerance δ | 2 panel points | `[ ]` |
| N and conditions | set by the rule in §9 from the pilot report; conditions `debate`, `self_reconsider` | `[ ]` |
| cumulative main time cap (`MAIN_MAX_HOURS`) | §9 projection × 1.15 | `[ ]` |
| bootstrap resamples | 10,000 (`scripts/reanalysis.py --n-boot`, default) | `[ ]` |
| stopping rule | below | `[ ]` |

**Stopping rule:** collect exactly the N questions in the split manifest; no outcome on any
main-pool question is examined until collection ends (the notebook's main stage shows only the
archive verification; status files show progress, parse and cap rates, never accuracy). The
cumulative compute cap for the main job is `MAIN_MAX_HOURS` = the §9 projection × 1.15, fixed in
§8 before generation. Collection stops early only for a
technical failure that cannot be fixed without changing the protocol, or when the compute budget
is exhausted; in either case the completed chunks are analysed as they stand and the shortfall
is reported. Collection is never stopped or extended because of results.

## 9. Sample size

N is the largest multiple of the chunk size such that the pilot report's projected A100-hours for
the main conditions (measured seconds per question, per condition), plus the planned child jobs,
plus 15% contingency, fit within the remaining budget of `[... A100-hours]`, capped at 1,200 (a
larger N requires amending this cap here before any main data is generated). It
is computed from the pilot report, written into §8, and frozen -- together with the split made
for exactly those N questions -- before any main-pool question is generated.

## Deviation log

| date | change | reason | made before or after seeing main-pool results |
|---|---|---|---|
| 2026-10-12 (draft, pre-dating) | model selection by grouped CV log-loss instead of AUROC; monotone calibration | found on the pilot's exploratory analysis: AUROC selection chose near-constant models and Platt slopes reversed rankings (docs/PILOT_ANALYSIS.md) | before any main-pool data exists |
