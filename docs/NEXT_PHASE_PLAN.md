# Next-phase plan (2026-10-29): commitment curves, structured debate, next-action prediction

**Status (amended 2026-10-29 after review).** The decisions in the next section are binding
and supersede any conflicting text further down. **Stage 0 (implementation and CPU work) is
approved.** No GPU session may start until the implementation and CPU findings have been reviewed.
CPU results are reported in docs/NEXT_PHASE_CPU_RESULTS.md, not here.

**Exploratory status.** Any analysis that reuses the pilot (master positions 1–120) or the
validation tranche (121–320) is exploratory: those questions have already been examined.
Positions 321 and later stay reserved for fresh final evaluation (docs/preregistration.md §10).

**Labels used throughout.** "Measured" marks numbers taken from logs or archives. "Estimate" marks
projections, given with the assumptions behind them.

## Review decisions (2026-10-29)

1. **Prediction-curve prefix.**
   - Primary standardized prefix: `Answer: (`. At f > 0 it follows a blank-line separator
     (`reasoning + "\n\n" + "Answer: ("`); at f = 0 it comes directly after the generation
     prompt.
   - Two bridge readouts keep our existing suffix `\nFinal answer: (` at f = 0 and f = 1.
   - The CPU preview uses the saved exposure probabilities.
2. **Exact replay and the standardized measurement are kept separate.**
   - **Exact replay** reproduces the saved `before_stated_answer` letter probabilities from the
     original token sequence (stored prompt ids + generated ids), ending immediately before the
     answer letter.
   - The **standardized 100% readout** (reasoning + `\n\nAnswer: (`) is computed separately.
     The two need not be identical, because formatting and tokenization can differ.
   - A readout that has already seen the answer is never the 100% point.
   - Sampled answers need not equal the most probable letter, so "argmax = stated answer" is a
     measured rate, not an assumption.
3. **Token construction.**
   - Saved generated token ids are preserved, and every truncation boundary is validated (logged
     when decode(ids[:k]) differs from the character cut).
   - The paper's re-tokenization variant is implemented for comparison. Running it on the GPU
     awaits review.
4. **Coverage.**
   - Eligibility stays as proposed: the revision's own and previous answers must be *stated*.
   - All eligible unchanged responses are read.
   - Exclusions are reported by condition and reason.
   - A capped response is **not** excluded merely for hitting the cap if it still contains a
     valid stated answer.
5. **Action selection: final-panel voting.**
   - stop = the 3 initial answers;
   - debate = the 3 revised answers;
   - additional sampling = the 6 independent answers.

   Exact expected accuracy is computed under uniformly random tie-breaking. Fractional outcomes
   are supported throughout training and reporting: classifiers accept fractional labels, nothing
   is cast to integers, and fix/harm is reported as expected gains and losses. Focused tests are
   added. The old pooled-vote analysis is preserved separately.
6. **Development set:** the seeded 70 disputed + 20 unanimous validation questions are approved.
   The two groups are reported separately. Raw accuracy on this enriched sample is not benchmark
   accuracy. Fresh evaluation questions (positions 321+) stay untouched.
7. **Structured prompt.** The order stays decisive issue → check → decision, with the same
   150-word guidance. "Change only if you identify an error" is replaced by: *"Choose the answer
   best supported by your check. Explain the specific evidence or reasoning supporting your
   decision; agreement alone is insufficient."* The optional forced opening is `Decisive issue:`.
8. **Controls and curves.**
   - Structured reconsideration *without peers* is a priority control.
   - Six-point prediction curves are budgeted for the newly generated structured responses and
     the matched current-debate controls, not just for existing responses.
   - Measured tokens and time are reported: equal generation counts are not equal computational
     cost.
9. **Actual checking.**
   - The same ~40 question–agent cases are coded in every condition, selected before any new
     output is viewed. Any extra changed-answer sample is analyzed separately.
   - A 20% subset is prepared for independent second coding. It is not assumed that anyone will
     do it, and nothing is blocked on it.
   - "Beat the better of two baseline samples" is replaced by a prespecified paired comparison
     with uncertainty (§C.4 as amended below).
10. **Storage.**
    - No activation arrays for the intermediate curve readouts.
    - The existing activation collection is kept for new debate responses.
    - Archives are verified and the pilot's second copy is created.
    - An off-device backup destination is identified; if none is configured, that is reported
      without blocking CPU work.
11. **Stage 4 stays deferred.** Its sample requirement is re-estimated after the corrected voting
    analysis. The 85-discordant-question estimate and the 20% threshold are provisional planning
    assumptions, not universal requirements.

### Amended Stage 3 design and gate (replaces the corresponding parts of §C and §Gates)

**Conditions.** All on the 90 development questions, reusing the archived round-0 answers:

| condition | source |
|---|---|
| current debate, sample 1 | existing val12b revisions |
| current debate, sample 2 | same-session re-run |
| **structured debate** | new |
| **structured reconsideration without peers** | new; priority control |
| plain reconsideration | existing val12b-self for the first 100 questions; generated for the rest |
| independent answers | existing |
| *optional:* forced opening `Decisive issue:` | new |

**Curves.** Six-point prediction curves (and the bridge readouts) for:

- current-debate sample 2;
- structured debate;
- structured reconsideration;
- the forced opening, if it is run.

Existing sample-1 and reconsideration responses are covered by Stage 2.

**Prespecified paired comparison.** Primary contrast: structured debate − current debate
sample 2 (same session).

| outcome | unit and method |
|---|---|
| **checking** | on the preselected ~40 question–agent cases: the paired difference in the share of revisions coded "decisive point engaged and actually checked" (codebook of the transcript review), with a case-cluster bootstrap 95% interval |
| **accuracy** | the per-question paired difference in mean per-agent revised accuracy, reported separately for the 70 disputed and 20 unanimous questions, with a question-bootstrap 95% interval. Also the final-panel vote accuracy (exact random-tie expectation) |
| **revisions** | expected helpful and harmful counts per question, paired, with intervals |
| **cost and failure** | truncation share, parse failures, generated tokens and GPU seconds per revision (measured) |

Secondary contrasts:

- structured reconsideration − plain reconsideration (does structure help without peers?);
- structured debate − structured reconsideration (does peer exposure add anything once
  structured?);
- current sample 2 − sample 1 (run-to-run noise).

**Gate to a fresh-question confirmation** (separate approval). All four must hold:

- (i) the checking difference is positive and its 95% interval excludes 0;
- (ii) the disputed-question accuracy difference is ≥ 0, with a lower 95% bound ≥ −0.05
  (non-inferiority margin, fixed now);
- (iii) truncation ≤ 5% of structured revisions;
- (iv) measured tokens per revision are reported. There is no fixed cost ceiling, but a cost
  above 1.5× current must be justified.

Otherwise the result is reported as a negative or inconclusive development result.

### Amended GPU estimates (estimates; Stage 3 now includes new conditions and their curves)

| stage | A100-h | units at 5.30/h | cap |
|---|---|---|---|
| 1 stress test | 0.4–0.7 | 2.1–3.7 | 6 |
| 2 curves on existing responses (core) | 0.6–1.1 | 3.2–5.8 | 8 |
| 2+ extra conditions / paper controls / re-tokenization comparison | +0.3–0.6 | +1.6–3.2 | +4 |
| 3 development run | 1.5–2.4 | 8–13 | **15** |
| 3+ forced opening (+ curves), 2-round subset | +0.5–0.9 | +2.7–4.8 | +5 |
| **phase total** (minimum / with Stage 3 / everything) | **1.0–1.8 / 2.5–4.2 / 3.3–5.7** | **5–10 / 13–22 / 17–30** | **38** |

**Stage 3 breakdown** (90 questions):

| item | estimate |
|---|---|
| current-debate re-run | ≈ 16 min (measured 10.5 s/question) |
| structured debate | 16–24 min |
| structured reconsideration | 12–20 min |
| plain reconsideration for ≈ 44 questions | ≈ 11 min (measured 14.8 s/question) |
| post and exposure readouts with activations | 5–8 min |
| curves: 3 new conditions × 270 responses × 8 passes ≈ 6,500 passes | 12–24 min (at 4.6–9 passes/s) |
| overhead and backup | 20–30 min |

## Goals

1. **Commitment curves.** Reproduce the within-response prediction curves of *Who Convinced
   Whom?* (Batra et al., arXiv:2610.35930, Figure 1, p. 5) on Gemma-3-12B / MMLU-Pro. Reuse our
   saved responses; regenerate nothing.
2. **Structured debate.** Test whether more structured debate prompts increase actual checking
   and accuracy.
3. **Next action.** Assess whether we can predict when another debate round, or more independent
   answers, will help.

## Summary of resources (estimates unless marked measured)

| stage | what | paid GPU (A100-h) | Colab units at 5.30/h | hard cap (units) |
|---|---|---|---|---|
| 0 | CPU: archive audit, saved-probability analyses, selection-rule fix, code | 0 | 0 | 0 |
| 1 | GPU timing/stress test (readout + structured-prompt smoke) | 0.4–0.7 | 2.1–3.7 | **6** |
| 2 | GPU readout of existing responses (commitment curves), core | 0.6–1.1 | 3.2–5.8 | **8** |
| 2+ | optional extra conditions and controls | +0.3–0.6 | +1.6–3.2 | **+4** |
| 3 | small structured-debate development run (~90 questions) | 1.0–1.8 | 5.3–9.5 | **12** |
| 3+ | optional forced opening and 2-round subset | +0.5–0.8 | +2.7–4.2 | **+5** |
| **phase total** | minimum (0–2) / with C (0–3) / everything | **1.0–1.8 / 2.0–3.6 / 2.8–5.0** | **5–10 / 11–19 / 15–27** | **35** |
| 4 | next-action prediction on fresh questions (conditional; separate approval) | ≈ 27–39 (see §D.3) | ≈ 140–210 | not authorized |

- **Rate:** 5.30 units per A100-hour (measured 2026-10-13/14).
- **Balance:** the last recorded balance was 258.86 units (2026-10-14). It must be re-checked
  before Stage 1.
- **Elapsed time:** about 2–3 weeks of calendar time (§Timeline). The paper is due in early
  December.

---

## A. Archive verification and CPU analysis

### A.1 Inventory

Checked by listing and reading manifests on 2026-10-29. The full integrity audit is task A0.

| item | local (laptop WSL disk) | second copy | on GitHub |
|---|---|---|---|
| pilot pilot12b: round 0 + debate, self_reconsider, debate_answer_only, misleading_one (120 q) | `runs/colab/pilot12b-*` (3.6 GB, 24 chunk dirs + 4 views) | **none found** | transcripts, agent_states, activation index/meta, round-0 jsonl under `artifacts/pilot12b/data/` (no activation arrays) |
| validation val12b: round 0 + debate + independent (200 q) | `runs/val12b/val12b-*` (≈3.5 GB) | `C:\Users\mmcph\mas_backup\val12b` (verified 10-14) | no transcripts or states; only derived outputs (`artifacts/val12b/frozen/` states.csv, scores, decisions; investigation; audit) |
| val12b-self: self_reconsider on the first 100 validation q | `runs/val12b/val12b-self-*` (≈0.65 GB) | same backup folder (verified 10-14) | derived outputs only (`artifacts/val12b/frozen_self/`) |
| frozen scorers | `runs/frozen/pilot12b_scorers_3857a72.pkl` | `C:\...\mas_backup\frozen` (SHA-256 8e5785fe…) | no (pickle) |
| analysis caches | `runs/cache/{probe12b,val12b,val12b_self}` (16 GB) | none; rebuildable from the archives | no |
| code, docs, compact results | working tree | GitHub `review-9e45160-fixes` | yes |

**What each archive holds.** Per run view:

- **`transcripts.jsonl`:** the messages, the full templated prompt string (with `<bos>` and the
  generation prompt), the response text, the generated `token_ids`, `gen_start_char`,
  `stop_reason`, and the shown-peer sources and answers.
- **`agent_states.parquet`:** three phases per agent and condition: `pre` (round 0), `post`
  (revision) and `peer_exposure` (prompt-only readout).
  - **Answer probabilities:** `readout_logprobs` (10 letter log-probs, stored as a comma
    string); `pos_before_stated_answer_logprobs` and `pos_stated_answer_logprobs`.
  - **Answers:** stated answer, source (stated or readout fallback), correctness.
  - **Other fields:** confidence, token counts, `replay_exact`.
- **Chunk directories:** `acts_*.npy` (bf16, all 49 hidden-state layers × 3840) at
  `readout_last`, `generation_last`, `stated_answer` and `before_stated_answer`, with index and
  meta parquets and SHA-256s in `completion.json`.
- **Model and tokenizer settings:** recorded in the job identity:
  - `google/gemma-3-12b-it` at commit `96b6f1ec…`;
  - chat template SHA-256 `7de1c58e…`;
  - transformers 5.17.0;
  - bf16; temperature 0.7, top-p 0.95; 768 new tokens.

  Pilot, validation and the self child all show the same model commit and chat-template hash.
  The code commits differ: pilot `5ff52b3`, validation `2621a96`.

**Spot checks already made** (inventory only, not the audit):

- Every view has the expected state counts. For example, val debate has 600 states per phase.
- Every eligible revision has a stored `before_stated_answer` letter distribution.
- In 592 of 600 val debate revision prompts, the shown peers' round-0 text appears verbatim. The
  8 exceptions must be explained in A0 (likely whitespace or truncation).

**Measured eligible responses.** A response is eligible when both its revised and previous
answers are *stated*, not readout fallbacks. Changed answers are counted among eligible
responses.

| pool / condition | revisions | eligible | changed (helpful / harmful / wrong→wrong) | questions with a change |
|---|---|---|---|---|
| pilot debate | 360 | 342 | 99 (29 / 30 / 40) | 44 |
| pilot self_reconsider | 360 | 341 | 170 (34 / 77 / 59) | 84 |
| pilot debate_answer_only | 360 | 342 | 92 (28 / 24 / 40) | 47 |
| pilot misleading_one | 360 | 342 | 150 (27 / 56 / 67) | 82 |
| val debate | 600 | 571 | 152 (50 / 41 / 61) | 68 |
| val self_reconsider (first 100 q) | 300 | 285 | 138 (32 / 61 / 45) | 62 |
| val independent (fresh sample, not a revision) | 600 | 561 | 132 differ from the agent's first sample | 79 |

**Sequence lengths** (prompt + response, measured): median about 1,000 and at most 3,711 tokens
in debate; median 670–750 in self-reconsideration.

### A.2 Already completed; do not repeat

From docs/PROBE_ACTION_TRANSCRIPT_REVIEW.md (2026-10-28) and earlier documents:

- **Probe estimator audit** on fixed layers: pilot CV, validation, paired comparison against the
  output models, learning curves, refit stability, calibration. The peer-context score shift is
  documented there and in docs/R1_INVESTIGATION.md.
- **Action choice at the round-0 boundary**: stop, independent or debate on the 200 validation
  questions, with simple and learned policies. It used the pooled-vote rule audited in A.4. Only
  that rule-dependent part is redone (A3).
- **Transcript review**: a seed-2026 sample of 30 cases, hand-coded with a codebook, plus
  population counts over all 600 val debate revisions: switch rates by peer agreement,
  deference wording, confidence changes, capped peers. The prompt recommendations were *not*
  tested; Stage 3 tests a subset of them.
- **Exposure-state probes for correctness**: docs/PEER_EXPOSURE.md (1B, ARC) and
  docs/PILOT_ANALYSIS.md (12B pilot, gates). These predicted whether the final answer is
  *correct*. Neither measured whether the exposure readout already holds the *eventual* answer,
  which is A2.

### A.3 What the saved probabilities alone can answer (task A2, CPU)

**What a "0% point" means here.** The `peer_exposure` readout is the exact revision prompt, with
peers, instruction and generation prompt, followed by `\nFinal answer: (`. It is read before the
agent writes anything.

**The analysis.** For every eligible revision, compute:

- whether the exposure argmax equals the eventual stated answer ("committed at read");
- whether it equals the agent's previous answer;
- whether it equals the answer key;
- p(eventual) and p(previous).

**Splits.** Report these by condition (debate, self_reconsider; pilot answer-only and misleading
as extras) and by outcome (unchanged / changed; helpful / harmful / wrong → wrong), with
question-cluster bootstrap intervals. Report pilot and validation separately, then pooled.

**Paired debate-minus-reflection difference.** This uses questions with a change in both
conditions: the 120 pilot questions and the first 100 validation questions, where both
conditions exist. Report the mean and a sign test, as in the paper's Table 9.

**Echo check.** When the exposure argmax is a peer's answer, is that peer's text the last thing
before the readout point? This tests the "echoes the nearest letter" explanation flagged in
docs/PEER_EXPOSURE.md.

**What this gives early.** The paper's headline quantity (f = 0) on our data, before any GPU
work.

**Caveat.** It uses our suffix `\nFinal answer: (` rather than the agents' own answer format,
so it is not identical to Stage 2's f = 0. Stage 2 measures both, to bridge them.

**Sanity anchor at 100%.** `pos_before_stated_answer_logprobs` is the letter distribution at the
`(` the agent itself wrote after its full reasoning. It has never seen the letter, so it is a
natural 100% point with the agent's own prefix. Its argmax should equal the stated answer almost
always. A post-response readout (`readout_last`, after `Answer: (X)\nConfidence: NN`) has already
seen the answer, and is **not** used as the 100% point.

### A.4 Audit of the action-choice selection rule (task A3, CPU)

**What the rule did.** scripts/action_choice.py gave each action's final answer as a plurality
vote over *all* answers available: 3 for stop, 6 for independent and debate. Ties went to the
round-0 plurality, then to agent order.

**Mechanical consequences.**

- **Unanimous panels can never change.** A unanimous round-0 panel (X, X, X) can never change
  under independent or debate. Three new answers give another option at most 3 of 6 votes, and
  a 3–3 tie goes back to X. That accounts for 130 of 200 validation questions.
- **2–1 panels need a unanimous reversal.** A 2–1 panel (X, X, Y) changes only if all three new
  answers are Y (Y 4, X 2); Y, Y, X gives a 3–3 tie, which returns X.
- **Debate answers are counted twice.** The pooled rule counts debate revisions as three extra
  *independent* votes, although they come from the same agents conditioned on round 0. This
  double-counts the original panel and pulls debate toward its starting point.

**How this affects the conclusions.**

| previous conclusion | revision |
|---|---|
| "only 10 of 200 questions distinguish the actions" | partly a property of the rule: 65% of questions were excluded by construction |
| "unanimous → stop" | was built into the rule, not learned from data |
| the learned-policy comparison | effectively used only the 70 disputed questions |
| debate's 116 vs independent's 120 | the rule disadvantages debate specifically. Under a revised-answers-only vote, debate scored 119 (already in `action_choice.json`) |

The qualitative conclusion, that the sample is too small to learn a policy, probably survives.
Its stated magnitude does not.

**Proposed rule for future comparisons.** Each action's final answer is the plurality of the
panel that action produces, with compute matched by the number of generations:

| action | final answer | generations |
|---|---|---|
| stop | vote of the 3 round-0 answers | 3 |
| debate | vote of the 3 revised answers (the standard multi-agent-debate final answer, as in the paper) | 6 |
| independent | vote of all 6 independent samples (the paper's "equal-compute vote") | 6 |

**Tie handling.**

- **Primary:** expected accuracy under uniformly random tie-breaking, computed exactly by
  averaging over the tied options. No preference for round 0.
- **Sensitivities:** (a) the round-0 tie-break; (b) per-agent accuracy, which needs no vote.

A3 recomputes §2 of the review under this rule, as a re-analysis of seen data. The rule is fixed
in this plan *before* those numbers are computed.

### A.5 Stage 0 tasks (CPU)

| id | task | effort (estimate) | CPU runtime (estimate) |
|---|---|---|---|
| A0 | Archive integrity. Re-hash every shard and view against `completion.json` / `view.json` (pilot, val, val-self). Make a hash-verified second copy of the pilot archive in `mas_backup`, since none exists. Propose an off-device third copy (§Storage). | 2–3 h | 15–30 min |
| A1 | Alignment audit. For every state, check: <br>• the prompt string re-encodes to the readout prompt ids; <br>• `replay_exact` holds; <br>• the stated-answer parse matches `answer`; <br>• the letter-token alignment holds; <br>• shown peer texts equal the peers' round-0 texts (explain the 8/600); <br>• `peer_exposure` count = `post` count; <br>• activation index ↔ state keys; <br>• model commit and chat-template hash identical across jobs. <br>Output: `artifacts/next_phase/archive_audit.json` with an inventory table. | 4–6 h | 10–20 min |
| A2 | Committed-at-read from saved probabilities (§A.3). | 3–4 h | < 5 min |
| A3 | Selection-rule re-analysis (§A.4). | 2 h | < 2 min |
| A4 | Fix the Stage 3 development set and manifest (§C.3) by a seeded rule written before any new generation. | 1 h | < 1 min |
| A5 | Code for Stages 1–3, with CPU/stub tests (§Code changes). | 2–4 days | tests ≈ 5 min |

**Stage 0 gate.** Proceed to Stage 1 only if all of these hold:

- every hash verifies;
- ≥ 99% of eligible revisions pass the replay and letter-alignment checks;
- every peer-text mismatch is explained;
- the pilot second copy is verified;
- the Stage 1 code passes its tests.

Otherwise, fix the problem or stop and report.

---

## B. GPU readout of existing responses: commitment curves (Stage 2)

### B.1 Method: following the paper's code

The paper's code (`code/interp/commit_lit.py`, MAS-Interp repository) does the following:

- **Reasoning:** the response text before the final-answer span, right-stripped.
- **Truncation:** at fraction f ∈ {0, .1, .25, .5, .75, 1} *of its length in characters*, cut
  back to the last space before `int(len × f)`.
- **Context:** chat-templated round message + generation prompt + `body + "\n\n"` (if the body is
  non-empty) + `{final answer: (`.
- **Readout:** next-token log-probabilities over the option letters; the analysis uses the
  argmax.
- **Exclusions:** responses whose answer or previous answer did not parse are excluded.
- **Unchanged answers:** a 6% hash sample.

How we resolve each point for our data:

| issue | decision (proposed) |
|---|---|
| final-answer-line removal | Reasoning = response text before the start of the line holding the final parsed `Answer: (X)` field, right-stripped. This removes the `Confidence: NN` line after it too. An earlier mention of a letter inside the reasoning ("so the answer is (C)") is kept, as in the paper, and flagged so a sensitivity analysis can drop those responses. |
| answer prefix | **Primary:** the agents' own format, `Answer: (`. At f > 0 the context is reasoning + `"\n\n"` + `Answer: (`; at f = 0 it is the generation prompt + `Answer: (`. This copies the paper's principle (the prefix matches the response format) and its separator. Our responses end `...\n\nAnswer: (X)`, so at f = 1 the context reproduces what the agent saw when it wrote its letter. <br>**Bridge:** also read f = 0 and f = 1 with our existing suffix `\nFinal answer: (` (2 extra passes per response) to connect to A2 and the saved readouts. |
| token alignment | Our pipeline builds readout inputs from **token ids**, not re-tokenized text (re-tokenizing can merge tokens at the join). <br>• The prompt is the stored prompt string, encoded as in the original readout. <br>• The body is the longest prefix of the stored generated ids whose decoding is at most the character cut. Gemma attaches a leading space to the following word, so a cut before a space falls on a token boundary. <br>• The answer prefix is encoded separately. <br>• Every mismatch between `decode(ids[:k])` and the character cut is logged. <br>**Difference from the paper:** the paper re-tokenizes. The stress test compares the two on a subset. |
| missing answers | Excluded, as in the paper: revisions whose own or previous answer is a readout fallback, or did not parse (measured: 18–29 per condition, §A.1). Counts are reported per condition. |
| truncated (capped) responses | They have no final-answer line, so they are excluded from the replication, which also follows the paper. They are 2 of 600 val debate revisions but more of the round-0 answers, so a previous-answer exclusion also removes some revisions. Optional: a separate descriptive curve treating the whole capped text as reasoning. |
| reuse of existing 0% | Not reused as the primary f = 0, because it uses a different prefix (`\nFinal answer: (`). It is reused as the bridge and in A2. The stress test checks that re-reading it reproduces the saved values (§Stage 1). |
| 100% point | Measured fresh with the primary prefix. The saved `before_stated_answer` distribution is the same context with the agent's own `Answer: (`, so it is a **validation check**, not a substitute. A post-response readout (`readout_last`) is never used as the 100% point. |
| unchanged answers | Read **all** eligible responses, not a 6% sample (affordable). The paper's 6% comparison can be recovered by subsampling. |
| reflection condition | Our `self_reconsider`: same agent, its own previous response, instruction to reconsider, no peers. This is the closest analog of the paper's "reflection" (round message with peers removed). |

### B.2 Scope

**Core** (debate + self_reconsider, both pools; 1,539 eligible responses, measured):

- pilot debate 342;
- pilot self 341;
- val debate 571;
- val self 285.

Each response gets 6 fractions + 2 bridge passes, about **12,300 forward passes**.

**Optional extensions:**

- **(a) More conditions:** pilot debate_answer_only (342), pilot misleading_one (342), and val
  independent (561; f = 0 is the bare question, a "no context" baseline). About 10,000 passes.
- **(b) The paper's controls at f = 0 on changed responses:**
  - neutral filler of matched token length;
  - another question's reasoning from the same agent and subject.

  559 changed core responses × 2, about 1,100 passes.
- **(c) Round-0 responses.** Their f = 0 is the question alone. They show the curve with no prior
  answer. About 5,000–5,800 passes.

### B.3 Outputs and analysis

**Rows.** One row per (response, fraction, prefix): 10 letter log-probs plus keys. No
activations by default (§Storage).

**Curves**, per fraction, each with 95% intervals from 2,000 question-cluster bootstrap
resamples (all observations of a resampled question kept together):

- (i) argmax = eventual answer ("agreement");
- (ii) argmax = answer key ("correctness");
- (iii) argmax = previous answer;
- the mean probability of each.

**Subsets:**

- condition (debate, reflection; extras);
- changed vs unchanged;
- among changed: helpful, harmful, wrong → wrong;
- all responses;
- pool (pilot, validation, pooled).

**Paired statistic:** the debate − reflection gap at f = 0, per question, over questions with a
change in both conditions, with a sign test (paper Table 9).

**Labelling.** This is a *replication on Gemma-3-12B-it and MMLU-Pro*. It differs from the paper
in:

- **model and data:** a different model; 10-option knowledge questions only;
- **rounds:** one revision round, not three;
- **decoding:** temperature 0.7 / top-p 0.95 / 768 tokens with a 150-word instruction, not
  1.0 / 0.9 / 2,048;
- **answer format:** `Answer: (X)` + confidence, not `{final answer: (X)}`;
- **tokenization:** token-id construction instead of re-tokenizing;
- **unchanged answers:** all read, not a 6% sample;
- **sample size:** about 560 changed core responses against the paper's thousands.

Every one of these is reported, not assumed away.

---

## C. Small structured-debate development experiment (Stage 3)

### C.1 Prompts (proposed text; hashed into the manifest like every template)

`STRUCTURED_BLOCK` replaces the current `PEER_BLOCK` instruction. The peer presentation is
unchanged, to isolate the instruction:

```
Here are the responses from the other members of your panel:

{peers}

Before giving your answer again, do three things.
1. Decisive issue: If the answers above (including yours) differ, name the single step, fact or
   calculation that decides between them. If they all agree, write "No substantive
   disagreement" and name the step most likely to be wrong.
2. Check: Redo that step yourself. For a calculation, recompute it and state the number you get.
   For a fact or definition, state what you know and how sure you are. Another panel member's
   statement that something is correct is not a check.
3. Decision: Choose the answer best supported by your check. Explain the specific evidence or
   reasoning supporting your decision; agreement alone is insufficient.
{brevity} {ending}
```

**Forced opening** (optional, separately identified condition `debate_structured_open`): the
same prompt, with the assistant turn prefilled with `Decisive issue:`. Generation continues from
that prefix, and the prefix is stored as part of the response text. The brief's example wording
("The decisive issue is…") is an alternative. Which one is used is a decision for review.

**Kept fixed:**

- the brevity line (150 words);
- the 768-token cap;
- the answer and confidence ending;
- decoding settings.

The only manipulation is the instruction. Truncation is measured, not prevented.

### C.2 Conditions (smallest useful design)

All conditions reuse the **identical saved round-0 answers** of each development question.

| condition | source | new generation? |
|---|---|---|
| current debate, sample 1 | existing val12b debate revisions | no |
| current debate, sample 2 | re-generated in the Stage 3 session with the same prompts | yes. It gives a same-session control and a measured noise floor between two runs of the *same* condition |
| structured debate | new | yes |
| reconsideration alone | existing val12b-self (first 100 q); generated for development questions outside them | partly |
| additional independent answers | existing val12b independent (3 fresh samples) | no |
| *optional:* forced opening | new | yes |
| *optional:* a second round for current vs structured debate on a ~30-question disputed subset | new; **needs multi-round code**, since `n_rounds` is fixed at 1 | yes |

### C.3 Development set (fixed in A4, before generation)

**Proposal: 90 validation questions.**

- all 70 questions with a disputed round-0 panel;
- 20 unanimous questions drawn with seed 2026, so the "No substantive disagreement" branch is
  exercised and harm to unanimous-correct panels can be measured.

**Why disputed questions dominate.** Nearly all informative changes happen there: when both
peers agree with the agent it switches in 4 of 376 cases (measured, review §3.2).

**Cost of this choice.** Validation questions become development questions. They are already
exploratory, so nothing confirmatory is lost. Fresh positions 321+ remain untouched.

**Reconsideration coverage.** 36 of the 70 disputed questions are in the first 100, so
reconsideration exists for them. The rest (≈ 45 questions with the unanimous draw) need
self_reconsider generation.

### C.4 Outcomes

None of these is "the prose is more organized".

1. **Accuracy:**
   - per-agent revised accuracy;
   - panel accuracy under the §A.4 rule;
   - per question, paired against current debate.
2. **Helpful and harmful revisions** (counts, net), and the switch rate when both peers disagree.
3. **Actual checking.** Use the codebook from the transcript review: decisive point engaged,
   actual check (none / partial / full / repeat), check correct, asserts correctness without
   checking, accepted a wrong number.
   - **Sample:** a stratified random sample of about 40 revisions per condition: all changed
     answers on calculation questions plus a random fill.
   - **Blinding:** prompts are stripped before coding. Full blinding is impossible, because
     structured responses look different; this is stated.
   - **Second coder:** a 20% subset (e.g. you) for agreement.
   - **Automated proxies (secondary):** an equation is written out; the stated decisive issue
     matches the actual disagreement.
4. **Truncation:** share of revisions at the cap; parse failures.
5. **Computational cost:** generated tokens per revision; GPU seconds per question.

**Power (estimate).** On the 70 disputed questions, current debate produced 49 helpful and 40
harmful eligible revisions (measured; the 130 unanimous questions produced 4 changes in all).

- A development run can detect only large effects, e.g. **halving harmful revisions** (40 → 20:
  z ≈ 2.6 under a Poisson approximation).
- Smaller effects are reported as estimates with intervals. The two current-debate samples show
  the run-to-run noise.

---

## D. Conditional expansion: next-action prediction (Stage 4)

### D.1 Design

- **Round boundaries:** at each saved round boundary, compare three actions: **stop**, **another
  debate round**, **three additional independent answers**. Final answers follow the §A.4 rule.
  Start with one-step choices.
- **Predictors:** compared by question-grouped cross-validation (all records of a question in
  one fold) within development data, then on fresh questions once:
  - visible features: agreement pattern, stated confidences, answer probabilities, lengths and
    caps, peer agreement;
  - activation features: frozen or refit probes on the boundary states;
  - the two combined.

  They may use only information available before the action.
- **Baselines:** always-X policies, disagreement rules, confidence thresholds. The hindsight-best
  action is shown only as an optimistic reference.

### D.2 What existing data supports

| boundary | data | supports |
|---|---|---|
| after round 0 | val12b (200 q): stop, debate, independent; pilot (120 q): stop, debate, self, no independent | development only, already done (to be redone under the new rule, A3). Measured: 10 of 200 questions distinguished the actions under the old rule |
| after round 1 (another debate round vs more independent answers) | **none**: every collection has one revision round | needs new multi-round collection |
| structured debate as an action | Stage 3 (90 dev q) | development only |

### D.3 How many informative differences are needed (estimate)

**Informative questions.** A policy can beat the best fixed action only on questions where the
actions give different final correctness.

**Test-set size.** Detecting a 65/35 split on such discordant questions against 50/50 (paired
sign test, 80% power, two-sided 5%) needs **about 85 discordant questions in a fresh test set**.
A similar number is needed for training. That is roughly **170–250 informative questions in
total**.

**Yield per question.**

| rule / scope | informative rate |
|---|---|
| old rule, all questions | ≈ 5% (10 of 200) |
| old rule, disputed questions | ≈ 14% (10 of 70) |
| new rule | possibly 15–25% of disputed questions (unknown until A3) |

**Collection this implies.**

- About 1,200–1,800 disputed questions, i.e. **3,500–5,000 round-0 questions** at the 35%
  disputed rate (measured, validation).
- Measured costs: 18.2 s per question for round 0; 10.5 s per question for debate and 17.0 s for
  independent generation.
- Running actions only on disputed panels, that is round 0 at 17.7–25.3 h plus actions at
  9.2–13.8 h: about **27–39 A100-hours (≈ 140–210 units)**, before any second round.
- **This exceeds the phase budget.**

### D.4 Recommendation and gate

**Defer Stage 4.** The CPU part runs on development data (A3 plus Stage 3 outputs). Request a new
collection only if both hold:

- (a) the new rule or a Stage 3 action raises the informative rate to **≥ 20% of disputed
  questions** on development data;
- (b) a fresh-question collection of ≥ 85 expected discordant test questions fits a separately
  approved budget.

**Fine-tuning and reinforcement learning.** Language-model fine-tuning or RL for the action
choice stays deferred. They would cost far more than the collection above. With about 10 informative questions
there is no training signal, so there is no concrete reason for them now.

---

## Task table

| id | task | depends on | data reused | new data | deliverables | code changes |
|---|---|---|---|---|---|---|
| A0 | archive integrity + pilot second copy | — | all archives | none | `artifacts/next_phase/archive_audit.json`, backup log | `scripts/archive_audit.py` (wraps `verify_archive`) |
| A1 | alignment audit | A0 | transcripts, states, activation index, job identities | none | audit section of the same JSON; short note in the results doc | same script |
| A2 | committed-at-read from saved probabilities | A1 | `peer_exposure` and `before_stated_answer` log-probs | none | `artifacts/next_phase/exposure_commit/`, curves table at f = 0 | `scripts/exposure_commit.py` |
| A3 | selection-rule re-analysis | — | `action_choice/question_records.csv` + states | none | `artifacts/val12b/action_choice/` (new rule), note | `scripts/action_choice.py --rule {pooled,final_panel}`; exact random-tie expectation |
| A4 | development-set manifest | — | val12b states | none | `artifacts/next_phase/dev_set.json` | small script |
| A5a | readout-at-fractions code | A1 | transcripts | none | tested code | `src/mad/backends/hf.py`: letter-only readout from token ids (no hidden states); `scripts/commit_curves.py` (`build` / `run` / `analyze`); stub backend support; tests for answer-line removal, the char→token cut, the f = 1 ≡ `before_stated_answer` identity on the stub, bridge prefixes, exclusions |
| A5b | structured-debate code | — | val12b round 0 | none | tested code | `src/mad/prompts.py`: `STRUCTURED_BLOCK` (hashed), opening prefix; `src/mad/multi.py`: kinds `debate_structured`, `debate_structured_open`, assistant-prefill generation; **reuse of round 0 from an existing archive** (`--round0-from runs/val12b`, keyed and hash-checked); tests |
| A5c | *(optional)* multi-round | — | — | — | tested code | lift `n_rounds == 1` in `config.py` / `multi.py`: rounds ≥ 2, per-round states and exposure; tests. Largest code change (2–3 days) |
| S1 | GPU stress test | A0, A1, A5a, A5b | 40 + 20 responses; 3 dev questions | ≈ 30 MB | `artifacts/next_phase/stress/` report | none beyond A5 |
| S2 | commitment readout | S1 | eligible responses (§B.2) | ≈ 20–60 MB rows | `artifacts/next_phase/commit_curves/` (rows csv.gz, curves, paired stats), results doc | — |
| S3 | structured-debate development run | S1, A4 | dev-set round 0, existing debate / independent / self | ≈ 2–3 GB archive (with activations), transcripts | `runs/next_dev/`, compact outputs + coding sheets in `artifacts/next_phase/structured_dev/`, results doc | — |
| S4 | next-action prediction (conditional) | A3, S3, gate D.4 | everything above | fresh questions 321+ | separate plan | separate |

## Timing and cost details (estimates)

**Readout (Stage 2).** Measured readout throughput on the A100, with all hidden states written,
batch 4 (val debate chunk 0):

| readout | states | time | rate |
|---|---|---|---|
| post-response | 60 | 12.96 s | 4.6/s |
| exposure | 60 | 10.39 s | 5.8/s |

- **Assumption:** a letter-only readout skips hidden-state output and so is 1–2× faster, at
  4.6–9 passes/s.
- **Core:** 12,300 passes take 23–45 min.
- **Overhead:** setup, model load, the reproduction checks and backup add about 20–30 min.
- **Stage 2 core total:** 0.6–1.1 h. Extensions add 0.3–0.6 h.

**Generation (Stage 3).** Measured per-question costs:

| step | cost |
|---|---|
| debate generation | 10.5 s/question |
| independent generation | 17.0 s/question |
| self-reconsider child, all-in | 14.8 s/question |

- **Assumption:** structured revisions are 1.0–1.5× as long as current ones.
- **90 questions:** structured 16–24 min; current re-run about 16 min; self for about 45
  questions about 11 min; post and exposure readouts about 5 min; overhead and backup 20–30 min.
- **Stage 3 total:** 1.0–1.8 h.
- **Optional:** forced opening 16–24 min; second round on 30 questions × 2 conditions 15–25 min.

**Earlier sessions** spent 0.3–0.5 h of overhead on setup, smoke tests and backup. This is
included above. Colab A100 availability is not guaranteed: "Service Unavailable" occurred 3
times on 10-14.

**CPU** (laptop, 12 cores, 20 GB): all Stage 0 runs together take under 1 h of compute. Bootstrap
analyses of Stages 2–3 take minutes each.

### Timeline (estimate; calendar time includes review)

| step | elapsed |
|---|---|
| Stage 0 implementation and audit | 3–5 working days |
| Stage 1 session | ½ day, after approval |
| Stage 2 session + analysis | 1–2 days |
| Stage 3 code (part of A5) + session + coding and analysis | 3–5 days |
| **total** | **about 2–3 weeks**, leaving the rest of November for writing |

## First GPU step: bounded timing and stress test (Stage 1; not run)

**Cap:** **1.0 A100-hour, 6 units hard stop**. Expected 0.4–0.7 h (2.1–3.7 units).

**Session setup.** Colab A100 at a pushed, clean commit, matching the archives:

- model `google/gemma-3-12b-it` at commit `96b6f1ec…`;
- the chat template hash checked against `7de1c58e…`;
- transformers 5.17.0.

**Checks, each with a pass rule:**

1. **Reproduction of saved measurements.** 40 eligible revisions, stratified over the four core
   cells, are re-read.
   - (a) f = 0 with `\nFinal answer: (` vs the saved `peer_exposure` log-probs;
   - (b) f = 1 with `Answer: (` vs the saved `before_stated_answer` log-probs.

   **Pass:** argmax agreement ≥ 39/40 for each, and median |Δ log p| of the top letter ≤ 0.05.
   Saved values were bf16 batch-4 readouts, so small differences are expected.
   **Fail:** stop and debug the sequence construction on CPU.
2. **Token construction.** On the same 40, the token-id construction matches re-tokenized text.
   Argmax differences are reported, not gated.
3. **Batching.** Batched equals single-item readout (argmax identical; |Δ| ≤ 0.02).
4. **Memory and throughput.**
   - Run the 20 longest sequences (up to 3,711 tokens); peak memory must stay below 38 GB.
   - Measure passes/s at batch sizes 4, 8 and 16, and extrapolate Stage 2 time.
   - **Pass:** projected Stage 2 core ≤ 1.3 h.
5. **Structured-prompt smoke.** 3 development questions × 3 agents, reusing archived round 0,
   for `debate_structured` and `debate_structured_open`. Check:
   - parse rate;
   - truncation;
   - tokens per revision;
   - the reused-round-0 hashes.

   **Pass:** every response parses or its failure is explained, and the round-0 reuse verifies.
6. **Backup round trip.** Acknowledged backup of the outputs to the laptop and a hash check.

**Not done in Stage 1:** the full readout and any development generation beyond the 3-question
smoke.

## Storage, backup and activation retention

| output | size (estimate) | where | committed to git? |
|---|---|---|---|
| Stage 0 audit / A2 / A3 outputs | < 5 MB | `artifacts/next_phase/`, `artifacts/val12b/action_choice/` | yes |
| Stage 1 report | < 1 MB | `artifacts/next_phase/stress/` | yes |
| Stage 2 rows (letter log-probs only) | 20–60 MB uncompressed; 5–15 MB csv.gz | `runs/next_commit/` (raw), `artifacts/next_phase/commit_curves/` (compact) | compact only |
| Stage 3 archive | 2–3 GB (current pipeline: all 49 layers, 4 positions, bf16) | `runs/next_dev/` | transcripts + agent_states (≈ 10–20 MB, as for the pilot); no arrays |

**Activation retention.**

- **Stage 2 saves none** by default. The question is behavioral; letter log-probs suffice.
- **Optional:** layers 21/27/33 at the readout point for changed responses only. That is about
  0.1 GB, which would allow "commitment over the response" probes later. Decision for review.
- **Stage 3** keeps the pipeline's default, so Stage 4 could use internal features at the new
  boundaries. Reducing it to R1's layers would save about 80%. Decision for review.

**Destinations and recovery.**

- **Primary:** the laptop archive under `runs/`, via the acknowledged backup tool
  (`scripts/colab_backup.py`). It downloads each batch, hash-verifies it and acknowledges it;
  hung calls are retryable. Drive cannot be mounted from the CLI (failed twice), so it is not
  relied on.
- **Second copy:** a hash-verified copy in `C:\Users\mmcph\mas_backup\` after each session. The
  **pilot archive currently has no second copy**; A0 makes it.
- **Risk:** WSL and `C:` are on the **same physical laptop disk**. Proposed: an off-device third
  copy of the raw archives (≈ 8 GB now, plus ≈ 3 GB). Options are a OneDrive-synced folder or an
  external drive. Decision for review.
- **Recovery:** each chunk carries SHA-256s in `completion.json`, and views can be rebuilt from
  chunks. If a session dies, `--resume` continues from the last acknowledged chunk.
- **Stage 2** is cheap enough to re-run instead of being resumed.

## Minimum useful version, extensions, caps, gates

**Minimum useful version.** Stage 0 (A0–A4) + Stage 1 + the Stage 2 core.

- **What it delivers:**
  - the paper's commitment curves on our model and data, for debate vs reflection, changed vs
    unchanged, and helpful vs harmful, for agreement and correctness;
  - the saved-probability preview;
  - verified archives with a second pilot copy;
  - the corrected action-choice rule.
- **Cost:** about 1.0–1.8 A100-h (5–10 units).

**Optional extensions, in priority order:**

1. Stage 3 structured debate (+1.0–1.8 h).
2. The paper's filler and cross-item controls (+0.1 h).
3. Extra readout conditions (+0.2–0.5 h).
4. Forced opening (+0.3–0.4 h).
5. The multi-round subset (+0.3 h GPU, 2–3 days of code).
6. Stage 4, only via the D.4 gate.

**Caps.** Stage 1: 6 units. Stage 2: 8 units (+4 for extensions). Stage 3: 12 units (+5 for
extensions). **Phase total: 35 units (≈ 6.6 A100-h).** Every session stops at its own cap, even
mid-stage, with the outputs backed up.

**Gates:**

| after | proceed if | stop / rethink if |
|---|---|---|
| Stage 0 | hashes verify; ≥ 99% alignment; peer-text mismatches explained; tests pass | any unexplained misalignment, which would make the readouts uninterpretable |
| Stage 1 | reproduction checks pass; projected Stage 2 ≤ 1.3 h; smoke parses | reproduction fails, or the projection exceeds the cap → report, re-estimate |
| Stage 2 | always reported. Stage 3 does not depend on its result | — |
| Stage 3 → a fresh-question confirmation (separate approval) | on development data, structured debate shows **all** of: <br>• decisive-point engagement and actual-check rates ≥ 2× current; <br>• harmful revisions not above current; <br>• net (helpful − harmful) ≥ the better of the two current-debate samples; <br>• truncation ≤ 5% of revisions; <br>• ≤ 1.5× the tokens | no gain in checking, or lower accuracy → recommend no confirmation; report as a negative development result |
| Stage 4 | D.4 (a) and (b) | otherwise remains deferred |

## Remaining uncertainties and choices requiring review

1. **Answer prefix for Stage 2:** the agents' own `Answer: (` (proposed primary) vs our readout
   suffix `\nFinal answer: (`. The bridge passes measure both at f = 0 and f = 1 either way.
2. **Token-id construction vs the paper's re-tokenization** (proposed: token ids, with a
   comparison in Stage 1).
3. **Unchanged responses:** read all (proposed) or match the paper's 6% sample. All costs about
   20 min more and supports the "all answers" curves.
4. **Mentioned-letter responses:** whether to drop responses whose reasoning already names the
   final letter, as a sensitivity only (proposed) or in the primary analysis.
5. **Reflection analog:** whether `self_reconsider` is close enough to the paper's reflection.
   Its instruction differs ("Reconsider the question carefully on your own").
6. **Development set:** 70 disputed + 20 unanimous validation questions (proposed), or pilot
   questions (self_reconsider exists for all 120, but independent answers do not).
7. **Structured prompt wording,** and which forced-opening string to use; whether to keep the
   150-word limit (proposed) or relax it, which would confound structure with length.
8. **Checking measurement:** who codes; blinding is impossible to make complete; whether you will
   second-code 20%.
9. **Activation retention** in Stages 2 and 3 (proposed: none, and the pipeline default).
10. **Off-device backup** destination.
11. **Selection rule** (§A.4): confirm it before A3 is run.
12. **Uncertainties that cannot be settled in advance:**
    - Colab availability;
    - whether letter-only readouts are really faster than the measured 4.6 states/s;
    - the actual length of structured responses;
    - whether the 8/600 peer-text mismatches indicate a real alignment problem;
    - whether software on a new Colab image changes numerics. The pinned transformers version is
      reinstalled, and Stage 1 check 1 detects drift.
