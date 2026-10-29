# Next phase: implementation and CPU findings (2026-10-29)

Stage 0 of docs/NEXT_PHASE_PLAN.md, as approved with the review decisions recorded there. **No
GPU session was started and no GPU experiment was run.** Every analysis below uses
already-examined pilot (positions 1–120) and validation (121–320) questions, so all of it is
**exploratory**. Fresh questions (321+) were not touched.

Commands (all CPU):

```
python scripts/verify_archive.py --out-root runs/val12b --name val12b
python scripts/verify_archive.py --out-root runs/val12b --name val12b-self
python scripts/verify_archive.py --out-root runs/colab --name pilot12b --copy-to /mnt/c/Users/mmcph/mas_backup/pilot12b
python scripts/archive_audit.py                       # alignment audit -> artifacts/next_phase/archive/
python scripts/exposure_commit.py                     # saved-probability preview -> artifacts/next_phase/exposure_commit/
python scripts/action_choice.py --rule pooled --out <scratch>   # reproduces the 2026-10-28 analysis exactly
python scripts/action_choice.py                       # final-panel voting -> artifacts/val12b/action_choice/final_panel/
python scripts/dev_set.py                             # Stage 3 development set + checking cases
python scripts/commit_curves.py build --sources core extras     # GPU readout inputs, validated on CPU
```

## 1. Archives

### 1.1 Byte-level verification (`scripts/verify_archive.py`)

`verify_archive.py` checks every shard hash, rebuilds the views from the chunks, and loads every
activation and checks it is finite.

| job | result |
|---|---|
| val12b | 20 chunk directories verified |
| val12b-self | 5 chunk directories verified |
| pilot12b | 24 chunk directories verified |

**Pilot second copy.** The pilot had no second copy. One now exists at
`C:\Users\mmcph\mas_backup\pilot12b\` (3.6 GB). It was copied, verified there, and its
activations loaded identically to the original (`VERIFIED 24 chunk directories and a restored
copy`, 811 s).

**Second copies now held in `mas_backup`:** pilot, validation (both jobs) and the frozen scorers.

### 1.2 Alignment audit (`scripts/archive_audit.py` → `artifacts/next_phase/archive/archive_audit.json`)

Every check below was run on every archived response, and none failed.

| check | performed | failures |
|---|---|---|
| prompt re-renders from stored messages with the cached chat template | 5,880 | 0 |
| prompt token count = count recorded by the readout | 5,880 | 0 |
| saved generated ids decode to the stored text (replay) | 5,880 | 0 |
| re-parsed stated answer and status = recorded | 5,880 | 0 |
| revision messages rebuilt exactly from the peers' round-0 texts, in recorded order (debate) or with no peers (reconsideration) | 1,620 | 0 |
| one exposure state per revision, same keys | 2,940 / 2,940 | 0 |
| saved letter distributions finite and normalized | 8,820 readouts + 5,705 pre-letter | 0 |

**Negative control.** A tampered copy (one answer line misspelled, one revision prompt edited)
was flagged on all five relevant checks.

**The 8 of 600 peer-text mismatches from the planning spot check are explained.** That check
compared the peers' *unstripped* round-0 text, which still includes the `Confidence: NN` line.
The protocol strips that line before showing peers. The exact rebuild matches in all 1,620
cases.

**Identity.** The pilot, validation and self-reconsideration jobs share:

- model commit `96b6f1ec…`;
- chat-template SHA-256 `7de1c58e…`;
- transformers 5.17.0, torch 2.11.0+cu130.

The locally cached tokenizer is that same commit and template, so CPU-built readout inputs are
the inputs the GPU model will read.

**Minor provenance gap.** The pilot job did not record a dataset revision (validation pins
`b189ec76…`). It does not affect any analysis here.

**Capped responses.** 175 archived responses hit the 768-token cap, and **none** of them
contains a valid stated answer. The new rule (keep a capped response whenever it states an
answer) therefore changes nothing in the existing data. It is implemented for new runs.

### 1.3 Off-device backup

**No off-device destination is configured.** What exists:

- The primary archive (WSL disk) and the `mas_backup` copies (`C:`) are on the **same physical
  laptop SSD**.
- A OneDrive folder exists at `C:\Users\mmcph\OneDrive`, but whether it syncs cannot be
  verified from WSL.
- No external drive is visible.
- Google Drive mounting failed in both earlier sessions.

Nothing was copied to a cloud service. Copying research data there needs your confirmation.
This does not block CPU work.

## 2. GPU readout inputs prepared and validated on CPU (Stage 2, not run)

**Code:**

| file | role |
|---|---|
| `src/mad/curves.py` | construction |
| `scripts/commit_curves.py` | `build` / `run` / `analyze` |
| `HFBackend.letter_logprobs_ids` | letter-only readout, no hidden states |

**Build summary** (first round: one combined build; since §9.3 split into
`build_summary_core.json` / `build_summary_extras.json` and `responses_{core,extras}.csv.gz` in
`artifacts/next_phase/commit_curves/`, with identical totals), covering the core sets (pilot and validation debate and reconsideration)
and the extras (pilot answer-only and misleading, validation independent):

- **Eligibility:** 2,784 eligible revisions. Exclusions by condition and reason are in the
  JSON:
  - previous answer was a readout fallback: 13–27 per condition;
  - own answer was a readout fallback: 0–25.
- **Saved tokens:** every eligible response replays exactly from its saved ids. The stated
  letter sits on its own token in every response, so the exact replay is defined for all 2,784.
- **Cut boundaries:** 16,698 of 16,704 standardized cuts land exactly on a token boundary. The
  other 6 fall inside a two-space token and stop short by whitespace only (labelled
  `short_by_whitespace`).
- **Standardized 100% vs exact replay:** the two inputs are **token-identical for all 2,784
  responses**, because every agent wrote `\n\nAnswer: (`. Per decision 2 the two measurements are
  still kept separate. Here they coincide, so one GPU pass serves both, and the saved
  `before_stated_answer` distribution becomes a direct numerical check of the new readout.
- **The paper's re-tokenization** gives byte-identical inputs for **16,690 of 16,704** items.
  Only the 14 that differ need a GPU pass to compare the two constructions (`run --select
  retokenized`).
- **Answers named early:** in 48% of responses the reasoning already names the final option
  (e.g. "(C) is correct"), which is common in multiple-choice reasoning. This is recorded for a
  sensitivity analysis only.

**Workload** (unique forward passes after deduplication; minutes at an *assumed* 4.6–9 passes/s):

| set | passes | minutes |
|---|---|---|
| core (6 fractions + 2 bridge; replay shared with f = 1) | 12,312 | 23–45 |
| extras | 9,180 | 17–33 |
| re-tokenization comparison | 14 | < 1 |

The 4.6 passes/s floor is the measured rate of the full readout with all hidden states; a
letter-only pass should be faster. Stage 1 measures the real rate.

**Tests on CPU** (`tests/test_curves.py`):

- answer-line removal;
- the paper's word-boundary cuts;
- saved-id construction, including detection of inexact boundaries;
- bridge, replay and re-tokenized variants;
- eligibility.

On the real gemma-3-270m model, the exact replay and the f = 0 bridge reproduce the original
readout's saved distributions to within 2×10⁻⁴, including under batched left padding. A stub
run of `run` and `analyze` on the real 12B build exercised resume, deduplication and the
reproduction checks. The stub's numbers are meaningless and were not saved.

## 3. Saved-probability preview: committed at read (exploratory)

**What is measured.** For each eligible revision: does the argmax of the *saved* exposure
readout equal the answer the agent eventually stated? That readout was taken after the agent
read the round message and before it wrote anything (revision prompt + `\nFinal answer: (`).

Pooled over pilot and validation, with question-cluster bootstrap 95% intervals
(`artifacts/next_phase/exposure_commit/`):

| condition | changed answers | unchanged answers | helpful | harmful |
|---|---|---|---|---|
| debate | **0.82** [0.77, 0.87] (n = 251) | 0.96 (662) | 0.91 (79) | 0.76 (71) |
| reconsideration (no peers) | **0.41** [0.34, 0.48] (308) | 0.79 (318) | 0.49 (66) | 0.37 (138) |
| debate, answers only | 0.64 (92) | 0.96 (250) | 0.71 (28) | 0.63 (24) |
| one misleading peer | 0.57 (150) | 0.92 (192) | 0.89 (27) | 0.43 (56) |
| independent fresh sample (question only) | 0.28 (132) | 0.69 (429) | 0.34 (44) | 0.32 (38) |

The independent row is a "no context" baseline: its f = 0 input is the bare question.

**Paired gap (paper Table 9).** On questions with an answer change in both debate and
reconsideration, the per-question gap (debate − reconsideration) is:

| pool | gap [95% CI] | questions favouring debate / reconsideration (ties) | sign test |
|---|---|---|---|
| pilot | +0.39 [0.26, 0.50] | 27 / 2 (12) | |
| validation, first 100 questions | +0.43 [0.25, 0.60] | 19 / 4 (7) | |
| pooled | +0.40 [0.31, 0.50] | 46 / 6 (19) | p < 0.001 |

**Comparison with the paper.** It has the same direction as the paper's Figure 1:

- Qwen3.5-9B: 65 vs 22;
- Gemma-4-E4B: 48.5 vs 12.6.

Our levels are higher in both conditions.

**Unlike the paper, kept answers are not near-ceiling in reconsideration.** The paper reports
97–99% for kept answers in both conditions. Ours are 0.96 in debate but only 0.79 in
reconsideration. So the gap among *changed* answers sits on top of a generally less committed
reconsideration readout.

**Peer order affects both the readout and the decision.** Peer order is randomized per agent.
Among debate revisions whose two shown peers stated different answers:

| pool | exposure argmax = peer shown last | = peer shown first | final stated answer = last | = first |
|---|---|---|---|---|
| validation (130 revisions) | 95 | 33 | 91 | 34 |
| pilot (87 revisions) | 63 | 22 | 52 | 29 |

The readout favouring the last-shown peer could be an "echo of the nearest letter". But the
agents' *written* answers show almost the same asymmetry, so most of it is a real recency effect
on decisions, not only on the readout.

**Caveats.**

- **The prefix differs from the planned standardized readout.** These are readouts after
  `\nFinal answer: (`, not after `Answer: (`. Stage 2 measures both, which bridges them.
- **A cheaper explanation for part of the debate effect:** peer letters appear just before the
  readout point.

**Anchor at 100%.** The saved letter distribution at the agent's own `Answer: (` has its argmax
equal to the sampled letter in 99.6–100% of responses per condition. This was measured, not
assumed.

## 4. Action choice under final-panel voting (decision 5)

**Code.** `src/mad/analysis/actions.py`, called from `scripts/action_choice.py`:

- **Rules:** stop = vote of the 3 initial answers; debate = vote of the 3 revised answers;
  independent = vote of all 6 independent answers.
- **Ties:** uniformly random, scored *exactly* by expectation.
- **Fractional outcomes throughout:**
  - policies are trained with a fractional-label logistic model (each row enters as a weighted
    positive and negative);
  - fix and harm are reported as expected gains and losses;
  - no integer casts.

**Old analysis preserved.** It is in
`artifacts/val12b/action_choice/pooled_vote_2026-10-28/`, and `--rule pooled` reproduces it
question for question (116 / 120 / 116 / 119). Tests are in `tests/test_actions.py`, including
the equivalence with the original `panel_vote`.

**Results** (`artifacts/val12b/action_choice/final_panel/`; 200 validation questions; expected
correct):

| policy | expected correct | vs stop [95% CI] | expected gain / loss | extra answers |
|---|---|---|---|---|
| always stop | 117.7 | — | — | 0 |
| always independent | **121.8** | +0.021 [−0.008, +0.051] | 8.7 / 4.5 | 600 |
| always debate | 118.3 | +0.003 [−0.042, +0.048] | 12.0 / 11.3 | 600 |
| unanimous → stop, else independent | 121.3 | +0.018 [−0.010, +0.048] | 8.2 / 4.5 | 210 |
| unanimous → stop, else debate | 119.0 | +0.007 | 12.0 / 10.7 | 210 |
| confidence ≥ 95 → stop, else independent | 120.7 | +0.015 [−0.001, +0.035] | 3.8 / 0.8 | 96 |
| learned, visible features (5 CV seeds) | 119.2–120.3 | | | |
| learned, visible + internal (5 seeds) | 119.0–121.5 | | | |
| *hindsight best (optimistic reference)* | *133.8* | *+0.081* | *16.2 / 0* | *60* |

**How the corrected rule changes the earlier conclusions.**

- **More questions distinguish the actions.** Under the old rule 10 of 200 did; now 35 do:
  33 of the 70 disputed (47%) and 2 of the 130 unanimous (1.5%). The old figure was largely an
  artifact of the rule.
- **Debate moves much more than the old rule showed** (expected gain 12.0 and loss 11.3 against
  stop), but its net is about zero.
- **By initial agreement:** on 2–1 panels debate scores 22.0/51, against stop 18.0 and
  independent 20.3. On three-way splits it scores 0/19, against stop 2.7 and independent 4.0.
  This is a post-hoc pattern on 51 and 19 questions, recorded as a hypothesis only.
- **Learned policies are no better than always adding independent answers.** The conclusion
  that this sample cannot support a learned policy stands.

**Stage 4 re-estimate** (provisional; decision 11; `stage4_reestimate` in the JSON):

- **Rates:** 17.5% of all questions (47% of disputed ones) are informative under this rule.
- **The provisional "85 discordant test questions" assumption** then needs about 490 questions,
  or about 180 disputed questions if actions run only on disputed panels (≈ 520 round-0
  questions at a 35% disputed rate).
- **Alternative framing:** detecting a 3-point mean accuracy difference between two fixed
  actions takes about 400 (independent vs stop) to 930 (debate vs stop) questions, using the
  observed per-question SDs.

These figures replace the earlier 3,500–5,000-question estimate, which assumed the old rule's 5%
informative rate. They remain planning assumptions. Stage 4 stays deferred.

## 5. Stage 3 prepared (not run)

**Development set** (`artifacts/next_phase/dev_set.json`; seed 2026; fixed before any new
output exists):

- **Questions:** all 70 disputed validation questions + 20 unanimous (SHA-256 `438ccdd8…`).
- **Reconsideration coverage:** 45 of the 90 lack an existing plain reconsideration revision;
  their ids are in `dev_self_fill.json`.
- **Checking cases:** 40 question–agent pairs (32 disputed, 8 unanimous; 13 on calculation
  subjects), the same pairs in every condition.
- **Second coding:** 8 of the 40 are marked for independent second coding (prepared, not
  assumed).

**Prompts.** Templates in `src/mad/prompts.py`, hashed into every manifest:

- `STRUCTURED_PEER_BLOCK`: decisive issue → check → decision, with the decided decision wording
  and the same 150-word line and ending;
- `STRUCTURED_SELF_BLOCK`: the no-peer control, step 3 verbatim;
- `OPENING_PREFILL = "Decisive issue:"`.

**Conditions** (`src/mad/multi.py`, `scripts/run_experiment.py`):

- `debate_structured`;
- `debate_structured_open`: the prefill belongs to the prompt, and the exposure readout is taken
  before it;
- `self_structured`;
- `debate_s2`: a second current-debate sample with its own seed.

**Reusing the archived round 0.** Child jobs take `--subset-ids`. A child must keep every
template its parent recorded; new templates are allowed. Both real parents (`val12b`,
`pilot12b`) pass that check with the current code. Each revision records its generated tokens,
so cost is reported per condition.

**Tests** (`tests/test_structured.py`, stub; a child of a stub parent):

- round 0 is identical to the parent's;
- the second debate sample has a distinct seed;
- structured prompts contain the required steps;
- the prefill is recorded and parsing is correct;
- archive verification passes;
- the template rule allows additions but not changes.

**Session commands, for review** (outputs alongside a copy of the parent archive):

```
# needs on the runtime: val12b-job.json, val12b-questions.jsonl, val12b-round0-c000..c009 (~0.9 GB)
python scripts/run_experiment.py --config configs/base.yaml configs/colab_12b.yaml --out-root OUT \
    --name dev12b --parent val12b --subset-ids artifacts/next_phase/dev_questions.json \
    --conditions debate_s2 debate_structured self_structured [debate_structured_open]
python scripts/run_experiment.py ... --name dev12b-self --parent val12b \
    --subset-ids artifacts/next_phase/dev_self_fill.json --conditions self_reconsider
python scripts/commit_curves.py build --sources --source val:debate_s2:OUT/dev12b-debate_s2 \
    val:debate_structured:OUT/dev12b-debate_structured val:self_structured:OUT/dev12b-self_structured
```

## 6. Tests and checks

| suite | result |
|---|---|
| new: `tests/test_actions.py` | 7 passed |
| new: `tests/test_curves.py` (incl. the real-model reproduction on gemma-3-270m) | 12 passed |
| new: `tests/test_structured.py` | 4 passed |
| full suite (`pytest`) and `ruff` | 273 passed; ruff clean |

## 7. Outstanding issues

1. **Off-device backup:** none configured (§1.3). Your decision is needed on OneDrive or an
   external drive.
2. **Peer-order recency effect (§3)** is large in both readouts and decisions. It is not part of
   any planned contrast. If Stage 3 runs, peer order is already randomized and recorded, so it
   can be analyzed there.
3. **Stage 2 interpretation.** The saved preview already shows the debate–reconsideration gap.
   Stage 2 adds the remaining fractions, the standardized prefix, and correctness curves. Its
   value is the full curve and the bridge, not a new direction.
4. **Throughput.** The letter-only readout speed is unmeasured. Estimates assume 4.6–9
   passes/s.
5. **Stage 3 confounds.** The structured self-reconsideration prompt keeps "agreement alone is
   insufficient" verbatim, so the two structured prompts differ only in peer exposure. With no
   peers the phrase is less natural; review whether to keep it.
6. **Pilot dataset revision not recorded** (provenance only).

## 8. Updated GPU estimates (estimates; to be replaced by Stage 1 measurements)

| stage | what | A100-h | units at 5.30/h | cap |
|---|---|---|---|---|
| 1 | stress/timing: 40-response stratified subset (+ the 14 re-tokenized differences), 20 longest inputs, batch sizes 4/8/16, reproduction checks, 3-question structured smoke | 0.4–0.7 | 2.1–3.7 | 6 |
| 2 | core curves: 12,312 passes | 0.7–1.2 | 3.7–6.4 | 8 |
| 2+ | extras: 9,180 passes | +0.3–0.6 | +1.6–3.2 | +4 |
| 3 | dev run, 90 questions: debate_s2, debate_structured, self_structured, plain self for 45, and curves for the new conditions | 1.5–2.4 | 8–13 | 15 |
| 3+ | forced opening (+ curves) | +0.4–0.6 | +2.1–3.2 | +5 |
| **total** | minimum (1 + 2) / with 3 / everything | **1.1–1.9 / 2.6–4.3 / 3.3–5.5** | **6–10 / 14–23 / 17–29** | **38** |

**Stage 1 pass rules** (from the plan, sharpened by these findings):

- bridge f = 0 vs saved exposure, and exact replay vs saved `before_stated_answer`: argmax
  agreement ≥ 39/40 and median |Δ log p| of the top letter ≤ 0.05;
- batched = single within 0.02;
- peak memory below 38 GB on the 3,143-token longest input.

## 9. Second review (2026-10-29, afternoon): fixes before any GPU session

Earlier sections are unchanged. Still no GPU session.

### 9.1 Peer order matched across prompt conditions

**The problem.** Order was derived from each condition's *kind*, so structured debate would have
shown peers in a different order from ordinary debate.

**The fix.**

- Every debate-family condition (`debate`, `debate_s2`, `debate_structured`,
  `debate_structured_open`) now uses one shared order stream: the ordinary debate's.
- A child job reuses the parent's **archived** debate order explicitly. It is read from the
  parent's hash-recorded debate chunk and verified to name exactly the peers available
  (`PeerOrderMismatch` otherwise).
- Each chunk's manifest records whether order was supplied (archived) or derived.
- Generation seeds remain keyed on the condition name.

**Checked on the archives.** The derived stream reproduces the stored order for all 600
validation and 360 pilot debate revisions, so both routes agree.

**Tests** (`tests/test_structured.py`), parametrized with and without an archived parent debate:

- identical peer sources, identical peer text blocks and identical order across all
  debate-family conditions and the parent;
- order really is randomized;
- distinct generation seeds;
- refusal of an order naming the wrong peers.

### 9.2 Curve readouts bound to their inputs and model; safe resume and stopping

All in `src/mad/curve_runner.py`, tested in `tests/test_curve_runner.py`.

**Binding.**

- **Build identity.** A build directory has a `build_id`, the hash of its four files plus its
  parameters. The files are re-hashed on every load, and an altered build is refused.
- **Output binding.** Each output directory's `run_manifest.json` binds it to the `build_id`,
  the backend identity (backend, repo, model commit, chat template, dtype) and a purpose
  (`run:all`, `smoke`).
- **Refusal.** Any mismatch is refused: a rebuilt input set, a different model, or stub outputs
  resumed into a model run or analyzed as results.

**Row validation.** Every stored row carries its item's sequence hash and option count. On every
load each row is checked against the build: known item, matching hash and option count,
correct length, finite and normalized.

**Stopping.**

- There is one deadline for the whole command (`--max-minutes`), across all modes and batch
  sizes.
- Completed rows are flushed atomically every 25 batches and on any exit: time limit, keyboard
  interrupt, or out-of-memory.
- `run_status.json` records the state, plus expected, done and missing items per mode with a
  sample of missing item ids.
- Byte-identical inputs are read once and copied, with a flag on the copy. For example, the
  standardized f = 1 input equals the exact replay.

**Coverage.**

- `analyze` refuses when any expected item is missing.
- `--partial-smoke` produces an explicitly labelled `PARTIAL_SMOKE` analysis, in separate files,
  restricted to responses complete at all six fractions. The response population is therefore
  the same at every fraction.

**Tests** (11):

- complete run with deduplication;
- time limit → persisted partial output → resume that matches an uninterrupted run;
- interruption keeps completed batches;
- changed inputs refused;
- changed backend refused, including stub;
- tampered rows refused (hash, option count);
- altered build refused;
- core run refuses a build with other sources;
- final analysis refuses incomplete coverage, while the labelled partial analysis drops exactly
  the incomplete response;
- smoke time limit stops every remaining mode;
- smoke report passes only when reproduction holds.

### 9.3 Bounded smoke procedure

**Separate builds.** Core and extras now have separate builds and output directories, and
`run --expect-sources core` refuses any other build:

| build | responses | unique inputs | tokens | longest input | `build_id` |
|---|---|---|---|---|---|
| `runs/next_commit/build_core` | 1,539 | 12,312 | 10.96 M | 3,143 | `216216d7…` |
| `runs/next_commit/build_extras` | 1,245 | 9,180 | 5.93 M | | `9d95a61e…` |

**`commit_curves.py smoke` reads one fixed set:**

- 40 stratified responses (5 changed + 5 unchanged per pool × condition cell), all their inputs;
- the 20 longest inputs;
- every re-tokenized input that differs from the saved-id construction (6 in the core build),
  with its counterpart.

On the core build that is 392 items, 347 unique inputs and 332k tokens.

**Modes.** Each input is read once singly (batch 1), then the same inputs at batch sizes 4, 8
and 16. Each mode's readings are stored separately.

**Report** (`smoke_report.json`). Each check carries its value, threshold and pass flag:

| check | pass rule |
|---|---|
| completeness | every mode complete |
| rows | finite and normalized |
| bridge f = 0 vs saved exposure | argmax agreement ≥ 0.975 and median \|Δ log p\| of the top letter ≤ 0.05 |
| exact replay vs saved `before_stated_answer` | same as above |
| each batch size vs single | argmax agreement ≥ 0.99, median \|Δ\| ≤ 0.02, max \|Δ\| ≤ 0.10 |
| memory per batch size | no OOM, peak ≤ 38 GB |
| re-tokenized vs saved ids | reported, not gated |
| projected core run | ≤ 78 min (core tokens ÷ measured tokens per second at the fastest batch size that passed agreement and memory; setup excluded) |

**CPU dry run** on the real core build with the stub backend:

- all modes complete;
- the two reproduction checks FAIL, as they must with stub outputs;
- every other check passes.

So the selection, the modes and the report work end to end on the real inputs. The stub numbers
are meaningless and were not kept.

### 9.4 Forced opening: both contexts explicit

- **Transcripts.** New transcripts record `prefill` and `exposure_prompt_end`. The exposure
  context is `prompt[:exposure_prompt_end]`; the generation context is that plus the prefill.
- **Readout inputs.** For prefill responses the build makes `bridge_pre_opening` (the
  pre-opening context + `\nFinal answer: (`, compared with the saved exposure readout). All other
  inputs are labelled `standard_after_opening` / `bridge_after_opening`.
- **Separation.** These are never pooled with ordinary `standard` curves. The analysis reports
  each family separately.
- **Archive audit.** It checks the split (`prompt[:end] + prefill == prompt` and generation
  starting after the prefill). It is unchanged on the existing archives, which have no forced
  openings.
- **Tests:** unit (`tests/test_curves.py`) and end to end from a stub child run
  (`tests/test_structured.py`).
- **Decision.** The forced opening stays in scope as an optional condition; the added
  complexity was modest.

### 9.5 Backups

**Re-verified copies.** The copies in `C:\Users\mmcph\mas_backup\` were re-verified in place with
`verify_archive.py`:

| copy | chunk directories | time |
|---|---|---|
| val12b | 20 | 447 s |
| val12b-self | 5 | 110 s |
| pilot12b | 24 | 563 s |
| frozen scorers | SHA-256 `8e5785fe…`, matches the primary copy | |

**These do not protect against hardware failure.** The primary archive (WSL disk image) and
`mas_backup` (`C:`) are on the **same physical SSD**. A disk failure, theft or loss of the laptop
loses both.

**What an off-device copy needs:**

| option | requirement | size |
|---|---|---|
| **(a)** OneDrive | confirm in Windows that `C:\Users\mmcph\OneDrive` is signed in to an account with ≥ 15 GB free and shows "Up to date"; confirm that storing these data there is acceptable | ≈ 7.7 GB now, + ≈ 3 GB after Stage 3 |
| **(b)** external USB drive | appears in WSL as `/mnt/<letter>`; ≥ 16 GB | |
| **(c)** an institutional store | — | |

**Procedure for (a) or (b):**

- `scripts/verify_archive.py --copy-to <dest>/<job>` for each job (it copies, verifies the copy
  and test-loads it);
- `sha256sum` of the frozen scorers;
- a dated `BACKUP_LOG` line.

Nothing has been copied off the laptop.

### 9.6 Optional peer-order diagnostic (scoped, not implemented or run)

**Aim.** Test the last-shown-peer effect (§3) causally: the same peer arguments shown in opposite
orders, everything else fixed.

**Sample.** About 25 **pilot** disputed questions, keeping the validation questions for the
Stage 3 development set. The contexts are revisions whose two shown peers stated *different*
answers: 87 such revisions exist in the pilot, on about 40 questions. That gives about 60–75
contexts.

**Readout-only version (no generation).**

- **Inputs.** For each context, the exposure input (f = 0) under the archived order and under
  the reversed order, each with `Answer: (` and `\nFinal answer: (`. Plus the agent's existing
  revision teacher-forced under the reversed order at the six fractions; the archived-order
  curve comes from Stage 2.
- **Size:** about 10 passes per context, about 750 passes in all.
- **Cost:** about 2–4 minutes of GPU. About 0.05 A100-h (≈ 0.3 units) if appended to the Stage 2
  session; about 0.4 h (≈ 2 units) alone, dominated by setup.
- **Measure.** For a context with peer answers X (first) and Y (last), the recency preference is
  m = log p(last-shown) − log p(first-shown). It is averaged over both orders, so the answers'
  own content cancels. Report:
  - the paired change in p(previous answer) and p(eventual answer) between orders;
  - argmax flips;
  - each with question-cluster bootstrap intervals.

**Generation version.**

- **Inputs.** New revisions under the reversed order **and** fresh revisions under the archived
  order, 2 samples each. Sampling noise is then measured by the same-order pair.
- **Size and cost:** 25 questions × 3 agents × 2 orders × 2 samples = 300 generations, about
  18 min of generation plus readouts. About 0.35 h (≈ 1.9 units) inside a Stage 3 session, or
  about 0.7 h (≈ 3.7 units) alone.
- **Code needed.** A `debate_reversed` condition: a reversed `peer_orders` override. It is small,
  because the override now exists. Not written.
- **Measures:**
  - **final answers:** the share matching the last-shown peer in each order, and the paired flip
    rate against the same-order resample rate;
  - **correctness:** per-agent accuracy and fractional panel-vote accuracy, reversed minus
    archived, paired by question, with intervals.

**Recommendation.** Run the readout-only version inside the Stage 2 session if you approve the
marginal ~0.3 units. Defer the generation version.

### 9.7 Proposed GPU launch (for review; not run)

**Stage 1 only: the curve smoke.**

**Hard cap:** **1.0 A100-hour = 5.3 units for the whole session**, including setup and model
loading. Stop at a balance of (start − 5.3). The smoke itself carries `--max-minutes 25`.

**Expected cost:**

| step | minutes |
|---|---|
| create the session and setup | 10–15 |
| model download and load (24 GB) | 8–12 |
| smoke | 4–10 |
| download and stop | 5 |
| **total** | **27–42 min ≈ 2.4–3.7 units** |

The smoke estimate is 4 modes × 332k tokens at an assumed 4.6–9k tokens/s, with the batch-1 mode
slower.

**Before the session** (laptop):

```
COMMIT=$(git rev-parse HEAD)            # the reviewed commit, pushed
git bundle create /tmp/repo.bundle review-9e45160-fixes
tar -C runs/next_commit -czf /tmp/build_core.tgz build_core
sha256sum /tmp/build_core.tgz           # -> BUILD_SHA256
# /tmp/curves_env.json: {"COMMIT": "<COMMIT>", "NAME": "smoke", "BUILD_TGZ": "/content/build_core.tgz",
#   "BUILD_SHA256": "<hash>", "CURVE_ARGS": ["--batch-sizes", "4", "8", "16", "--max-minutes", "25"]}
colab usage                              # record the balance; abort if < 20 units
```

**Session** (official Colab CLI, `--auth=adc`, the same route as the validation session):

```
colab new -s curves --gpu A100
colab upload -s curves /tmp/repo.bundle /content/repo.bundle
colab upload -s curves /tmp/build_core.tgz /content/build_core.tgz
colab upload -s curves /tmp/curves_env.json /content/curves_env.json
colab upload -s curves <hf token file> /content/hf_token.tmp        # never printed
colab exec -s curves -f artifacts/next_phase/ops/setup.py          # pinned commit, packages, GPU report
colab exec -s curves -f artifacts/next_phase/ops/curves_launch.py  # verifies BUILD_SHA256, starts the smoke detached
colab exec -s curves -f artifacts/next_phase/ops/curves_status.py  # poll every ~5 min (no curve results printed)
# when state is final: tar -C /content/next_commit -czf /content/smoke_out.tgz smoke smoke.log; sha256sum
colab download -s curves /content/smoke_out.tgz runs/next_commit/smoke_out.tgz
colab stop -s curves
```

**After the session** (laptop):

- check the sha256;
- extract to `runs/next_commit/smoke/`;
- copy to `mas_backup/next_commit/`;
- record a dated `OPS_LOG` entry with usage.

**Stop rules.**

- Stop the runtime at the cap, whatever its state.
- If setup or the model load passes 30 min, stop and report.
- If `smoke_report.json` is FAIL, stop and report the failing checks. Do not start the core
  run.

**Afterwards: the core run, as a separate approval.** Proposed with
`commit_curves.py run --build .../build_core --expect-sources core --batch-size <smoke's best> --max-minutes 75`.

- **Projected cost:** 10.96 M tokens at 4.6–9k tokens/s is 20–40 min, plus 20–30 min of
  overhead = **0.7–1.2 A100-h (3.7–6.4 units)**. **Cap: 8 units.**
- **Alternative to save ≈ 2 units of overhead:** run it in the same session as the smoke, only
  if the smoke reports PASS and its projection is ≤ 78 min. Combined cap: 11 units. Proposed
  only if you prefer to skip the review between the two.

### 9.8 Updated GPU estimates (estimates)

| stage | A100-h | units | cap |
|---|---|---|---|
| 1 curve smoke (as §9.7) | 0.45–0.7 | 2.4–3.7 | 5.3 (session) |
| 2 core curves | 0.7–1.2 | 3.7–6.4 | 8 |
| 2+ extras (separate build) | +0.3–0.6 | +1.6–3.2 | +4 |
| 2+ optional peer-order readout diagnostic | +0.05 | +0.3 | +1 |
| 3 structured dev run (incl. new-response curves) | 1.5–2.4 | 8–13 | 15 |
| 3+ forced opening (+ curves) | +0.4–0.6 | +2.1–3.2 | +5 |
| 3+ optional peer-order generation diagnostic | +0.35 | +1.9 | +3 |
| **total** | minimum (1 + 2) 1.15–1.9 / with Stage 3 2.65–4.3 / everything 3.4–5.9 | 6–10 / 14–23 / 18–31 | 41 |

### 9.9 Tests and remaining limitations

**Tests:**

| suite | result |
|---|---|
| `tests/test_curve_runner.py` (new) | 11 passed |
| `tests/test_structured.py` | 8 passed |
| `tests/test_curves.py` | 13 passed |
| full suite and ruff | 289 passed; ruff clean |

**Remaining limitations:**

- **Throughput and memory are unmeasured** for the letter-only readout. The smoke measures
  them; every GPU time above is an estimate.
- **The Stage 3 runbook is not yet written to the same level** as §9.7. That will be done after
  the curve smoke, with its measured throughput.
- **Colab CLI environment passing.** The ops scripts read `/content/curves_env.json`, so they do
  not depend on how `colab exec` passes environment variables. The upload paths follow
  docs/COLAB.md but have not been exercised for this procedure.
- **No off-device backup** (§9.5).

## 10. Stage 1 GPU curve smoke (run 2026-10-29): FAIL on one gate

**Session.** Approved by you ("run smoke") and run at `d7a251b` on a Colab A100 (log:
`artifacts/next_phase/OPS_LOG.md`).

- **Duration:** 12 minutes in all; the smoke itself took 207 s.
- **Cost:** **1.11 units**. The attached rate was 6.73 units/h, not the 5.30/h assumed in the
  estimates.

**Outputs.** The report, status and selection are in `artifacts/next_phase/smoke/`. Raw rows are
in `runs/next_commit/smoke/` and in `mas_backup/next_commit/smoke_out.tgz` (hash-checked). The
outputs are bound to the local core build (same `build_id`) and to the model identity: commit
`96b6f1ec…`, chat template `7de1c58e…`, bf16.

**Report: overall FAIL.** By the stop rule no core run was started.

| check | result | pass |
|---|---|---|
| completeness | all 4 modes × 392 items | ✓ |
| finite, normalized rows | 1,568 | ✓ |
| bridge f = 0 vs saved exposure | argmax 40/40; median \|Δ log p\| of the top letter 7×10⁻⁵ | ✓ |
| exact replay vs saved `before_stated_answer` | argmax 45/45; median 1×10⁻⁷ | ✓ |
| re-tokenized vs saved ids (not gated) | argmax 6/6; median 9×10⁻⁶ | — |
| memory, peak at batch 4 / 8 / 16 | 26.1 / 27.8 / 31.0 GB, no OOM | ✓ |
| batch 4 / 8 / 16 vs single | argmax 0.992 / 0.992 / 0.997; median \|Δ\| 2×10⁻⁶; **max \|Δ log p\| over any letter 1.93 / 1.49 / 1.41** (gate 0.10) | ✗ |
| projected core time | not computed, because no batch size passed the gate above | ✗ |

**Measured throughput.** Unlike the earlier estimates, these are measurements.

| batch size | passes/s | tokens/s | projected core run (10.96 M tokens) |
|---|---|---|---|
| 1 | 6.30 | 6,022 | 30.3 min |
| 4 | 6.84 | 6,533 | 28.0 min |
| 8 | 6.97 | 6,658 | 27.4 min |
| 16 | 6.91 | 6,599 | 27.7 min |

**Diagnosis of the failed gate** (CPU, from the downloaded rows).

- **Most letter differences are in the tail.** The largest per-letter difference falls on a letter
  with p < 10⁻³ in about 90% of inputs, which the any-letter gate does not distinguish.
- **But some differences are substantive.** Between a batched and the single reading, the
  probability of some letter changes by:
  - > 0.05 on 12–16 of 392 inputs (3–4%);
  - > 0.10 on 4–9 (1–2%);
  - at most 0.40.
- **The argmax changes** on 3 of 392 inputs at batch 4 and 8, and 1 at batch 16. Each is a
  near-tie between the top two letters (0.55/0.43, 0.56/0.44, 0.62/0.37).
- **The batched modes also differ from each other** (13 inputs above 0.05 between batch 4 and 8).
  So this is batch-composition-dependent bf16 numerics with left padding, not a construction
  error.
- **Single reads reproduce the archived readouts near-exactly.** Those were themselves read in
  batches of 4.

**What this means for the replication.** The planned metric is argmax agreement, which moved on
≤ 0.8% of inputs, and only at near-ties. Probability-level summaries such as p(eventual) carry a
small batch-dependent noise term. Reading at **batch size 1** removes the dependence entirely, at
a measured cost of about +3 minutes on the core run.

**Recommendation (for your decision; not acted on):**

- Run the core at `--batch-size 1`. That makes the failed batched-vs-single gate moot, because
  no padding is used.
- Restate the batched-vs-single gate for any future batched use in probability terms, e.g.
  share of inputs with max |Δp| > 0.05 at most 5%, plus argmax agreement at least 0.99. Its
  any-letter log-probability form was over-sensitive to tail letters.
- **Updated core estimate** at the measured rate of 6.73 units/h: setup and model about 3 min,
  run about 30 min, packing and download about 3 min, so about 0.6 h ≈ **4–4.5 units**.
  Proposed cap 6 units, with `--max-minutes 45` on the run.

## 11. The smoke's batched-vs-single discrepancies: diagnosis from the downloaded rows (CPU)

This **corrects §10**. It is based on `scripts/smoke_discrepancy.py`, with outputs in
`artifacts/next_phase/smoke/`:

- `discrepancy_by_sequence.csv.gz`: one row per unique input sequence, giving every mode's full
  probability vector and the metrics below;
- `discrepancy_worst_and_controls.csv`: the 15 worst sequences plus 15 stable controls;
- `discrepancy_summary.json`.

**Reconstructing the batches.** The smoke rows did not record batch composition. It was
reconstructed exactly as the runner formed it, and verified against every logged batch (sizes,
token sums, maximum lengths), so each input's left-padding and batch-mates are exact.

### 11.1 What §10 overstated

- **Some tail differences are irrelevant, but not all.** The largest log-probability difference
  falls on a letter with p < 10⁻³ in about 90% of sequences, and such differences are
  practically unimportant. But:

  | | batch 4 | batch 8 | batch 16 |
  |---|---|---|---|
  | unique sequences (347) with some letter's probability changed by > 0.05 | 4.6% | 3.8% | 3.5% |
  | same, by > 0.10 | 2.6% | 1.4% | 1.2% |
  | largest change | 0.40 | 0.34 | 0.36 |

  Item-level shares are similar (> 0.05: 4.1% / 3.3% / 3.1%). These changes are not negligible.
  They fall on the eventual answer and the correct answer in the worst cases, and the argmax flips
  on 3 / 3 / 1 sequences.
- **"Near-tie" was wrong.** A 0.62 vs 0.37 distribution is not a near-tie. One input with a 0.83
  top margin (0.915 → 0.651) is among the worst.
- **"Harmless bf16 rounding" was not established by §10.**

### 11.2 Diagnosis: the readout's letter logits are bf16-quantized

- **The pattern.** The changes take recurring discrete values across unrelated inputs:
  0.776/0.223 ↔ 0.623/0.378, 0.915/0.085 ↔ 0.651/0.349, 0.559/0.436 ↔ 0.833/0.164. Their log-odds
  are exact multiples of 0.125 (1.25 vs 0.50; 2.375 vs 0.625; 0.25 vs 1.625).
- **The test.** In every smoke row, the gaps between the top letter and the next three lie on a
  0.0625 grid (100%; 97.7% on a 0.125 grid).

  | readout | gaps on the 0.0625 grid |
  |---|---|
  | archived `peer_exposure` readouts (same path: the model's own logits) | 100% |
  | archived `before_stated_answer` readouts (fp32 projection of the final hidden state) | 1.5% (chance) |
  | new curve readout (model logits) | 100% |

- **The mechanism.** The model returns logits in bfloat16. At Gemma's soft-capped logit scale
  (|z| ≈ 16–30) bf16 can only represent steps of 0.0625–0.125. Any tiny difference in the final
  hidden state, such as a different kernel path for a batch, can move a letter's logit by one or
  more whole steps. In a two-letter contest that moves the probability by 0.15–0.4.
- **Not padding.** Several of the worst cases have **zero or one** padding token. Their shift is
  *identical* at batch sizes 4, 8 and 16, so neither the padding amount nor the batch-mates
  explains them. A batch-shape-dependent change in the final hidden state is the leading
  explanation; G1 tests this directly.
- **Every path is affected.** On the 85 inputs with archived references, the single, batch-4,
  batch-8 and batch-16 reads each match the archive with about equal fidelity (worst |Δp|
  0.04–0.08). Single reads are not uniquely correct; all paths carry the quantization.
- **Not run-to-run variability.** Batched modes also disagree with each other (> 0.05 on 2–4%).
  Single-read repeat variability has not been measured; G1 measures it.

### 11.3 What this means and what changes

- **Argmax metrics.** Curve metrics based on the argmax are affected only where two letters'
  quantized logits are close. There were no exact ties among the 392 single reads.
- **On the 40 smoke responses,** the six-point `is_eventual` curve under single and batch-4 reads
  differs at one fraction by one response (0.775 vs 0.800 at f = 0.1). Probability-level summaries
  differ by ≤ 0.007. This is not extrapolated to the full set.
- **Fix implemented:** `HFBackend.letter_logprobs_ids(head="fp32_head")`, now the default. It
  projects the final hidden state onto the option-letter rows in float32 with Gemma's
  soft-capping. This is the path whose archived outputs are not quantized, and the 270m-model test
  shows it equals the logits path when the model itself is float32. The head is part of the
  output binding, so `bf16_logits` and `fp32_head` rows cannot be mixed.
- **Also implemented:**
  - rows record batch index, left padding and batch-mate lengths;
  - the status file is written at every flush;
  - an optional explicit-position path for the diagnostic.
- **What remains unresolved without GPU:** whether batch shape changes the fp32-head readout
  materially once quantization is removed, and how large single-read repeat variability is.
  G1 answers both. The original smoke stays **FAIL**; no threshold was loosened.

## 13. Proposed GPU steps (for review; none launched)

All units are at the **observed 6.73 units/h** and must be re-checked per session. Measured and
estimated quantities are labelled.

### 13.1 G1: targeted readout diagnostic (`scripts/readout_diagnostic.py`)

**Plan.** Fixed on CPU in `artifacts/next_phase/diagnostic/plan.json`:

- 12 worst sequences + 8 stable controls, with exact token ids from the core build;
- each target's original batch-of-4 composition;
- partners giving pad 0 and pads of about 50, 300 and 1,500 tokens.

**Conditions,** each read under both heads (`bf16_logits`, `fp32_head`); each changes one thing:

| | condition | isolates |
|---|---|---|
| D1 | single read ×3 | run-to-run repeatability |
| D2 | [target ×4], no padding | batch shape alone |
| D3 | the original composition | reproduction of the smoke |
| D4 | partner co-batches | padding amount and batch-mates |
| D5 | D4 with explicit position ids | position handling |
| D6 | D1 + D4 under `attn_implementation="eager"` | attention implementation, if switchable at runtime |

**Audit.** Eval mode, attention implementation, sliding window, soft-cap, logits dtype,
pad/mask conventions, letter ids and their decoding, and the bf16-vs-fp32 head gap. The
option-letter tokenization after both answer prefixes is already verified on CPU
(`tests/test_curves.py`). A full float32 12B model (~48 GB) is not attempted.

**Estimated cost:**

| item | estimate |
|---|---|
| work | ≈ 1,000 planned passes (≈ 2 M tokens, ≈ 6 min at the measured 6,000 tok/s; the fp32 head's hidden-state output may add ≤ 50%) + eager subset |
| setup, model and packing | ≈ 10 min |
| **total** | **0.3–0.45 h ≈ 2–3 units. Hard cap 4 units; `--max-minutes 20`** |

**Commands** (same CLI route and checks as §9.7; `/content/session_env.json` carries the
parameters):

```
# session_env.json: {"NAME": "diag", "CMD": ["scripts/readout_diagnostic.py", "run", "--build",
#   "/content/next_commit/build_core", "--out", "/content/diag", "--max-minutes", "20"],
#   "REQUIRE": {"/content/build_core.tgz": "4fb26825646e978ced65d236ea81a58bc8d525a5a6b3ef734b027a17f871d951"},
#   "EXTRACT": {"/content/build_core.tgz": "/content/next_commit"}}
colab new -s diag --gpu A100;  upload repo.bundle, build_core.tgz, session_env.json, curves_env.json, hf token
colab exec -f artifacts/next_phase/ops/setup.py          # pinned commit (reads COMMIT from curves_env.json)
colab exec -f artifacts/next_phase/ops/launch_cmd.py     # verifies REQUIRE, unpacks the build, starts detached
colab exec -f artifacts/next_phase/ops/job_status.py     # poll
# pack /content/diag, download, sha256, copy to mas_backup/next_commit/, colab stop
python scripts/readout_diagnostic.py report --out runs/next_commit/diag    # CPU
```

**Decision rule** (fixed now):

- **Quantization confirmed** if, under `fp32_head`, D2–D4 vs D1 shows at most 1% of targets
  with max |Δp| > 0.05, while `bf16_logits` reproduces the smoke's jumps.
- **Unresolved** if `fp32_head` still shows such jumps that track padding (D4) or positions (D5),
  or that disappear under eager attention (D6). G3 would then stay at batch size 1 with
  `fp32_head`, and the issue would be reported.

### 13.2 G2: development timing pilot (new protocol)

**Questions.** 12 validation-pool questions (positions 121–132:
`artifacts/next_phase/pilot_questions.json`). They are development questions, never part of the
fresh collection; 7 overlap the Stage 3 development set.

**Parent job.**

```
python scripts/run_experiment.py --config configs/base.yaml configs/colab_12b.yaml configs/main_250.yaml \
  --out-root /content/pilot --name pilot250 --dataset mmlu-pro --pool validation --pilot-size 120 \
  --validation-size 200 --n-questions 12 --chunk 4 --set data.dataset_revision=b189ec765aa7ed75c8acfea42df31fdae71f97be \
  --conditions pool debate3 structured3 debate5 structured5 --max-hours 1.2
```

It covers the 10-response pool, both panels and both prompts. A memory stress pass runs at the
worst 5-agent readout length (≈ 8.3 k tokens) before chunk 1.

**Second rounds and control** on 4 of the 12 (`pilot_r2_subset.json`), as a child job that
reuses round 1:

```
python scripts/run_experiment.py --config configs/base.yaml configs/colab_12b.yaml configs/main_250.yaml \
  --out-root /content/pilot --name pilot250_r2 --parent pilot250 \
  --subset-ids artifacts/next_phase/pilot_r2_subset.json \
  --conditions debate3_r2 structured3_r2 debate5_r2 structured5_r2 self_structured5 --max-hours 0.4
```

Its stress pass runs at the round-2 worst case (≈ 14 k tokens).

**Measured outputs:**

- per-condition generation time;
- input and output tokens;
- capped responses and parse rates (chunk completion `timings`, rows);
- peak memory at the configured batches;
- activation bytes;
- laptop backup time (`colab_backup.py watch`) and the off-device copy size.

**Estimate** (scaling the measured validation per-response times by 1.4–2.0 for the longer
protocol):

| item | minutes |
|---|---|
| pool | 17–24 |
| debate | 18–27 |
| second rounds | 7–11 |
| control | 2–3 |
| two stress passes | ≈ 10 |
| setup and dataset | ≈ 7 |
| backup | 5–10 |
| **total** | **1.1–1.6 h ≈ 7.5–11 units. Hard cap 12 units** |

`--max-hours` is set per job.

### 13.3 Main collection and extensions (estimates; G2 replaces them with measurements)

**Measured base** (validation run, A100, 150 words / 768 cap, generation batch 8, readout batch 4):

| step | measured |
|---|---|
| first answers (with readout) | 6.05 s per response |
| independent re-answers | 5.66 s |
| debate revisions | 3.51 s, + 0.46 s for their two readouts with activations |
| response lengths | median 161 tokens (first answers), 142 (revisions); 4.8% of first answers hit the cap |
| peak memory reserved | 41.6 of 42.4 GB |

**Assumptions for 250 words / 1,280 cap:**

- generation ×1.4–2.0 per response;
- debate readouts ×1.5–3.0 (longer 5-agent contexts, readout batch 2);
- second-round generation ×1.8–2.6.

| item | content | inference (generation) | extraction (readouts + activations) | storage (all 49 layers / R1 band only) | units at 6.73/h |
|---|---|---|---|---|---|
| **Base G4** | 200 q × 26 responses | 8.3–11.9 h | 1.0–1.6 h | 11.5 GB / 2.1 GB | 63–91 (+ sessions 5–7) = **68–98** |
| E1 second round | 50 q × 16 | 1.4–2.0 h | 0.2–0.4 h | 2.2 GB / 0.4 GB | **11–16** |
| E2 structured reconsideration control | 50 q × 5 | 0.34–0.49 h | 0.05–0.1 h | 0.7 GB / 0.1 GB | **2.7–4** |
| E3 curves, full | 3,200 revisions × 8 passes, ≈ 61 M tokens | — | 2.8–4.2 h (diagnostic readouts, bs 1, no activations) | < 0.1 GB | **19–28** |
| E3 curves, subset | 50 q | — | 0.7–1.05 h | < 0.1 GB | **4.7–7** |
| session overhead | per session: setup + model ≈ 5 min, dataset ≈ 2 min, final verification/backup 10–20 min | | | | ≈ 1.5–3 per session |

**Feasibility.** Balance after this session ≈ 254 units. Base + E1 + E2 + E3-subset ≈ 86–125
units, which fits. E3-full adds 19–28.

**Recommendation.** Decide E3 full vs subset after G2 measures readout throughput on the new
contexts.

**Storage and backup.** Raw archives go to the laptop by the acknowledged backup tool, with a
verified second copy. **An off-device copy is still not configured** (§9.5); for an ~15 GB
collection it should exist before G4.

**Activation retention.** Keeping all 49 layers is the agreed default. Keeping only R1's band
(layers 15–39, step 3) would cut storage by about 82%. Decision for review.

## 12. Core prediction curves (GPU run 2026-10-29, batch size 1, `bf16_logits` head)

**Run.** Approved ("go ahead") before the evening review arrived. It ran at batch size 1, as the
review also asked. Session log: `artifacts/next_phase/OPS_LOG.md`.

- **Measured:** 29.6 min of reading for 13,851 inputs; 3.70 units for the session.
- **Coverage:** complete (13,851 / 13,851), so this is the **FINAL** analysis
  (`artifacts/next_phase/commit_curves/core_bf16_bs1/`).
- **Caveat:** these readouts use the quantized bf16-logits head (§11). They are deterministic at
  batch size 1, but every letter logit is rounded to 0.0625–0.125. Argmax-based curves are
  affected only where two letters' rounded logits are close. G3 would re-read the same inputs
  with `fp32_head` (estimate 4–5.5 units).

**Reproduction against the archives** (same inputs, different prefix-free contexts):

| comparison | argmax agreement | median \|Δ log p\| (top) |
|---|---|---|
| bridge f = 0 vs saved exposure | 99.2% | 2×10⁻⁵ |
| exact replay vs saved `before_stated_answer` | 99.9% | 0 |

At f = 1 the standardized readout's argmax equals the stated answer in 99.87% of responses (a
measurement, not an assumption).

**Curves.** Exploratory: the pilot and validation questions were seen. The standardized prefix
is `Answer: (`. Pooled; question-cluster bootstrap 95% intervals in `curves.csv`.

*Argmax equals the eventual answer:*

| responses | 0% | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|
| debate, changed (n = 251) | **0.77** [0.71, 0.82] | 0.76 | 0.81 | 0.90 | 0.95 | 1.00 |
| reconsideration, changed (n = 308) | **0.15** [0.10, 0.20] | 0.37 | 0.53 | 0.71 | 0.84 | 1.00 |
| debate, unchanged | 0.97 | 0.98 | 0.98 | 0.99 | 0.99 | 1.00 |
| reconsideration, unchanged | 0.95 | 0.87 | 0.89 | 0.88 | 0.93 | 1.00 |

**Paired gap** (debate − reconsideration, changed answers, per question; 71 questions):

| fraction | gap [95% CI] | questions favouring debate / reconsideration | sign test |
|---|---|---|---|
| 0% | +0.63 [0.54, 0.72] | 60 / 2 | p < 0.001 |
| 25% | +0.30 | | |
| 50% | +0.28 | | |
| 75% | +0.21 | | |

**Correctness** (argmax equals the answer key, changed answers):

- Debate is flat at about 0.30–0.35 across fractions.
- Reconsideration falls from 0.43 at 0% to 0.21 at 100%.
- For **harmful** reconsideration changes, the readout before the agent writes anything still
  picks the correct answer 86% of the time. It falls to 0% as the agent writes its way off it.
- For harmful debate changes the 0% readout is already off the correct answer (0.23 correct).

**Comparison with the paper (Figure 1).** The direction is the same as the paper's: Qwen 65 vs
22, Llama 52 vs 35, Gemma-4 48.5 vs 13 at f = 0. Our gap is larger, and kept answers sit at
95–97% at f = 0, close to the paper's 97–99%.

**The prefix matters for reconsideration.** The saved-exposure preview (§3), which used
`\nFinal answer: (`, put reconsideration's f = 0 at 0.41 (changed) and 0.79 (unchanged). Under the
agents' own `Answer: (` format these are 0.15 and 0.95. Debate is much less prefix-sensitive
(0.82 vs 0.77).

**Not shown here:** the extra conditions and the paper's filler and cross-item controls (not run).

## 14. Tests for §11–§13

| suite | result |
|---|---|
| `tests/test_panels.py` (new, 11) | pool and nested panels with identical initial answers; 5-agent panels showing 4 peers, matched across prompts; second-round history policy, order and previous answers; versioned 250-word protocol applied everywhere; per-condition archive verification; legacy identity unchanged; round ordering; worst-case lengths; second round on a subset from the parent's round 1; panel-event categories; nested independent prefixes |
| `tests/test_curve_runner.py` (13) | adds the readout head as bound identity, batch composition in rows, status while running |
| `tests/test_curves.py` (15) | adds fp32 head = logits path on a float32 model, explicit positions, and option letters as single tokens after both prefixes (cached 12B tokenizer) |
| full suite | 304 passed; ruff clean |
