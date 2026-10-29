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
