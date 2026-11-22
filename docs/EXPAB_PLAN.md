# Experiments A and B: committed plan

**Status.** Committed 2026-11-20 on branch `review-9e45160-fixes`, **before any generation**.

**Stop points:**

1. Stop 1 (this commit): code, CPU tests and these plans. Nothing launches until reviewed.
2. Stop 2: after the GPU smoke test (cap 4 units), with measured throughput and a re-priced
   plan.

**Every GPU run:**

- a hard cap and a stop line;
- the laptop kept awake (Windows power settings);
- the supervisor memory-capped (`systemd-run --user --scope -p MemoryMax=6G`);
- per-chunk backup to the laptop and Google Drive;
- the runtime stopped automatically on completion, at the cap, or on failure.

The balance is about 247 units. The proposed caps total at most about 200, leaving about 45
in reserve.

## 0. Rules built in (from the literature review)

| pitfall | what the code does |
|---|---|
| silent thinking caps, context overflow | Two phases in thinking mode: the trace (≤ cap, stopped at `</think>`), then the visible answer. A capped trace, or one ending at EOS, is closed explicitly (`\n</think>\n\n`) and flagged (`thinking_capped` / `thinking_eos`). A request whose prompt plus caps exceeds `max_model_len` is **refused and recorded** (`skipped = "context"`), never truncated. `max_model_len` = max(needed, 32 k), capped at the model's limit. |
| answer extraction | The **last** stated `Answer: (X)` of the visible answer (`mad.parsing`). The thinking trace is never parsed for the answer. Tested with a response that quotes a peer's answer first. |
| cost accounting | Every request records prompt, thinking, answer and generated tokens, plus the generate call(s) it belongs to. Every call records wall-clock seconds and token totals. Methods are compared at matched GPU-seconds and tokens, never at matched call counts. |
| chooser format (GenSelect) | One representative rationale per distinct answer letter, drawn by seed. Vote counts hidden. Candidates in a seeded random order. The answer is an index (`Best candidate: N`, the last one parsed). 3 runs with permuted order, then a vote over the picks. The picked position is logged. |
| readouts | Batch size 1, fp32 head (G1: batched readouts drift). |

## 1. Generation (vLLM) — `src/mad/gen/`, `scripts/gen_run.py`, `scripts/gen_session.py`

**Environments.**

- vLLM lives in its own virtual environment on the runtime (`artifacts/expab/ops/setup_vllm.py`):
  `vllm==0.33.0`, falling back to `0.32.0`. It pins torch 2.14 and transformers 5.16–5.20, while
  the system HF stack uses torch 2.11 and transformers 5.17.
- The HF backend stays in the system environment for readouts and activations
  (`scripts/hf_readouts.py`).
- `gen_session.py` runs vLLM and HF steps one after the other, never on the GPU at the same time.
  The first failing step ends the session.

**Models and modes (`engine.MODES`):**

| mode | model | template | sampling | caps |
|---|---|---|---|---|
| gemma | `google/gemma-3-12b-it` | — | T 0.7, top_p 0.95, top_k 64, rep 1.0 | 1,280 tokens |
| qwen_nt | `Qwen/Qwen3-8B` | `enable_thinking=False` | the same | 1,280 tokens |
| qwen_t | `Qwen/Qwen3-8B` | `enable_thinking=True` | T 0.6, top_p 0.95, top_k 20, rep 1.0 | thinking 4,096, then an answer of ≤ 400 tokens |

All use the 250-word protocol: the main collection's system prompt, persona and brevity line, and
the same option permutation (seed 12345).

**Per request, recorded:**

- the seed (sha256 of question, agent and condition);
- prompt and generated token ids;
- the stop reason of each phase and whether thinking was capped or ended at EOS;
- prompt, thinking, answer and generated token counts;
- the answer and whether it parsed, the confidence, the visible text and the thinking text;
- the call ids.

**Per call:** wall-clock seconds and token totals.

**Outputs.** Chunks are hash-recorded (`mad.gen.archive`) and resumable. A complete chunk holding
different questions is refused, not skipped. Part names carry the split and tag, so no two
selections share a chunk name.

**Tests** (CPU, stub engine): seeding; cap, EOS and natural closes; context refusal; last-answer
parsing; token accounting; hash checks; resume; the session driver; supervisor integration.

## 2. Question set (committed before any generation: `1dc5600`)

`artifacts/expab/plan/manifest.json` and `questions.jsonl` (`scripts/expab_questions.py`):

- **1,500** MMLU-Pro test questions at revision `b189ec76…`.
- Master order with order seed 7; its sha256 equals main250's (`c4c67294…`).
- **Positions above 520 only** (used: 522–11,996); no overlap with any earlier set.
- `mad.data.stratified_sample`, as equal as possible per subject (107–108 each), seed 20261120.
- **600 development / 900 held-out**, stratified by subject (largest remainder), seed 20261121.
- Hashes: question ids `0a466bb8…`, assignment `25f7d4c8…`, snapshot `5385b3c3…`, plus
  per-question content hashes.
- Questions only; no outcome was read.
- **Experiment A's configuration choices may use development only.**

## 3. Experiment A: is choosing better than voting when the chooser is no stronger than the generators?

**Sample bank (every question):**

- Gemma ×10 and Qwen3 non-thinking ×10;
- Qwen3 thinking ×5, with the answer and thinking text kept. Peers and choosers later see only the
  post-thinking answer text (see decision D1).

**Choosing arms** (`tasks.ARMS`), only on questions whose pool has at least 2 distinct parsed
answers. Elsewhere the arm *is* the pool's plurality vote.

| arm | chooser | pool |
|---|---|---|
| A1_qt5 | Qwen3 thinking | its own 5 thinking samples |
| A1_qnt5 / A1_qnt10 | Qwen3 non-thinking (choosing without thinking) | its own first 5 / all 10 |
| A1_g5 / A1_g10 | Gemma | its own first 5 / all 10 |
| A2_5 / A2_10 | Qwen3 thinking | Qwen3 non-thinking, first 5 / all 10 (same weights generate and choose) |
| A3 | Qwen3 thinking | 5 Gemma + 5 Qwen3 non-thinking (cross-family) |

**Baselines.**

- The plurality vote over each pool.
- Each chooser model voting with itself (self-consistency) at every sample count the bank allows:
  the exact average over all subsets of size n, with measured seconds.

**Pre-registered analysis** (`scripts/expa_analysis.py`). Held-out 900, run once; paired question
bootstrap, 10,000 resamples, seed 0. Scores are exact expected accuracy: a plurality with random
tie-breaking scores 1/k when the gold is among k tied leaders; unparsed answers do not vote.

- **Arm decision:**
  - not called (pool unanimous): the pool plurality;
  - called: the vote over its 3 runs' picks, with unparsed picks dropped;
  - no pick parsed: the pool plurality, flagged and counted.
- **P1:** each arm minus its own pool's plurality vote (all held-out questions; also on called
  questions).
- **P2:** each arm minus the chooser model's self-consistency at **matched GPU-seconds**,
  interpolated linearly in cost between integer n, per question. Matched tokens are reported as
  well. Beyond the curve's ends, the end value is used and flagged.
  - **Cost rule:** a part's GPU-seconds per request = its total measured call seconds / its
    requests. The arm's cost = its pool's samples at their bank cost + 3 chooser calls if called.
- **P3, the verification–generation gap:** on questions whose candidates include both the correct
  answer and a wrong one, the chooser's accuracy minus the chooser model's expected single-answer
  accuracy from its bank samples, on the same questions.
- **Decomposition** (When Debate Helps), per arm against its pool vote:
  - headroom H (correct present, vote wrong);
  - recovery R;
  - new-correct I (necessarily 0 for a chooser);
  - damage L;
  - R + I − L = the mean difference.
- **Also reported:**
  - when no candidate is correct, how often the final pick equals the pool's plurality answer;
  - position bias: the share picking position 1 against the expected 1/m, and the share picking
    the last position.
- **Headline rule (fixed now):** the arm with the largest **development** P1 gain, evaluated once on
  held-out. All arms are reported on held-out regardless. No multiplicity correction; there are 8
  arms × 3 primary measures, and the number is stated with the results.

**Exploratory** (`scripts/expa_neff.py`, CPU):

- n_eff = n / (1 + (n − 1) ρ), with within-family ρ (Gemma, Qwen non-thinking) and between-family
  ρ fitted on development only.
- Checked against held-out voting curves for single-family pools (×1–10) and mixed pools
  (k Gemma + k Qwen, k = 1–5). The prediction is the independent-voter model at n_eff,
  interpolated between integers.

## 4. Experiment B: do reasoning agents debate differently? (Qwen3-8B, thinking on vs off, same weights)

**Panels.**

- 3-agent panels from the bank's first 3 samples of Qwen3 non-thinking, and separately of Qwen3
  thinking.
- **Disputed panels only:** all three answers parsed and not all equal. Panels with an unparsed
  answer are excluded and counted.
- One round.

**Arms per mode** (`tasks.b_requests`). Each agent's own previous (post-thinking) answer is an
assistant turn; peers' texts have their confidence lines removed.

| arm | peers shown |
|---|---|
| debate | the other two, in fixed agent order, labelled "Panel member <agent number>" (the standard peer block) |
| alone | none (the standard reconsideration block) |
| dissent | as debate, plus: "State your honest answer even if it contradicts the others; change only if you can identify a specific error." |
| anon | the standard peer block; peers labelled "Response A/B" in a seeded random order (recorded). It differs from debate only in labels and order. With one round, the debate arm's labels carry little information, so this arm mainly tests order randomization. |

**Pilot.**

- On development questions: ~100 disputed panels per mode, arms debate and alone only
  (`--tag pilot`).
- Report change rate and helpful / harmful / wrong→wrong counts (`scripts/expb_analysis.py
  --pilot`).
- **Sizing rule:** panels needed per mode × arm = ⌈150 / (helpful+harmful per pilot panel)⌉, with
  the measured GPU-seconds per panel. The full run gets the most panels the budget allows up to
  that target, on **held-out** questions.
- **Stop and report before the full run.**

**Measures** (full run, `scripts/expb_analysis.py`):

- **Change rate and helpful/harmful ratio** per mode × arm, with paired arm contrasts on the same
  panels (debate − alone, dissent − debate, anon − debate; panel bootstrap).
  - Kinds: unchanged, helpful, harmful, wrong→wrong.
  - An unparsed revision is counted separately, never as a change.
- **Order effect:** changed agents whose two shown peers differ and whose new answer is one of
  them. Report the share adopting the last-shown peer against the first-shown.
- **Commitment:** the forced readout right after reading the peers, before any writing (in thinking
  mode, at the start of the thinking). Report the probability on the agent's own previous answer,
  for changed and unchanged agents.
- **Answer-prediction curves:** the thinking trace cut at 0/10/25/50/75/100% of its tokens, closed,
  then read out. Report the probability of the final answer by kind, plus whether an option letter
  had already been mentioned.

**Activations (fixed now).**

- Residual stream at Qwen3-8B hidden-state indices **14, 22 and 31** (relative depth ~0.4, 0.6
  and 0.85 of 36).
- Two positions, batch size 1:
  - **A**, just after reading the peers: the last token of the natural context, after `<think>\n`
    in thinking mode;
  - **B**, end of thinking before the answer letter: the token immediately before the letter of
    the last stated answer.
- Stored as float16 in the chunk, hashed.

**Probes** (`scripts/expb_probe.py`, exploratory).

- **Positive control first:** ~50 development contexts whose prompt instructs a letter (A–D).
  - The readout's argmax must recover it at least 0.9 of the time.
  - A 5-fold probe at position A must recover it with accuracy at least 0.8 at some layer.
- **Targets:**
  - "will change" (from A);
  - "helpful vs harmful" among changed agents (from A and from B).
- L2 logistic. The layer × position is chosen on **development** by grouped-CV within-question
  AUROC. The probe trained on development is evaluated once on held-out.
- **Primary measure:** within-question AUROC.

## 5. Judge confirmation (approved earlier; analysis unchanged)

**What changes.** The confirmation run already priced in `docs/REVISION_JUDGE_PLAN.md` is ported to
vLLM generation:

- the 303 helpful/harmful events not yet judged with thinking
  (`artifacts/revision_gate/judge/confirm_events.json`, sha256 `3cf31592…`; 152 helpful, 151
  harmful, 71 questions), both orders;
- 3 thinking own answers per question on all 149 questions.

**What stays the same.**

- The templates, the 2,000-token thinking cap and the sampling settings (mode `qwen_t_judge`).
- The forced "1"/"2" readout on HF's fp32 head at batch size 1.
- `scripts/judge_confirm_analysis.py` implements the committed analysis, reusing the committed
  per-judge code: the primary decision bar on the 303 new events, the secondary on all 391.

**Not reproducible:** HF's per-batch seeds. vLLM seeds each request instead (decision D2).

## 6. GPU smoke test (after review; cap 4 units)

The session plan is `artifacts/expab/ops/smoke_session.json`. All parts are tagged `smoke`, so
nothing collides with the real runs.

**Generation:**

- 20 development questions per model and mode, with the bank's sample counts;
- every chooser arm;
- B (debate and alone, 10 disputed panels per mode);
- the judge confirmation (10 events, 10 own-answer questions);
- the visible-rationale variant of the thinking bank (tag `smokevis`, for D1).

**HF readouts:** B readouts with activations and curves, judge and own readouts, and the positive
control (20 contexts).

**Sanity check:** vLLM on 20 judge questions that already have HF-generated answers under the same
protocol (`--task sanity`). Compare parse rate, length and accuracy (paired).

**Report** (`scripts/smoke_report.py`):

- tokens/s overall and per sequence, and peak GPU memory per step (sampled), with vLLM's KV-cache
  lines;
- parse rate, mean lengths, and thinking-cap and answer-cap rates;
- context refusals;
- the sanity comparison;
- a **re-pricing** of the bank, chooser arms, B pilot and full run, judge confirmation and readouts
  (measured seconds per request × the full request counts at 6.73 units/h, plus per-session
  overhead).

Smoke calls are smaller than the full chunks, so per-request seconds from the smoke are, if
anything, high.

**Rough pre-smoke estimate (not a measurement):**

- about 50 M generated tokens in total:
  - bank ~25 M (dominated by the thinking traces);
  - choosers ~15 M;
  - B ~10 M;
  - judge ~1.5 M;
- at 1.5–3 k tokens/s aggregate, that is 5–9 GPU-hours;
- plus the HF readouts, about 2 s per B row with curves (~4 h for ~7 k rows);
- in all, roughly **60–110 units**, which the smoke test replaces.

## 7. Decisions for review (D) and deviations

- **D1 — what thinking-mode peers and choosers see.**
  - In the judge run, Qwen3-thinking's visible answer was **only** "Answer: (X) / Confidence: NN"
    (44 of 44, 12 tokens); all reasoning stays in the thinking block.
  - Under "peers and choosers see only the post-thinking text":
    - A1_qt5 would choose among **bare letters**;
    - in Experiment B, thinking agents would get answer-only peers while non-thinking agents get
      full rationales. That confounds "reasoning agents debate differently" with "agents shown less".
  - The code has `--visible-rationale` (OFF by default, recorded in every part). It adds to the
    thinking-mode system prompt: "When you have finished thinking, write out your reasoning for the
    reader (at most about 250 words), then end with the two lines." The same prompt is reused in B.
  - The smoke test measures both. **Recommendation: turn it on for the bank's thinking samples and
    Experiment B** (and consider raising the 400-token answer cap if the smoke shows it binding).
- **D2 — seeds in the judge confirmation.** vLLM seeds each request (question, order or sample,
  condition); HF's per-batch torch seeds cannot be reproduced. The sampling distribution is
  unchanged.
- **D3 — the anon arm** differs from debate in labels and order only; same block text.
- **D4 — disputed panel:** all three parsed and not unanimous; panels with an unparsed answer are
  excluded and counted.
- **D5 — B pilot on development, full run on held-out.** Probe layers are chosen on development.
- **D6 — the positive control instructs letters A–D** (≈ 12 contexts per letter for 5-fold CV).
- **D7 — the cost attribution rule** in section 3. vLLM batches many requests per call, so
  per-request seconds are part averages, not individual timings.
- **D8 — chooser text.** A non-thinking chooser writes up to 1,280 tokens. A thinking chooser
  writes its trace (≤ 4,096) then ≤ 400 visible tokens, which is enough for `Best candidate: N`.

**Not done at Stop 1:**

- **No GPU work.** `VLLMEngine` is written against vLLM's documented interface but unexercised.
  Whether output tokens include the stop token, and the exact `finish_reason` strings, are handled
  both ways and will be confirmed by the smoke test.
- **HF readouts** are tested with a fake model only. Real HF readouts reuse the existing,
  GPU-validated fp32-head code.

## Amendment after the Stop-1 review (2026-11-20, committed before the smoke test)

1. **Curves are well-formed.**
   - The thinking trace's reasoning body is cut; a leading `<think>` / `<think>\n` generated by the
     model is stripped first.
   - The open marker `<think>\n` is present exactly once: from the rendered prompt if the template
     already opened thinking, otherwise added.
   - So every point, including 0%, is an open-then-closed think block followed by
     `Final answer: (`.
   - Each row records `trace_began_with_think`. The generator records `think_in_prompt` and
     `think_in_generation`, which the smoke test reports for Qwen3-8B.
2. **Commitment.**
   - Thinking rows get a second readout with a closed, empty think block: prompt +
     `<think>\n\n</think>\n\n` + READOUT_SUFFIX (`commit_closed_logprobs`).
   - For non-thinking rows, Qwen3's template already ends with that block, so the two readouts
     coincide.
   - **The closed-think readout is the primary cross-mode commitment measure**
     (`expb_analysis.commitment`, "primary_closed_think"). The open-think readout, "at the start
     of the thinking", is secondary, reported for thinking rows only.
3. **D1 decided: the visible rationale is ON** for the Qwen3-thinking bank and Experiment B
   (`gen_run.py` default; `--no-visible-rationale` turns it off). The **qwen_t answer cap is raised
   to 700** tokens. The smoke test runs the thinking bank with and without it on the same 20
   questions and reports the answer-cap rate of both.
4. **P2 coverage.**
   - The smoke report gives each arm's measured cost per question in units of one qwen_t bank
     sample, and of its own chooser's samples.
   - If any arm exceeds about 4 qwen_t samples, it proposes and prices raising the qwen_t bank
     from 5 to 8 samples. Otherwise that arm's matched-seconds self-consistency point lies beyond
     the 5-sample curve.
5. `test_bank_requests_use_the_main_protocol` is skipped when `runs/main250` is absent.

## Stop 2: GPU smoke test results and re-pricing (2026-11-20)

**Run.**

- Two attempts, **2.95 units** of the 4-unit cap, balance 244.12. The runtime stopped
  automatically both times.
- Attempt 1 ran out of GPU memory in vLLM's start-up profiling of Gemma-3-12B (fixed in `812ec0f`).
- Attempt 2 was trimmed to fit the remaining cap: 10 development questions per model and mode,
  5 B panels per mode, 5 judge events, 10 sanity questions.
- Report: `artifacts/expab/smoke/smoke_report.json`.

| measured | Gemma-3-12B | Qwen3-8B, no thinking | Qwen3-8B, thinking |
|---|---|---|---|
| aggregate tokens/s, bank call | 689 (100 sequences) | 943 (100) | 1,567 (50) |
| per-sequence decode, small calls | ~26–32 | ~40–60 | ~37–50 |
| peak GPU memory (vLLM step) | 33.6 GB (KV cache 8.4 GiB) | 34.7 GB (KV 16.6 GiB) | (same process) |
| parse rate (bank / sanity / B) | 1.00 / 1.00 / — | 1.00 / 1.00 / 1.00 | 1.00 / — / 1.00 |
| mean generated tokens (bank) | 232 | 213 | 797 (thinking 727 + visible 99) |
| thinking-cap rate | — | — | bank 0/50, choosers 0/42, judge traces 0/10, judge own 1/15 |
| answer-cap rate | 0 | 0 | 0 with the visible rationale (mean 99 tokens) and 0 without (11 tokens) |
| context refusals | 0 | 0 | 0 |

**Other measurements.**

- HF readouts: peak 17.1 GB; about 0.70 s per thinking B row (with curves), 0.13 s per
  non-thinking row, 0.15 s per judge readout.
- Qwen3-8B's `<think>` is **always generated, never in the rendered prompt** (all thinking rows).
- **Chooser picks parse 100%** (every arm); the thinking choosers' visible answer is just
  "Best candidate: N".
- **Position bias** on the smoke questions (6–15 runs per arm, too few to read): the share picking
  the first position is near 1/m for most arms. The thinking self-chooser A1_qt5 picked first in 7
  of 9 runs against 4.5 expected.
- **Sanity: vLLM against HF,** same protocol and questions, 10 questions × 3 samples:

  | | parse | mean tokens | accuracy |
  |---|---|---|---|
  | Gemma | 1.00 / 1.00 | 290 / 351 | 0.37 / 0.47, difference −0.10 [−0.33, +0.07] |
  | Qwen3, no thinking | 1.00 / 1.00 | 231 / 257 | 0.47 / 0.47 |

  Accuracy is within noise. vLLM's responses ran 10–17% shorter in both. That is not significant
  at 30 samples but is worth watching in the bank.

**Re-pricing.**

- The smoke calls were small (6–100 requests), so seconds per request there measure latency, not
  throughput. Pricing by seconds per request gives 131 units in all, mostly an artefact for the
  choosers' 6–15-request calls.
- **Token-based pricing** (`smoke_report.price_tokens`):
  - the measured mean generated tokens per request, times the full request counts;
  - divided by the highest aggregate throughput measured per model. Full runs keep up to 128
    sequences busy, so this errs toward overpricing;
  - plus measured readout seconds and 0.3 h of overhead per launch;
  - chooser call rates: the larger of the smoke rate and main250's disagreement rate (pools of 5:
    0.40; 10: 0.47; mixed: 0.55).

**P2 coverage.** Each arm's measured cost per question, in units of one thinking-bank sample (in
GPU time):

| A1_qt5 | A1_g10 | A3 | A2_10 | A1_g5 | A2_5 | A1_qnt10 | A1_qnt5 |
|---|---|---|---|---|---|---|---|
| 7.9 | 7.7 | 7.3 | 4.8 | 4.2 | 4.0 | 2.9 | 1.6 |

- Several arms exceed 4, so **the thinking bank goes to 8 samples** (already in launch 1's price).
- A1_qt5, at 7.9, sits at the edge of an 8-sample curve; 10 samples would add about 2.9 units.
- Arms whose pool is 10 samples also exceed their own chooser's 10-sample curve by construction.
  P2 then uses the curve's end value, flagged as pre-registered.

| launch | content | estimate | proposed cap |
|---|---|---|---|
| **1** | bank (Gemma ×10, Qwen non-thinking ×10, **Qwen thinking ×8**) on all 1,500 + judge confirmation (606 traces, 447 own answers, readouts) | 4.2 h, **28.4 units** | **36** |
| **2** | B pilot: up to 100 disputed development panels per mode, debate + alone, with readouts | 0.55 h, **3.7 units** | **6** |
| 3 | all 8 chooser arms on all 1,500 (3 runs where the pool disagrees) | 3.0 h, **20.3 units** | 27 |
| 4 | B full: 4 arms × 2 modes × 150 panels, with readouts (re-sized after the pilot) | 1.0 h, **6.9 units** | 12 |
| | **total** | **≈ 59 units** | **81** (≤ 200; ~160 in reserve) |

**Issue for Experiment B: panel supply.**

- On the 10 smoke questions, only 1 of 10 first-3 panels per mode was disputed (all three parsed,
  not unanimous).
- At that rate the 600 development questions give about 60 pilot panels per mode, short of 100.
  The 900 held-out questions give about 90 per mode, which may fall short of 150 helpful+harmful
  events per mode × arm.
- **Proposal (no outcome involved; disputedness uses no gold answer):** also form panels from
  disjoint later sample triples, counted after launch 1:
  - Qwen non-thinking: samples 0–2, 3–5, 6–8, i.e. 3 per question;
  - Qwen thinking at 8 samples: 0–2 and 3–5.
- That needs a plan amendment before the pilot. Otherwise the pilot simply uses every disputed
  development panel available.

## Amendment after the Stop-2 review (2026-11-20, committed before launch 1)

1. **B panel supply.**
   - Panels are formed from every **disjoint sample triple**: Qwen3 non-thinking samples 0–2, 3–5
     and 6–8; Qwen3 thinking samples 0–2, 3–5 and 6–8. That is up to 3 panels per question per
     mode.
   - A triple is a disputed panel iff all three parsed and not all equal. Parse status and answers
     only; **no gold answer is used**.
   - Agents keep their bank sample index (distinct seeds); peers are labelled by panel position
     (1–3).
   - **Every B interval and contrast resamples by question,** not by panel. Arm contrasts are
     paired by panel and bootstrapped over questions.
   - Panels per question are reported.
2. **Held-out extension** (`artifacts/expab/plan_ext/`, `scripts/expab_questions_ext.py`).
   - 2,000 more MMLU-Pro questions from unused positions above 520 (524–12,025), excluding the
     original 1,500.
   - The same as-equal-as-possible stratification (142–143 per subject), seed 20261124,
     outcome-blind.
   - **All held-out:** held-out 900 → **2,900**; development stays 600.
   - Hashes: ids `35a6fee1…`, snapshot `d03da88d…`. The manifest records the original manifest's
     hashes it extends.
   - `tasks.load_questions` checks and merges both (`--set original | extension | all`).
   - **Experiment A's analysis and headline rule are unchanged except for the held-out set**
     (now 2,900).
3. **Thinking bank at 10 samples** per question, replacing 8, for all questions.
   - P2 coverage note: at the smoke's measured costs, every arm is within 10 thinking-bank samples
     (the largest, A1_qt5, ≈ 7.9).
   - Arms whose pool is 10 samples exceed their own chooser's 10-sample curve by construction; P2
     then uses the curve's end value, flagged.
4. **B full-run sizing target:** **~400** helpful+harmful events per mode × arm (was 150), drawn from
   all held-out disputed panels (original held-out plus the extension, so after launch 3).

**Launches** (supervision as before; a hard cap and stop line per launch):

| # | content | cap |
|---|---|---|
| 1 | bank on the original 1,500 (Gemma ×10, Qwen non-thinking ×10, Qwen thinking ×10) + judge confirmation | 38 |
| 2 | B pilot on development (debate + alone, both modes, with readouts) | 6 |
| 3 | bank on the 2,000-question extension | 42 |
| 4 | all chooser arms on all 3,500 questions (after review) | 58 |
| 5 | B full, sized from the pilot (after review) | 30 |

Launches 1–3 may run back to back. Launches 2–5 are re-priced from launch 1's measured throughput
(total ≤ ~186 units). **No held-out choosing outcome is looked at before its one committed
analysis.**

## Launch 1 and the pilot (2026-11-21) — and a proposed amendment (pending review)

**Launch 1** (25.05 units of 38).

- Bank on the original 1,500:
  - parse 98.9% (Gemma), 99.9% (Qwen non-thinking), 99.9% (Qwen thinking);
  - thinking-cap 4.1%; answer-cap 1.1% / 0 / 0.2%.
- Gemma's vLLM bank: mean length 321 tokens (main250 HF 311), stated-answer rate 98.9% (99.15%),
  capped 1.1% (0.85%).
  - Accuracy 0.588 [0.567, 0.610], against 0.643 for main250's HF answers, which are **different
    questions**.
  - A like-for-like check (Gemma ×10 on main250's own 200 questions) is added to launch 3.
- **The judge confirmation does not meet its committed bar.** On the 303 new events, accuracy is
  0.746 [0.642, 0.838] and the own-answer rule scores 0.792 (`artifacts/revision_gate/judge/confirmation_results.json`).
- Full-scale throughput (generated tokens/s): Gemma 1,472; Qwen non-thinking 4,060; Qwen thinking
  2,250.

**Launch 2: the B pilot** (4.02 units of 6). On development: 100 disputed panels per mode, 1.1–1.2
panels per question.

| mode × arm | change rate | helpful / harmful | helpful+harmful per panel |
|---|---|---|---|
| non-thinking, debate | 0.55 | 52 / 39 | 0.91 |
| non-thinking, alone | 0.10 | 14 / 3 | 0.17 |
| thinking, debate | 0.48 | 71 / 31 | 1.02 |
| thinking, alone | 0.12 | 20 / 6 | 0.26 |

**The positive control fails at position A.**

- The commitment readout recovers the instructed letter in 100% of the 50 contexts per mode.
- Linear probes on the activations at position A (the last token of the natural context) reach
  only 0.36–0.48 (thinking) and 0.20–0.32 (non-thinking), against chance 0.25. Standardized and
  regularized probes, at C from 0.001 to 1, are no better.
- So the letter is not linearly readable at A from these activations, and the A-position probe
  analysis is not validated by its own positive control.

**Amendment (proposed 2026-11-21; approved at the review of launches 1–2, 2026-11-21):**

- Also store the residual stream at **position R, the readout position** (the last token of
  prompt + `Final answer: (`). It comes from the same forward pass as the commitment readout, so
  it costs nothing extra; activations become [A, B, R].
- Rerun the positive control with **200 development contexts per mode** (tag `rerun200`, added to
  launch 3's session; exploratory, about 3 minutes).
- If R passes and A does not, the B probes would be reported at R (and B), with A shown as failing
  its control.

### Re-pricing of launches 3–5 (after launch 1 and the pilot; estimates, not measurements)

Sources:

- Launch 1's full-scale throughput.
- The pilot's measured seconds per panel: generation plus HF readouts at batch size 1 (the pilot's
  25-panel chunks; larger chunks may batch generation better).
- The smoke test's chooser tokens per request (6–15 requests per arm, so uncertain).
- Units at 6.7 per A100-hour; ~1.5–2 units per launch for setup, staging and model loads.

| launch | basis | estimate | cap |
|---|---|---|---|
| 3 | 22,000 Gemma + 20,000 Qwen non-thinking + 20,000 Qwen thinking requests at launch-1 tokens/request and tokens/s | ~31 | 42 |
| 4 | chooser requests = disagreement rate × 3,500 × 3 orders, at smoke tokens/request (A1_qt5 0.66 h, A2_5 0.85, A2_10 0.69, A3 0.98, the rest 0.6 together) | ~27 (range ~25–35) | 58 |
| 5 | see the options below | 26–43 | 30 |

**Launch 5 (B full).**

- Four arms per mode, contrasts paired by panel. Dissent and anon were not piloted, so they are
  priced as debate.
- Seconds per panel (generation + readouts):
  - non-thinking: debate 1.56, alone 1.06;
  - thinking: debate 6.78, alone 5.93.
- Panels needed for 400 helpful+harmful events, at the pilot's event rates:

  | mode | debate | alone |
  |---|---|---|
  | non-thinking | 440 | 2,353 |
  | thinking | 393 | 1,539 |

Options:

1. **All four arms on the debate-sized panels** (440 non-thinking, 393 thinking): **~26 units**.
   - Alone then gets about 75 (non-thinking) and 100 (thinking) helpful+harmful events, not 400.
2. **Option 1 plus alone on further panels up to its target**: **~43 units**.
   - Above the 30 cap, but within the ~186 total.
   - The debate − alone contrast stays paired on the first 440 / 393 panels; the extra alone panels
     feed only the per-arm alone estimates.
   - It uses nearly all of the projected held-out supply for non-thinking (~2,500 disputed panels
     after launch 3; to be measured).
3. **All four arms on the alone-sized panels**: **~100 units**. Not affordable.

Spent so far: 29.07 units (launches 1–2). With launches 3 and 4 at their estimates, about 87 units
are used, which leaves about 99 of the ~186.

## Review of launches 1–2 (2026-11-21): decisions

1. **Position R approved.**
   - Activations are stored as [A, B, R], plus the 200-context positive-control rerun (in launch 3).
   - Pass rule, unchanged from the plan: readout recovery ≥ 0.9 and best probe accuracy ≥ 0.8, per
     position.
   - If R passes its control and A does not, B's probes are reported at R and B, with A shown as
     failing.
2. **Launch 4** (cap 40) runs every chooser arm on all 3,500 questions (`artifacts/expab/ops/launch4_session.json`).
   - It runs after launch 3 if the launch-3 report shows no surprises.
3. **Launch 5 = option 2** (cap 50; `artifacts/expab/ops/launch5_session.json`).
   - All four arms on the first **440** (non-thinking) / **393** (thinking) held-out disputed panels
     of the seeded order (tag `full`). The paired contrasts use these panels.
   - Then reconsider-alone on the **following** panels of the same order (`gen_run --panel-start`;
     tag `fullx`), up to about 400 helpful+harmful events at the pilot's rate: 1,913 more
     non-thinking and 1,146 more thinking, or fewer if the supply runs out.
   - The two sets are disjoint by construction. Debate − alone stays paired on the shared panels;
     the extra alone panels feed only the per-arm alone estimates.
   - HF readouts follow each half, so the shared panels are complete first.
4. **The like-for-like Gemma check.** This rule is committed before its data were seen.
   - **Data:** Gemma ×10 under the bank protocol on main250's own 200 questions (vLLM), against
     main250's HF pool (10 samples per question).
   - **Statistic:** the per-question accuracy difference (vLLM − HF), with a 95% bootstrap CI that
     resamples questions (`expab_launch_report.main250_like_for_like`).
   - **Rule:** **a real gap = that CI excludes zero.** A real gap stops the work for review before
     launch 4. Otherwise launch 4 then launch 5 run.

## Launch 3 results (2026-11-21)

Launch 3 used **32.85 of its 42 units**, about 1 of them idle while a zombie process kept the runtime
alive (fixed in `43dd746`). Results are in `artifacts/expab/launch3/`.

**Extension bank** (2,000 held-out questions × 10 samples per mode):

| | Gemma | Qwen non-thinking | Qwen thinking |
|---|---|---|---|
| parse rate | 98.8% | 99.9% | 99.9% |
| mean generated tokens | 326 | 244 | 1,150 (thinking 1,033 + visible 117) |
| thinking-cap rate | – | – | 4.7% |
| answer-cap rate | 1.2% | 0.04% | 0.2% |
| context refusals | 0 | 0 | 0 |
| tokens/s | 1,491 | 4,074 | 2,240 |

**Like-for-like Gemma check (rule committed before its data):**

- main250's own 200 questions: vLLM 0.6425, HF 0.643.
- Difference −0.0005, 95% CI [−0.018, 0.018]. **No real gap.**
- So the bank's lower accuracy (0.59 on the original and on the extension) reflects the question
  set, not the vLLM path.

**Positive control, 200 development contexts per mode** (pass rule: readout ≥ 0.9 and best probe
≥ 0.8):

| mode | readout | probe at A (layers 14 / 22 / 31) | probe at R (layers 14 / 22 / 31) | A | R |
|---|---|---|---|---|---|
| thinking | 1.00 | 0.63 / 0.77 / 0.70 | 0.97 / 0.97 / 1.00 | fails | passes |
| non-thinking | 1.00 | 0.48 / 0.73 / 0.38 | 0.99 / 0.96 / 1.00 | fails | passes |

- Chance is 0.25.
- By the approved rule, B's probes are reported at R and B, with A shown as failing.
- **Interpretation limit:** R is the position where the model is about to emit the letter.
  Passing there shows the instructed letter is linearly readable where it is output. It does not
  show an earlier internal commitment.

**Held-out disputed-panel supply** (original + extension):

| mode | panels | launch 5 needs |
|---|---|---|
| non-thinking | 2,602 | 440 shared + 1,913 more alone = 2,353 |
| thinking | 1,852 | 393 + 1,146 = 1,539 |

Development supply: 524 non-thinking and 355 thinking.

## Launch 4 results and the launch 4b amendment (review of launches 3–4, 2026-11-21)

**Launch 4** stopped at its stop line (39.94 units of 40, including 7.26 lost to two failed staging
attempts).

- **A3 complete:** 2,114 questions, 6,342 requests.
- **A2_5 partial:** 5 of 9 chunks.
- Not started: A2_10, A1_qt5, A1_qnt5, A1_qnt10, A1_g5, A1_g10.
- Outcome-blind operational stats (`artifacts/expab/launch4/choose_report.json`):

  | arm | picks parsed | thinking-cap rate | tokens per request | tokens/s |
  |---|---|---|---|---|
  | A3 | 99.9% | 18.6% | 2,096 | 1,083 |
  | A2_5 | 99.8% | 27.7% | 2,365 | 1,256 |

  Answer-cap rates are 0.1–0.2%, and no requests were refused for context.
- **Why it overran:** the smoke-based estimate missed by about 3.5×. Tokens per request were
  1.3–1.7× the smoke's, and throughput was about half the bank's.

**Launch 4b (cap 58) — committed before it runs:**

1. **Order.** The unfinished arms run in this priority order: finish A2_5; then A1_qt5, A1_qnt5,
   A1_g5, A2_10, A1_qnt10, A1_g10 (`artifacts/expab/ops/launch4b_session.json`).
   - Each arm completes before the next starts; the first failing step ends the session.
   - So the cap can cut only the lowest-priority arms.
2. **Protocol.** Every setting is as in launch 4, including the 4,096 thinking cap and chunks of
   150 questions (A2_5 resumes into its own chunks).
3. **If any arm is not run, or is cut part-way:**
   - The **headline rule applies to the completed arms only**: the arm with the largest
     development P1 gain among them, evaluated once on held-out.
   - A partial arm is never analysed.
   - The omission is reported: which arms, and how many chunks each completed.
   - Implemented in `expa_analysis.arm_completion` / `run`: `arms_analysed`, `arms_omitted`.
4. **Report after the launch:**
   - completion per arm;
   - units;
   - outcome-blind operational stats per chooser arm, including the **thinking-cap rate**
     (`scripts/expab_choose_report.py`).
   - No held-out outcome is looked at before the committed analysis.

**Then launch 5** (B full, option 2), cap 50, as committed above.

**Operations:**

- If GPU requests keep failing with Service Unavailable, back off and retry.
- If setup (including staging) fails twice for a launch, stop and report rather than spend units on
  further attempts.
