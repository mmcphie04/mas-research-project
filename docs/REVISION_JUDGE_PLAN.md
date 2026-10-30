# Cross-model judge of revisions: committed plan (exploratory)

**Status.** Committed 2026-10-30, **before any judge or own-answer output exists**. Everything
here is exploratory. Context: E1/E2 were dropped. The question is whether helpful and harmful
revisions can be told apart, either by a gate on the agent's own signals or by another model
judging. Section 4 of `artifacts/revision_gate/feasibility.md` is the own-signal gate; this plan
covers the judge. main250's 98 test questions are reserved and appear nowhere here.

## Events

`artifacts/revision_gate/judge/` (`scripts/judge_events.py`):

- `events.jsonl`: 644 events, with no labels in the file;
- `questions.jsonl`: 149 questions;
- `manifest.json`: the rule, seeds, counts and sha256 of both files.

**Population.** Every changed answer in the gate's sources: revisions made after seeing peers,
all Gemma-3-12B datasets, main250 test excluded.

**Rule (outcome-blind).**

- **Questions** are drawn, not events: 149 by `default_rng(20261031)`, a number fixed from the
  population rate of helpful/harmful events per question so as to give about 400.
- Every changed answer of every source on a drawn question is included.
- **Realized:** 195 helpful, 196 harmful, 253 wrong→wrong.

**Deviation from the request.** The request was about 400 helpful/harmful plus about 100
wrong→wrong *events*. Sampling questions makes every panel complete, so the gated panel can be
simulated. As a result wrong→wrong is about 250, not 100; these cost one forward pass per order
for non-thinking judges.

**Thinking subsample.** Questions in a second permutation (seed 20261032), taken until they hold
at least 150 events: 44 questions, 157 events.

## Judges (`scripts/judge_run.py`, `src/mad/judge.py`)

| key | model | mode |
|---|---|---|
| gemma | `google/gemma-3-12b-it` | same-model control |
| qwen3 | `Qwen/Qwen3-8B` | `enable_thinking=False` (template inserts an empty think block) |
| qwen3_think | `Qwen/Qwen3-8B` | `enable_thinking=True`; trace capped at 2,000 tokens, stopped at `</think>`, closed with `\n</think>\n\n` if capped; thinking subsample only |
| llama | `meta-llama/Llama-3.1-8B-Instruct` | the HF token has access (checked 2026-10-30) |

Model commits are recorded at load.

**Judge task** (`JUDGE_SYSTEM`, `JUDGE_USER`, `JUDGE_PREFILL`; hashes in `judge.TEMPLATE_HASHES`
and in every output summary):

- **Input:** the question block exactly as the agent saw it, so option letters match the
  responses. Then "Response 1" and "Response 2": the agent's original and its revision, with
  confidence lines removed and no original/revised labels.
- **Readout:** after the assistant prefill `Better response: (`, the probability on "1" against
  "2". This is the fp32 projection of the final hidden state, one sequence per pass. "1" and "2"
  are single tokens in that context for all three tokenizers (checked on CPU).
- **Both orders:**
  - P(revision correct) = ½ [P("2" | original first) + P("1" | revision first)];
  - position bias = ½ [P("1" | original first) + P("1" | revision first)] − ½.

**Own-answer baseline.** Each model answers each event's question with no responses shown,
under main250's protocol: the same system prompt and persona (tested equal), 250 words and a
1,280-token cap.

- **Samples:** 3 per question for non-thinking models; 1 for `qwen3_think`, on the subsample
  questions.
- **Sampling:**
  - own answers: temperature 0.7 and top_p 0.95, with **top_k 64 and repetition_penalty 1.0 set
    explicitly** for every model (Gemma's effective settings in G4);
  - thinking traces: Qwen3's recommended 0.6 / 0.95 / top_k 20.
- **Readout:** the option-letter readout after the answer (fp32 head, batch 1).
- **Answer parsing:** in thinking mode, from the text after `</think>`.
- No activations are stored.

## Analysis (all with question-grouped 95% intervals, 10,000 resamples, seed 0)

1. **Judge accuracy on helpful/harmful events.**
   - Decision: accept the revision iff P(revision) > ½.
   - Report accuracy and the AUROC of P(revision) for helpful vs harmful, overall and by
     dataset × condition.
   - The thinking judge is reported on the subsample, next to non-thinking Qwen3 on the same
     events.
2. **Bias.**
   - Position bias per judge.
   - **Revision preference:** the mean P(revision) over all events and over wrong→wrong events.
     98.4% of revised responses mention the panel, so a judge can recognize the revision.
3. **Judge against its own answer.**
   - Own-answer gate: accept iff the revised answer occurs more often than the original answer
     among the model's own samples (ties, including both absent: keep the original).
   - Report its accuracy on helpful/harmful events, the judge–own agreement, and judge accuracy
     split by whether the own answers match the revision, the original, or neither.
   - A judge "adds something" only where it beats the own-answer gate, in particular where its
     own answers match neither response.
4. **Simulated gated panels.** All sampled panels are complete. Panel and agent accuracy for:
   - the judge gate and the own-answer gate;
   - accept-all and reject-all;
   - the fixed rules (a) peer majority, (b) conviction ≥ 90 and (c) MAD-Conformist;
   - section 4's cross-validated logistic and boosted-tree gates (their out-of-fold decisions
     for the same events, `artifacts/revision_gate/oof_gate_decisions.csv`);
   - the perfect gate.

   Reported pooled and by dataset × condition.
5. **Cross-model correlation** (from the own-answer runs, on these 149 questions):
   - correlation of correctness between Gemma and each model;
   - same-wrong-answer rate between Gemma and each model, against 1/(n_options − 1);
   - count-matched 3-answer panels (GGG vs GGQ, GGL, GQL), exact expected accuracy averaged
     over the sample choices.

## Limitations (stated with every result)

- **Contamination.** Qwen and Llama may have seen MMLU-Pro; the own-answer accuracy measures it.
- **Events come from Gemma's revisions only.**
- **The revision is identifiable** from its content (panel mentions), so "no labels" does not
  hide which response is the revision.
- **The 149 questions are not a random sample of MMLU-Pro.** They are questions where some agent
  changed its answer, so they skew hard and disputed.
- **Mixed protocols.** pilot12b/val12b events are under the 150-word protocol; own answers are
  all at 250 words.
- **The thinking subsample is small** (44 questions).

## Smoke test (2026-10-30, 17:06–17:27; 1.94 units) and price

**Smoke.** One A100 session, code `20a4391`. Each judge ran in its own process on the first 10
events (both orders) and own answers on the first 10 questions (3 samples; 1 when thinking).
Summaries are in `artifacts/revision_gate/judge/smoke/`.

- **All four ran without errors.** Model commits: Gemma `96b6f1ec` (as G4), Qwen3-8B `b968826d`,
  Llama-3.1-8B `0e9e39f2`.
- **Judge readouts** sum to 1, with "1"/"2" as single tokens.
- **Position bias** on these 10 events: Gemma −0.17, Qwen3 −0.19, Qwen3-thinking −0.10,
  Llama −0.03. Negative means a lean toward "Response 2"; the order averaging removes it from
  P(revision).
- **Readout concentration:** Gemma's and Qwen3's readouts are near one-hot (45–55% above
  0.99); Llama's are not (median top probability 0.88).
- **Own answers:** parse rate 100% for Gemma and both Qwen3 modes, 93% for Llama (2 of 30
  unparsed, 1 capped at 1,280).
- No accuracy was computed on the smoke test.

| measured (smoke) | Gemma-3-12B | Qwen3-8B | Qwen3-8B thinking | Llama-3.1-8B |
|---|---|---|---|---|
| load | 97 s | 62 s | 15 s (cached) | 61 s |
| judge pass (≈ 700-token prompt, batch 1) | 0.195 s | 0.10 s | 14.8 s with the trace (batch 16; mean trace 1,427 tokens, 8 of 20 hit the 2,000 cap) | 0.085 s |
| own answer, 250 words | 7.8 s (batch 8; mean 357 tokens) | 2.3 s (batch 16; 256 tokens) | 12.8 s (943 tokens incl. trace; 1 of 10 trace capped) | 3.0 s (batch 16; 287 tokens) |
| peak memory reserved | 28.3 GB | 19.0 GB | 26.0 GB | 20.5 GB |

**Gemma own answers: reuse where they exist (decided after the smoke, before any full-run
output).**

- 42 of the 149 questions are from main250 or pilot250, whose pools already hold 250-word Gemma
  answers under the same system prompt and effective sampling. For these, Gemma's own samples
  are **pool agents 5, 6 and 7**: never on a debate panel, so independent of every event's
  original answer.
- The other 107 questions (pilot12b/val12b, 150-word data) get 3 fresh 250-word Gemma samples.

**Projected full run (HF backend, one session):**

| item | count | minutes | units at 6.73/h |
|---|---|---|---|
| Gemma: judge + 321 fresh own answers + load | 1,288 passes; 321 | 4 + 42 + 2 = 48 | 5.4 |
| Qwen3: judge + 447 own answers + load | 1,288; 447 | 2 + 17 + 1 = 20 | 2.3 |
| Qwen3 thinking: judge (157 events × 2) + 44 own answers | 314 traces; 44 | 48 (20 batches of 16 at ~148 s) + 9 = 57 | 6.4 |
| Llama (optional): judge + 447 own answers + load | 1,288; 447 | 2 + 22 + 1 = 25 | 2.8 |
| session setup, final pack/download/verification | | 5 | 0.6 |
| **total, all four** | | **≈ 155 min** | **≈ 17.4** |
| total without Llama | | ≈ 130 min | ≈ 14.6 |

**Above the expected 15 units with Llama included; flagged.** Options, none changing the
protocol:

- drop the optional Llama judge (−2.8);
- run the thinking traces at batch 32 instead of 16. Memory is estimated at about 33 GB (KV
  147 KB/token); unmeasured, so it would need its own stress check. That saves about 20 minutes
  (−2.2).
- 2 own samples instead of 3 (−1.6 across the three models).

**vLLM.**

- It would cut the thinking cost. HF here decodes about 216 tokens/s at batch 16, while vLLM
  with continuous batching typically reaches 1.5–3 k tokens/s for an 8 B model on an A100. That
  would take the ~57 thinking minutes to roughly 7–12 (−5 to −6 units).
- **Not measured.** It needs a new generation path (none exists in the repo) and an install that
  pins its own torch, so a separate environment and smoke test.
- The forced "1"/"2" readout should stay on HF's fp32 head, reading the vLLM traces (≈ 1 min).
- At this size it is not worth the new code; it would be if the thinking subsample grew several
  times.

**Proposed cap.**

- **18 units** for the plan as committed (all four judges), with the stop line at the balance
  at launch − 18.
- Or **15 units** without Llama, or with thinking at batch 32 after a memory check.

Supervision as for G4: a 6 GB memory cap on the laptop and the laptop kept awake. Outputs are
small, so they are fetched at the end, not per chunk.
