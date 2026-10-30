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
