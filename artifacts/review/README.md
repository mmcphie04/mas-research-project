# Review exports

Small, shareable outputs for reviewing branch `review-9e45160-fixes`. No activation arrays or
model weights. Everything here was produced on CPU; **nothing was run on a GPU.**

## `reanalysis/` — the corrected 1B reanalysis

`scripts/reanalysis.py --run runs/night2-debate --seeds 0 1 2` on the existing gemma-3-1b-it
debate run (ARC-Challenge, 350 questions × 3 agents, 64-token generations). Every choice
(layer, regularisation, tree settings, threshold, calibration) is made inside the training
questions; each seed is one grouped 5-fold split of questions, so every question is held out once
per seed. The three seeds reuse the same questions and are stability checks, not replications.

- `results.json` — per seed: chosen settings per fold, AUROC / within-question AUROC / Brier /
  log-loss / ECE per family and target, gate results, reference gates, paired panel differences
  with 95% bootstrap intervals over questions.
- `heldout_predictions.csv` — one row per seed × family × (question, agent): held-out P(old
  correct), P(new correct), the accept decision.
- `../reanalysis_stdout.txt` — the run log.

Summarised, with the correction, in `docs/REANALYSIS.md`.

## `stub_demo/` — the new driver end to end on the tier-0 stub

40 synthetic questions, chunks of 10, five conditions (`self_reconsider`, `debate`,
`debate_answer_only`, `misleading_one`, `adversarial`), at commit `0ac3e1e`. The stub's answers
are planted, so the accuracies mean nothing; the files show the record format.

- `demo-job.json` — the job file: identity, ordered question ids with content hashes, chunks,
  code commit, history.
- `demo-status.json` — state, chunk timings, failures, cumulative hours.
- `demo-summary.json`, `summary_stdout.txt` — the expanded summary.
- `debate-view.json` — the merged view's validation record.
- `debate-c000-manifest.json` — a chunk manifest (seeds, job identity hash, question ids).
- `misleading_one-rows.csv` — every state: canonical and readout answers, answer source, stop
  reason, shown vs panel peer metadata, misleading target, request ids.
- `misleading_one-c000-transcripts-sample.jsonl` — six transcript records: structured messages,
  rendered prompt, text, stop reason, peer order and sources.

## `tiny_cpu_smoke/` — the real Hugging Face path, on CPU

`run_experiment.py --smoke` with `google/gemma-3-270m-it` (18 layers × 640) on 2 ARC-Easy
questions, four conditions, three activation positions, 64-token cap. The 270M model barely
follows the answer format, so most answers fall back to the readout; that is expected and not
the point. It exercises the letter-in-context check, the stress pass, token ids and stop
reasons, extra positions, the smoke value checks, and batched-vs-single readout agreement.

- `smoke_stdout.txt`, `smoke-tiny-smoke-report.json`, `smoke-tiny-job.json` (includes the model
  and tokenizer identity block), `smoke-tiny-status.json`.
