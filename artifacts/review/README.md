# Review exports

Small, shareable outputs for reviewing branch `review-9e45160-fixes`. No activation arrays, no
model weights. Everything here was produced on CPU. **Nothing was run on a GPU.**

## `reanalysis_v2/` — the corrected 1B reanalysis (current)

`scripts/reanalysis.py --run runs/night2-debate --seeds 0 1 2` at commit `6b6fa42`, on the
existing gemma-3-1b-it debate run (ARC-Challenge, 350 questions × 3 agents, 64-token responses).
Exploratory grouped 5-fold CV over questions; each seed is one fold assignment of the same
questions (stability, not replication). Threshold on the same calibrated scores used at
evaluation, panel-accuracy objective under the tie-break actually applied, accept-all and
reject-all among the candidates; calibration cross-fitted inside training questions.

- `results.json` — per seed: AUROC (all / revisions / within question) with question-resampled
  intervals, calibration, gates under the panel and agent objectives (with beneficial lost and
  harmful blocked), reference policies, paired panel differences, AUROC differences; output
  feature names.
- `predictions.csv` — per seed × family × agent answer: raw and calibrated P(old correct) and
  P(new correct), thresholds, decisions, answer identities, fold.
- `panel.csv` — per seed × gate × question: panel answer and correctness, for every gate and
  reference policy (accept-all, reject-all, agent-optimal, panel-optimal).
- `choices.json` — per seed × fold × family: chosen layer / C / tree settings, Platt calibrator
  coefficients, thresholds.
- `stdout.txt` — the run log.

## `reanalysis_v1_b56c836/` — superseded

The previous export. Its threshold was chosen on raw scores but applied to calibrated ones, it
maximised agent rather than panel accuracy, and its output baseline did not include every
feature now used. Kept so the change is visible; see docs/REANALYSIS.md §5.

## `stub_demo/` — the driver end to end on the tier-0 stub (commit `925f513`)

40 synthetic questions, chunks of 10, four conditions, all four activation positions, and a child
job (`independent`, 20 questions) reusing the parent's round 0. The stub's answers are planted,
so accuracies mean nothing; the files show the record format.

- `demo-job.json`, `child-indep-job.json` — job identities, ordered question hashes, chunks,
  history (the child records its parent's identity hash).
- `demo-status.json`, `demo-ledger.jsonl` — state, per-chunk stage timings, the budget ledger.
- `demo-verify.json` — full archive verification, including a restored copy that loads identically.
- `demo-summary.json`, `summary_stdout.txt` — the summary.
- `debate-view.json`, `debate-c000-manifest.json`, `debate-c000-completion.json`,
  `debate-c000-activation-index.csv` — a view, a chunk manifest and completion record (with
  referenced round-0 checksums), and an index showing round-0 states referenced, not copied.
- `misleading_one-rows.csv`, `misleading_one-c000-transcripts-sample.jsonl` — row and transcript
  formats: parse status, canonical / readout answers, positions and their option probabilities,
  shown vs panel peer answers, misleading target, request ids, structured messages.

## `stub_pilot_report/` — `scripts/pilot_report.py` on a 120-question four-condition stub pilot

Demonstrates the report format only (stub timings are ~0, so the runtime and disk projections
here are meaningless). `report.md`, `report.json`, `transcripts.md`.

## `stub_candidates/` — `scripts/analyze_candidates.py` on a 40-question stub candidate run

## `tiny_cpu_smoke/` — the real Hugging Face path, gemma-3-270m on CPU (commit `925f513`)

`run_experiment.py --smoke` on 2 ARC-Easy questions, four conditions, four activation positions,
64-token cap, stress at batches 2 and 4 over the token lengths of a 120-question pool; plus a
candidate run on the same 2 questions. The 270M model barely follows the answer format (the
strict parser finds no valid answers) and simply adopts proposed candidates; that is expected.
It shows: exact replay on every generated row, missing-position reasons, batched-vs-single
agreement, the stress report, and the record formats.

- `smoke_stdout.txt`, `smoke-tiny-smoke-report.json`, `smoke-tiny-stress.json`,
  `smoke-tiny-job.json` (model and tokenizer identity), `smoke-tiny-status.json`,
  `smoke-tiny-ledger.jsonl`, `debate-rows-270m.csv`, `candidates-270m.csv`.
