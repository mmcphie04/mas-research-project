# Changes in response to the review of b56c836

Branch `review-9e45160-fixes`. **Nothing has been run on a GPU.** Three levels of verification
are kept apart below: **stub** (tier-0 backend, planted ground truth), **270M** (the real Hugging
Face path with `google/gemma-3-270m-it` on CPU — same Gemma 3 tokenizer family, not the 12B), and
**unverified on 12B**. The audit document the brief refers to (`GPU_Experiment_Audit_b56c836.md`)
was not available; this follows the brief.

## 1. Correctness

**1.1 Strict answer parsing** (`src/mad/parsing.py`). The old regex read "Answer: cannot
determine." as C, "Answer: Because…" as B, "Answer: Analysis…" as A. Now a line-anchored grammar:
`Answer: (X)` is strict; `Answer: X` alone on its line, or a letter followed by `)`, is permissive;
lowercase letters, letters followed by words, mid-sentence quotes and out-of-range letters are
rejected; the last field in the agent's own response decides, and an out-of-range last field
invalidates the response rather than falling back to an earlier one. Parse status is saved per
row (`stated_status`). The same parse supplies the answer-letter activation position. Tests: 24
cases including the three reported misreads (stub-free unit tests). On the 270M model the strict
parser finds no valid answers — the earlier single "parsed" answer was a misread.

*Missingness policy:* what a peer shows is what it stated. Natural peers show raw text; their
recorded shown answer is the stated one or −1, never the readout fallback. Answer-only peers with
no stated answer show "No final answer was given." Missing answers do not count in the shown-peer
majority. The fallback is used only for the agent's own canonical answer and the panel vote,
where it is flagged.

**1.2 Threshold and calibration** (`src/mad/analysis/gates.py`). τ was chosen on raw out-of-fold
probabilities and applied to calibrated test probabilities. Now one scoring path: raw →
calibrator → calibrated → difference → τ, at selection and at evaluation. Cross-fitted route:
each inner fold is calibrated by a Platt fit on the other folds before τ is chosen; split route:
calibrator and τ from validation. τ maximises **panel accuracy** under the declared tie-break
(agent-accuracy τ kept as a secondary rule), over a grid that includes −∞ (accept all) and +∞
(reject all). Tests: held-out-label flips change no choice and no prediction; with a strongly
non-identity calibrator, τ is optimal on the calibrated scale and evaluation uses calibrated
scores; accept-all and reject-all are reachable.

**1.3 Full-distribution schema** (`src/mad/analysis/features.py`). One adapter reads
`agent_states.readout_logprobs` (new runs) or the old `readout_full.parquet` sidecar, validates
widths against option counts and normalisation, and **refuses** to run a primary analysis without
full distributions. Output features by information time (pre / exposure / post), including full
distributions, old/new candidate probabilities under each context, margins, entropies, verbalized
confidence with missingness, lengths, response log-likelihood, fallback / cap / disagreement
indicators, shown-peer and panel agreement, categorical positions. Gold, correctness and
fabricated-peer construction metadata are never features; a test checks names. Feature names are
written to every results file. Tested on a new-style stub run with no sidecar.

**1.4 Position-aware loader** (`src/mad/analysis/states.py`). Keys are (condition, question, agent,
phase, position); duplicate keys are refused; a missing state is an error by default or masked
under a declared policy, never substituted. Test: synthetic states whose values encode their own
phase and position, across several shard files, retrieved exactly. `FeatureBank` now sits on it;
`scripts/peer_exposure_analysis.py` refuses multi-position runs (legacy-only).

**1.5 Preregistered route** (`scripts/make_split.py`, `scripts/reanalysis.py --mode split`).
Frozen question-level train / validation / test manifest (stratified by subject, records the job
identity hash); models on train, calibration and τ on validation, test scored once. The repeated
grouped CV is kept as `--mode cv` (exploratory). Exports: per-row raw and calibrated
probabilities, τ, decisions and answer identities (`predictions.csv`); per-question panel answer
and correctness under every gate and reference policy (`panel.csv`); every chosen layer, C, tree
setting, calibrator coefficient and τ (`choices.json`). Agent-level intervals resample questions
(no row-wise DeLong). Transfer: gates fitted on one condition applied frozen to another
condition's held-out questions (`--transfer-runs`).

## 2. Collection and archive

**2.1 Memory coverage.** Token lengths are measured over the whole intended pool (not the two
smoke questions) from actually rendered prompts with the real tokenizer, plus the measured
round-1 template overhead and three full responses. The stress pass tries several batch sizes
(default 4, 6, 8) at those lengths, records allocated and reserved peaks and finite values at
every layer, and is saved to `<name>-stress.json`; a job does not repeat a passed stress on the
same GPU type. 270M: coverage over 120 pool questions and the stress pass ran.

**2.2 Readouts.** All layers kept at four positions through the main run: `readout_last`,
`generation_last`, `stated_answer`, and new `before_stated_answer` (the token whose output
predicts the stated letter, restricted to the agent's own response). Option probabilities are
saved at every position, computed from only the letter rows of the output layer with the model's
soft-capping — 270M: identical to the model's own logits. **The readout input is now built from
token ids** (prompt tokens + the generated ids + suffix tokens): re-tokenising text merged the
response's final newline with the suffix's newline into one token, so the model read a different
sequence than it generated. `replay_exact` is saved per row (270M: true for every generated row).
Token indices and ids of each position, and a reason for each missing one, are saved. New:
response log-likelihood (mean / min / sum token log-probability at temperature 1), computed in
position chunks without full-sequence logits — 270M: equal to full-sequence logits.

**2.3 Archive verification** (`mad.job.verify_archive`, `scripts/verify_archive.py`). Checks every
recorded checksum (shards, rows, transcripts, index, activation metadata, referenced round-0
files) even when a view's metadata is unchanged; job membership; unique keys; states per
question; every layer loadable at the recorded shape; all values finite; states per phase and
position. `--copy-to` copies the job elsewhere, verifies the copy and checks it loads identical
arrays. Tests: a single flipped byte that `build_view` misses is caught; a corrupted shared round-0
file is caught; a copied archive verifies and loads identically.

**2.4 Storage, timing, budget.** Round-0 states are stored once per chunk and referenced by each
condition (the index carries file paths and layouts), with their checksums in each condition's
completion record. Stage timings (round 0, generation, readouts, writes) per chunk; an
append-only ledger (`<name>-ledger.jsonl`) of session start/end, model load, stress, chunk
done/failed; GPU, CUDA, driver and package versions in each session's history; `--max-hours`
now counts model loading and stress. Projections: docs/BUDGET.md.

## 3. Scope

- `debate_answer_only` shows each peer's stated letter **and option text** (documented).
- **Child jobs** (`--parent NAME --subset-n M`): run more conditions on the parent's first M
  questions, reading round 0 from the parent's verified caches (never regenerating), with the
  parent's dataset, pool, config, prompts and model enforced. Tests: identical round-0 answers
  and distributions to the parent; refusal without the parent's round 0; refusal on a config
  difference.
- **`independent` condition:** a fresh round-0 answer with a new seed, for comparing debate with
  sampling more answers.
- **Pilot report** (`scripts/pilot_report.py`): quality, headroom above the better baseline,
  class support with distinct questions, measured runtime by stage, disk projection, power
  sensitivity, and a fixed transcript sample plus every malformed answer.
- **Summary** adds correct-answer coverage before/after, answers introduced by round 1, a vote of
  all six answers, headroom above the better baseline, parse-status shares.

## 4. Secondary analyses prepared

In `scripts/reanalysis.py`: pre / exposure / post information times, old and new correctness
separately, overall / revisions / within question; beneficial lost vs harmful blocked per gate;
output logistic, gradient-boosted trees, activation, combined and cross-fitted stacked families;
the pre-time gate (`--time pre`) is the protection gate; transfer between conditions. Not
implemented: the joint beneficial/harmful/unhelpful model (optional in the brief).

## 5. Candidate evaluation (`src/mad/candidates.py`, `scripts/run_candidates.py`, `scripts/analyze_candidates.py`)

Per question, a correct and a random incorrect candidate, each under neutral, endorse-correct and
endorse-incorrect contexts: six readout-only passes, saving distributions and states. Analysis:
within-question ranking and endorsement log-odds shifts (output-only), and a candidate-correctness
probe trained on training questions vs output-only and combined, per context. Stub: tested end to
end. 270M: runs (the 270M model simply adopts the proposed answer, probability ≈ 1).

## 6. 1B reanalysis rerun

See docs/REANALYSIS.md §5 and `artifacts/review/reanalysis_v2/`. The earlier export is kept as
`artifacts/review/reanalysis_v1_b56c836/`.

## 7. Deliverables

| | where |
|---|---|
| commands: smoke, four-condition pilot, main, child jobs, candidates, verification | docs/COLAB.md |
| GPU-hour, disk and power estimates | docs/BUDGET.md |
| every GPU-dependent measurement saved or omitted | docs/GPU_MEASUREMENTS.md |
| preregistration, consistent with the code | docs/preregistration.md |
| pilot report | `scripts/pilot_report.py`; demonstrated on a stub pilot (`artifacts/review/stub_pilot_report/`); the real one needs the GPU pilot |
| corrected 1B reanalysis with per-question panel exports | `artifacts/review/reanalysis_v2/` |
