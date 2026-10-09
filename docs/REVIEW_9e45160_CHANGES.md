# Changes in response to the review of commit 9e45160

Branch `review-9e45160-fixes`. Nothing here has been run on a GPU. "Verified" below means
verified on CPU: by the test suite, by the driver on the stub, or by the driver on the real
Hugging Face path with the 270M Gemma model.

## 1. Provenance and resume (P0)

| problem found in 9e45160 | change | verified by |
|---|---|---|
| `--dataset` / `--n-questions` never reached the config, so manifests recorded stale values | driver writes them into the resolved config before anything is recorded | `test_resolved_config_records_the_cli_dataset_and_size` |
| no job-level record; a resume could mix question samples, chunk sizes, settings, models or code | `mad/job.py`: `<name>-job.json` holds identity (dataset, pool, chunk size, conditions, full config minus batch sizes, prompt hashes, model commit / tokenizer / chat template / special tokens / letter ids), ordered question ids with content hashes, chunk membership, code commit, and an append-only history. Any mismatch on resume stops the driver and names what differs | `test_resume_with_different_settings_is_refused` (chunk size, a model setting, conditions, pool), `test_resume_at_a_different_commit_needs_explicit_permission`, `test_a_chunk_from_another_job_is_refused` |
| `stratified_sample` sorted its output by id, so asking for more questions reshuffled the sample under existing chunks | one fixed master order (subjects interleaved, seeded hash within subject); a job is a prefix; extension appends; an extension that would change a chunk with output is refused | `test_extension_appends_to_the_same_question_order`, `test_extension_that_would_change_a_started_chunk_is_refused` |
| no pilot/main separation | first 120 questions of the master order are the pilot pool, main pool is the rest (`--pool`) | `test_pilot_and_main_pools_are_disjoint` |
| round 0 regenerated on retry, so conditions finished before a crash and those finished after did not share round 0 | round 0 saved per chunk (texts, token ids, stop reasons, readouts, hidden states, checksums) before any condition; reloaded and checked on retry | `test_retry_reuses_saved_round0_and_skips_finished_conditions`, `test_saved_round0_for_other_questions_is_refused` |
| a condition was rewritten in place, so a crash could leave an old `completion.json` beside new partial files | each condition is written under a temporary name and renamed when complete; completion records checksums of shards, rows, transcripts and index | same retry test (no directory left behind for the failed condition) |
| merge compared only chunk counts and copied every array | validated **view** per condition: job membership of each chunk, checksums, unique state keys, exact states per question, consistent shapes; the index points at chunk shards (no copies); rebuilt when any chunk's completion record changes | `test_full_run_writes_views_and_status`, `test_view_loads_the_same_activations_as_the_chunks`, `test_a_corrupted_shard_fails_the_view`, `test_a_lost_chunk_is_redone_and_the_view_rebuilt` |
| summary picked runs with `"-c0" not in name`, which breaks at chunk `c100` and picks up stray directories | summary and notebook find conditions through the job file | `test_summary_ignores_directories_outside_the_job` |
| a failing chunk was logged and the next one tried, indefinitely | stop on any out-of-memory error or two consecutive failures; failure type and message in the status file | `test_repeated_failures_stop_the_run`, `test_out_of_memory_stops_immediately` |
| no budget cap | `--max-hours`, cumulative across sessions (persisted in the status file) | `test_time_cap_persists_across_sessions` |
| notebook pulled moving `master` and force-pushed a shared `colab-results` branch | notebook checks out a required 40-character `COMMIT`, verifies it and a clean tree; results go to a new `colab-results/<name>-<time>` branch without force | `test_notebook_runs_a_pinned_commit`, `test_notebook_never_pushes_to_master_never_forces_and_is_off_by_default` |

## 2. Answers, prompts and records

- **Canonical answer is now the stated answer** (`debate.canonical_answer: stated`): the letter in
  the agent's last `Answer: (X)` line, which is what its peers read. If none parses, the readout
  answer is used and the row is flagged `answer_source = readout_fallback`. The readout is kept in
  `readout_answer` / `readout_correct`; disagreement is reported. Peer metadata and the misleading
  target use the canonical answer. Old runs (no `answer` column) still analyse on the readout.
  `readout` mode reproduces the 1B protocol.
- **Prompts:** a brevity instruction (about 150 words); round 1 asks for exactly the lines round 0
  asked for (previously always "Answer / Confidence", even when round 0 asked for one line).
  Every template, including the fabricated-peer and new answer-only templates, is in the hash
  inventory; a test fails if a module-level template is missing.
- **Tie rule:** `base.yaml` said `round0_plurality`, which nothing implements, and the setting was
  never read. It is now `round0_then_agent0`, validated at load (unknown values raise), and the
  summary reads it from the job file.
- **Records:** structured messages, rendered prompts, generated token ids, stop reason (stop token
  vs cap), request ids, per-condition seeds, peer order and sources, a local snapshot of every
  question, the model/tokenizer identity, option-letter token ids and special tokens.
- **Letter scoring in context:** at load, the HF backend checks that `... Final answer: (` + letter
  tokenizes as the prefix plus exactly the scored letter token, for every letter. Passed on the
  Gemma 3 tokenizer (270M); runs again on the 12B at load.
- **Shown vs panel metadata:** `shown_peer_*` (what the agent saw) and `panel_peer_*` (what its real
  panel-mates answered) are separate columns. The legacy columns keep the "shown" meaning; in
  `self_reconsider` they are now −1 rather than silently panel-based.
- **Seeds:** round 0 and each condition's round 1 get independent derived seeds per chunk
  (previously every chunk used the same seed, and every condition used `seed + 1`). HF sampling
  remains per-batch seeded (transformers has no per-sequence generator); batch composition is
  recorded, and batch-size changes between sessions are allowed and logged.

## 3. Protocol

- **Misleading/adversarial target:** a wrong option drawn uniformly at random (never gold, never the
  agent's own answer) from a stable seed. Previously the lowest eligible letter, which is
  confounded with the measured position bias (1B accuracy 70% when the answer was A, 23% when D).
- **Peer order** randomised per (condition, question, agent) and recorded; in `misleading_one` the
  genuine peer is chosen at random from the two real ones.
- **New condition `debate_answer_only`:** peers' stated letters and option text, no reasoning.
- **Activation positions:** `readout_last` (the forced `(`), `generation_last` (last generated token;
  for the peer-exposure state, the last prompt token), `stated_answer` (the letter in the agent's
  own final answer line). Configurable; all three on for the 12B pilot (~1.1 GB per condition per
  120 questions). Position alignment tested against the real tokenizer.
- **Preregistration draft:** `docs/preregistration.md`. Draft only; open decisions are bracketed.

Not implemented, because each is a scope change for you to decide: the candidate-scoring API and
controlled endorsement pairs (scoring fixed answer candidates rather than sampled revisions).

## 4. GPU measurement code paths (written; not run on a GPU)

- Readout: `use_cache=False`; outputs released after each batch; separate `readout_batch_size`.
- Memory: allocated and reserved peaks per chunk; the start-up estimate now uses the longest
  question plus four full responses (the round-1 readout), not one question plus three.
- `--smoke` now runs a stress pass first (full-length prompts forced to generate the whole token
  budget, then a readout at the longest length), then checks finite and non-degenerate states,
  normalised letter distributions, per-row option ranges, expected state counts per position, and
  batched-vs-single readout agreement.
- On CPU with the 270M model: stress pass completed; batched and single readouts identical.

## 5. Evaluation

- **Threshold leakage fixed.** `scripts/reanalysis.py` chose τ by maximising accuracy on the held-out
  rows' own labels, and `docs/REANALYSIS.md` described it as tuned on training questions. Now τ,
  layer, regularisation, tree settings and calibration are all chosen from out-of-fold predictions
  inside the training questions; the outer loop is grouped 5-fold so every question is held out
  once per repetition; panel differences have paired bootstrap intervals over questions. A test
  flips every held-out label and checks that no choice and no prediction changes.
- **Stronger output baseline:** full letter distributions before and after, P(old) and P(new) under
  each, margins, entropies, panel agreement with old and new answers, lengths, categorical answer
  positions; fitted as logistic regression and as gradient-boosted trees.
- `docs/REANALYSIS.md` corrected: see its top section. `docs/PEER_EXPOSURE.md`: overlapping splits are
  no longer called replications, and spreads are labelled as spreads, not intervals.
- **Summary** adds paired differences (after round 1 vs the first vote, vs self-reconsideration,
  panel-optimal vs after), revisions by type with distinct-question counts, cap-hit and
  stated/readout disagreement rates, answer-source shares, completeness, and labels the
  agent-optimal policy as not a panel bound beside the eight-pattern panel optimum.

## 6. Commands

See `docs/COLAB.md` → "Commands". In order: smoke (`--smoke`), pilot (`--pool pilot --n-questions
120`), summary, then — only after the preregistration is dated — the main run (`--pool main`).
Diagnostics: `<name>-status.json` during a run, `smoke-<name>-smoke-report.json` after the smoke
test, `summarize_run.py` after a run.

## 7. Exports

`artifacts/review/` — see its README.
