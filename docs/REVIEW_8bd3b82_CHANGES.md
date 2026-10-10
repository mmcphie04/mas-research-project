# Changes in response to the review of 8bd3b82

Each finding was checked against the code before changing anything. **Tests for this round are
written but have not been run yet** (at the author's request); the commands are at the end.
Nothing has been run on a GPU. Verification levels: **stub** (tier-0), **270M** (real Hugging
Face path, gemma-3-270m on CPU), **12B** (not yet).

| # | finding | verdict | change | tests (not yet run) |
|---|---|---|---|---|
| 1 | stress bypasses likelihood and extra-position extraction | **confirmed**: the readout stress called the model directly | the stress readout now runs `HFBackend.readout()` itself on synthetic worst-length transcripts ending in a valid answer line, so every layer, every configured position, option probabilities and the response likelihood run; records GPU allocated/reserved and host RSS; likelihood chunk size is a setting (`activations.likelihood_chunk`, an execution field); only the configured batch sizes are required to pass (extras are informational; no automatic promotion) | `test_hf_stress_exercises_the_full_extraction_path` (270M) |
| 2 | stress result reused across batch sizes / lengths / environments | **confirmed**: reuse compared identity, GPU name and pass flag only | `stress_reusable`: reuse only if the configured generation and readout batches were tested and passed, current worst-case lengths and cap do not exceed the tested ones, and identity, GPU identity, packages, code commit, likelihood chunk and positions are unchanged; the report records all of these | `test_stress_report_is_reused_only_when_it_covers_the_current_settings` |
| 3 | smoke does not validate what it promises | **confirmed** (3 of 49 layers; no expected shape; no fixture; letter-token not checked; batched/single lacked `gen_token_ids`; no pre-completion validation) | smoke checks every layer and the known 12B shape (48 × 3,840); a fixed own-response transcript with distracting answer lines (earlier own answer, two peers) checks all four positions, token ids, the scored letter token, the preceding token, and option probabilities against the model's own full logits (0.02, bf16-safe); exposure rows must end at the prompt's last token and lack stated-answer states; a permissive `Answer: C` must be stored on the scored token or flagged; batched/single gets the generated ids and must replay exactly. The backend flags `letter_token_mismatch` instead of storing a stated state on a non-scored token. Records (finite states at every layer/position, normalised distributions, exact replay) are validated before any completion record is written | `test_hf_position_fixture_from_the_smoke_test_passes`, `test_hf_positions_land_on_the_right_tokens_and_skip_peer_answers` (270M); smoke tests (stub) |
| 4 | pilot headroom is a per-question oracle; projections | **confirmed** | headline `room_above_better_baseline` = mean(panel-optimal) − max(mean vote, mean accept-all), max recomputed in each bootstrap draw; the old measure kept as `per_question_oracle_room_above_baselines`; the "upper reference" claim removed; disk and runtime projected from completed questions, each planned condition priced from its own stage timings (`--main-conditions` takes names) | `test_headroom_headline_is_not_a_per_question_oracle`, `test_pilot_report_projects_from_completed_questions_and_per_condition_timings` |
| 5 | verifier accepts edited view tables; child copy omits parent round 0 | **confirmed** | `verify_archive` rebuilds each view from the verified chunks beside it and requires identical tables; `build_view`'s unchanged-stamp shortcut documented as a cache, not a check; `archive_set` copies a child's parent job file, question snapshot and round-0 caches; children cannot be parents. Docs: Drive via the notebook is the mandatory persistent destination; the CLI route has no backup loop and is for short checks only | `test_an_edited_view_table_fails_verification`, `test_a_child_alone_restores_into_a_clean_location_with_its_parent_dependency`, `test_a_child_cannot_be_a_parent` |
| 6 | `Answer: (A)\nAnswer: (Z)` keeps A | **confirmed** (fields matched only A–J) | any uppercase letter is a field; outside the options it invalidates the response | `test_a_final_field_with_a_letter_beyond_J_is_still_the_final_field` |
| 7 | split not operational | **confirmed** (split made after collection; reanalysis ignored metadata; candidates put unmapped questions in train) | `make_split.py` selection mode builds the split before generation from the driver's deterministic selection; `--split-manifest` on the driver requires exact coverage and records the hash in the job identity (children inherit it); `load_split` verifies hash, names, non-empty partitions and membership; reanalysis requires the run to have recorded that split (`--allow-unregistered-split` only for exploration); candidate analysis rejects unmapped questions. Preregistration gets a freeze checklist (primary endpoint/comparator, position, N, split, tolerance, stopping rule) | `test_registered_split_route_end_to_end`, `test_split_integrity_and_membership_are_enforced`, `test_child_job_keeps_its_parents_split`, `test_candidate_analysis_rejects_unmapped_questions`, updated `test_split_route_scores_only_test_questions` |
| 8 | `pos_*_logprobs` never used | **confirmed** | position-matched output features (distribution, P(old)/P(new), margin, entropy) with explicit missingness, by information time (no stated-answer position at exposure); defaults to the activation position; the readout-only baseline kept as the `output_readout_only` ablation | `test_position_matched_output_features_have_explicit_missingness`, `test_future_response_fields_cannot_change_pre_or_exposure_features`, `test_readout_only_ablation_and_constant_baseline_are_exported` |
| 9 | secondary analyses | **confirmed** (no protection rule; no constant baseline export; transfer aggregate-only) | gate rules `difference` and `protect` (keep the first answer iff calibrated P(old) > τ) under panel and agent objectives; `--time pre` + `protect` is the pre-only protection gate; `--time post_all` (pre + exposure + post states); `constant_training_rate` calibration baseline; transfer per-question predictions and panels exported. Old/new targets, revised-only and within-question results were already separate | `test_protect_gate_uses_only_the_old_answer_probability`, `test_readout_only_ablation_and_constant_baseline_are_exported` |
| 10 | 1B prose: "each interval includes or touches zero" | **confirmed** — combined − output, seed 2: +0.023 [+0.003, +0.046] excludes zero | §5 and the change note now say the results are mixed (one of nine intervals excludes zero, two touch it); the first-answer result is labelled post-time, not a pre-debate protection gate | — |

**Optional items.** Independent-answer children on the pilot questions are in the pilot commands
(docs/COLAB.md). Candidate collection: refuses to overwrite a completed run (`--force`), records
the model identity, rendered prompts and token counts, and the instructions inspect 10 questions
before the full set; it stays a single short job (no chunked resume). The candidate prompt format
is unchanged.

**Not done in this round:** the requested exports (stub pilot report, child-dependency
restoration evidence, a stress report from the full extraction path) — they need code runs, held
until the tests are run.

## Commands to run the tests

```bash
# targeted, the failure modes above (stub + 270M; ~5-10 min on this laptop)
uv run pytest -q tests/test_parsing.py tests/test_states.py tests/test_reanalysis.py \
    tests/test_run_experiment.py tests/test_backends.py tests/test_multi.py tests/test_candidates.py
# everything
uv run pytest -q
uv run ruff check .
```
