# GPU job specification: fresh-question validation of candidate assessment (DRAFT — not launched)

Prepared 2026-10-13. **Do not launch without the author's explicit authorization.** Rationale and
decision rules: docs/NEXT_STEP_MEMO.md; frozen analysis: docs/preregistration.md §10.

## Questions

- **Pool:** the 200 MMLU-Pro questions immediately after the 120-question pilot pool in the fixed
  master order (`--pool validation --pilot-size 120 --validation-size 200`): positions 121–320,
  subject-interleaved, so roughly equal numbers per subject (the master order is a round-robin
  over subjects).
- **Disjointness:** the pilot pool is positions 1–120; the main (final confirmation) pool starts at
  position 321 for every later job that passes `--validation-size 200` (the notebook passes it to
  every stage). The driver refuses any job whose questions overlap a job of another pool in the
  same output root (`mad.job.pool_overlaps`; `test_a_job_overlapping_another_pools_questions_is_refused`).
- **Status:** validation/development. Its results may change the main design, so it is not the
  final test.

## Conditions

| job | conditions | questions | why |
|---|---|---|---|
| `val12b` | round 0 (shared) + `debate` + `independent` | 200 | debate's six answers vs six fully independent answers, from the same first three |
| `val12b-self` (optional child) | `self_reconsider` | first 100 of `val12b` (`--parent val12b --subset-n 100`) | reconsideration produced the most new correct options in the pilot (9/120) |

The `independent` condition samples the round-0 prompt afresh (the agent sees nothing of its
first answer) and stores the same states as a revision. It has run only on the stub backend, so
the GPU smoke test below must include it.

## Cost (measured basis, then caps)

Pilot measurements (gemma-3-12b-it, A100-SXM4-40GB, bf16 storage, batches 8/4, cap 768 tokens),
seconds per question: round 0 19.3, debate 11.8, self-reconsideration 17.6 (artifacts/pilot12b/report).
`independent` is unmeasured; estimate 20 (a round 0 plus one exposure readout).

| part | raw chunk compute |
|---|---|
| `val12b`: 200 × (19.3 + 11.8 + 20) s | 2.84 h |
| optional `val12b-self`: 100 × 17.6 s | 0.49 h |

In the pilot, account usage was 1.43 × chunk compute (two smoke attempts, setup, a lost
attempt, idle time) and the driver's elapsed time 1.03 ×. Expected account usage for `val12b`,
smoke included: 3.3–4.1 h.

- **Caps.** `VALIDATION_MAX_HOURS = 3.3` (the driver's cumulative elapsed time; collection stops
  at the next chunk boundary). Account stop rule: if usage for this job reaches **26.5 units
  (5.0 h at 5.30/h)**, smoke included, stop the runtime and do not resume.
- **The child runs only if** usage after `val12b` is at most 21 units (4.0 h); its own
  `--max-hours 0.6`.
- **Quota checked 2026-10-13:** 281.34 units, no active assignment (`colab usage`). That is about
  53 A100-hours at 5.30/h, but the working budget is the author's 10–20 GPU-hour goal. About 3.5 h
  are used, so this job leaves at least 11.5 h of a 20-h budget for confirmation.
- **Disk:** about 3.6 GB of activations (round 0 4.5 MB per question; each condition 6.8 MB per
  question).

## Procedure (notebook `notebooks/colab_runner.ipynb`, mounted Drive)

1. **Pin the commit** that contains this specification, the validation pool and the frozen
   scorers (`COMMIT = <40-character hash>`).
2. **Freeze the analysis.** Date docs/preregistration.md §10 *before* launch. The scorers were
   frozen at `8a59105`: `runs/frozen/pilot12b_scorers.pkl`, SHA-256 `1a9b2ed58809fae60cf0672fd510bd3023738239ae61677ba7411f34c7399716`
   (`artifacts/pilot12b/probe_validation/frozen_scorers.json`). Confirm the laptop copy matches
   before analysing.
3. **Smoke run.** Settings: `RUN_NAME = "val12b"`, `STAGE = "smoke"`, `SMOKE_FOR = "validation"`
   (the smoke uses the validation pool and conditions). It must pass all checks for `debate` and
   `independent` at batches 8/4. Read the stress and smoke reports, and the measured seconds per
   question for `independent`. If the projected chunk compute exceeds 3.0 h, stop and report.
4. **Collect.** `STAGE = "validation"`, `VALIDATION_MAX_HOURS = 3.3`, Run all. During collection
   no outcome is displayed; only the archive check runs. Back up with mounted Drive plus an
   independent verified copy; the CLI backup has not been exercised live.
5. **Optional child.** Only under the usage rule above: same notebook,
   `RUN_NAME = "val12b-self"` with `--parent val12b --subset-n 100 --conditions self_reconsider`.
6. **Offline** (laptop, CPU):
   `verify_archive.py`; `probe_search.py cache --root <archive> --prefix val12b --cache runs/cache/val12b`
   (conditions discovered; checks shared round 0);
   `probe_validate.py apply --frozen runs/frozen/pilot12b_scorers.pkl --cache runs/cache/val12b --out artifacts/val12b/frozen`.
   This refuses if the output-feature columns differ from training. Then the §10 analysis.

## Stop conditions during the job

- Any smoke check fails → stop, no collection.
- Non-finite stored activations, or a failed chunk twice → the driver stops; report, do not
  rerun blindly.
- Usage cap reached → stop the runtime; analyse what is complete (whole chunks only), labelled
  as truncated.
