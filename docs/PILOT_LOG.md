# Pilot log: gemma-3-12b-it, 120 pilot-pool questions (2026-10-10)

What ran, what went wrong, and where every byte came from. Exports: `artifacts/pilot12b/`.
Full data (3.6 GB, all 49 layers at four positions): `runs/colab/` on the laptop (not in git).

## Route

The Colab CLI (`google-colab-cli`) drove an A100-SXM4-40GB runtime from the laptop, not the
notebook. Code reached the runtime as a git bundle checked out at the pinned commit (verified
clean); the Hugging Face token was copied to the runtime's standard location and never printed.
Because a CLI session cannot mount Drive unattended, **the laptop was the persistent
destination**: after each completed chunk, a runtime-side script packed every newly complete
directory (complete only after an atomic rename) and the laptop downloaded and unpacked it.
The scripts are in `artifacts/pilot12b/ops/`.

## Timeline (local time, EDT)

| time | event |
|---|---|
| 01:00 | session 1; setup at `7f0ab97` (transformers 5.17.0 on Colab's torch 2.11 installed cleanly) |
| 01:03–01:07 | smoke attempt 1: model loaded; **stress failed** -- stored activations non-finite at every readout batch size |
| 01:10 | diagnosis on the A100: the model's bfloat16 hidden states are finite, but one dimension reaches 163,840 at the readout position in layers 40–47 (over 800,000 elsewhere); the float16 storage cast overflowed. Session stopped |
| 01:10–01:18 | fix: activations stored as bfloat16 (`5ff52b3`); 209 tests pass |
| 01:20–01:26 | session 2; **smoke passed** at generation batch 8 / readout batch 4 (`artifacts/pilot12b/smoke/`) |
| 01:28 | pilot launched, `--max-hours 3` |
| 01:56, 02:21, 02:42, 03:06 | chunks 1–4 complete (80, 74, 63, 74 s/question); chunks 1–3 backed up and checksum-verified on the laptop |
| ~03:36 onward | **the laptop-side monitor (a 30-minute watch) was not re-armed**; chunk 4's backup never ran, chunks 5–6 were never backed up |
| by 09:19 | the runtime had been reclaimed by Colab after the pilot stopped; **chunks 4–6 (60 questions) were lost** with it. No part of that data was ever downloaded, read or analysed |
| 09:22 | session 3; job restored from the laptop (job file, question snapshot, status, ledger, stress report, and the small files of chunks 1–3 -- no arrays); same commit `5ff52b3`, so the job resumed without a code change; the stress report was reused (same settings, code, packages, GPU type) |
| 09:26–10:42 | chunks 4–6 **re-collected** (76, 67, 87 s/question), each backed up on completion by a non-expiring background watcher |
| 10:01 | the watcher exited on a false "session gone" (a timeout traceback contained "not found"); restarted within a minute after confirming the session with `colab status`; no data or time lost |
| 10:47 | pilot finished (0 failed chunks); final backup; session stopped automatically |

## Data lineage and deviations

- Chunks 1–3 (questions 1–60 of the pilot pool): collected in session 2.
- Chunks 4–6 (questions 61–120): collected in session 3, from the same commit, settings, seeds
  and prompts. The first attempt at these chunks was lost unseen, so the re-collection is not a
  selection on outcomes. Round-0 seeds are derived from the chunk, so the re-collection used the
  same seeds; Hugging Face sampling is not bit-reproducible, so the texts differ from the lost
  attempt's (which no one saw).
- The ledger (`pilot12b-ledger.jsonl`) has no record of the period after the last backup in
  session 2 (that runtime's copy was lost); the job history records `created` then `resumed`.
- `--skip-hash-check` was used for the resumed session's own view building on the runtime (the
  restored chunks 1–3 had no arrays there). On the laptop, `verify_archive.py` then checked every
  checksum of all 24 chunk directories, rebuilt and compared all four views, and loaded every layer:
  all 120 questions per condition, 49 layers × 3,840, finite (`pilot12b-verify.json`).
- **Compute accounting** (three different quantities; use the account balance for budgeting):
  - successful chunk compute (sum of the six completed chunks' timings): **2.47 h**;
  - job status `elapsed_hours_total`: **2.53 h** (adds model loading and the stress pass for the
    two recorded sessions; excludes the lost first attempt at chunks 4–6);
  - the ledger records only one `session_end` (session 2's ended unrecorded when its runtime was
    reclaimed), so ledger-derived session hours (~1.3 h) are a lower bound only;
  - **actual account usage: 18.66 units (300 → 281.34), ≈ 3.52 A100-hours at the 5.30/h rate**,
    including both smoke attempts, setup, the lost attempt and idle time before reclamation.
- **Truncation does not drive the outcomes.** 18 first answers on 11 questions hit the 768-token
  cap (verified from rows); excluding those 11 questions (109 left) gives the same pattern:
  original vote 65, self-reconsideration 53, debate 62, answer-only 64, misleading 61
  (`artifacts/pilot12b/outcomes/`). All three capped self-reconsideration revisions are on those
  same 11 questions.
- **Views rebuilt with checksum verification** on the complete laptop archive (2026-10-12): the
  runtime-built views had `hashes_verified: false` because the resumed session lacked chunks
  1–3's arrays; the rebuilt views are identical in content and record `hashes_verified: true`.

## What changes because of this

- Activations are stored as bfloat16 (the float16 overflow would have corrupted every 12B run).
- For CLI runs: the pilot's backup scripts marked a chunk as packed BEFORE downloading it, so a
  failed download followed by a repack could drop that chunk (reproduced in review). Replaced by
  `mad.cli_backup` / `scripts/colab_backup.py`: prepare -> transfer -> local checksum
  verification -> acknowledgment, the same batch resent until acknowledged, and the runtime
  stopped only after the COMPLETE local archive verifies (tests: `tests/test_cli_backup.py`). Not
  yet exercised against a live runtime. For main collection, mounted Drive through the notebook,
  plus an independent verified copy, remains the recommended route.
- MMLU-Pro question ids look numeric; transcript readers now read them as text (the pilot report
  would otherwise have matched no transcripts to rows).
