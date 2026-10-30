# Next phase: GPU operations log (times EDT)

## 2026-10-29: Stage 1 curve smoke (approved: "run smoke")

| time | event |
|---|---|
| 16:35 | balance 258.86 units, no active assignments; hard stop set at 253.56 (1.0 A100-h at the planned 5.30/h) |
| 16:36 | `repo.bundle` at `d7a251b`; `build_core.tgz` SHA-256 `4fb26825…`; `curves_env.json`; HF token uploaded (never printed) |
| 16:36 | session `curves` created (A100) |
| 16:36–16:37 | `ops/setup.py`: code at `d7a251b`; torch 2.11.0+cu130, transformers 5.17.0 (as the archives); NVIDIA A100-SXM4-40GB |
| 16:37 | `ops/curves_launch.py`: build hash verified; smoke launched detached (`--batch-sizes 4 8 16 --max-minutes 25`) |
| 16:40 | one status call lost its connection (CLI `RuntimeError: Connection was lost`); next poll fine. Attached rate **6.73 units/h** (not 5.30) |
| ~16:44 | smoke finished: 4 modes x 392 items complete in 207 s; model files fetched in 69 s |
| 16:47 | report: **FAIL**. Per the stop rule, no core run |
| 16:47 | outputs packed (`smoke_out.tgz`, SHA-256 `ff6a9fa7…`), downloaded, hash matched, copied to `mas_backup/next_commit/` |
| 16:48 | session stopped; balance **257.75** (this session: **1.11 units**, ~12 min) |

## 2026-10-29: core curve run, batch size 1 (approved: "go ahead")

| time | event |
|---|---|
| 16:53 | balance 257.75; hard stop at 251.75 (6 units); session `core` (A100) created; code at `0f4e29e` (no code change since `d7a251b`) |
| 16:53 | setup verified (torch 2.11.0+cu130, transformers 5.17.0, A100-SXM4-40GB); build hash verified; `run --expect-sources core --batch-size 1 --max-minutes 45` launched detached |
| 16:56–17:26 | polled every 4 min with balance checks (cap never approached). The runtime's code wrote status only at the end; fixed since (status at every flush) |
| ~17:26 | complete: 13,851 / 13,851 items, 1,778 s of reading (**measured 29.6 min**, smoke projection 30.3) |
| 17:30 | `core_out.tgz` (SHA-256 `4093f163…`) downloaded, hash matched, copied to `mas_backup/next_commit/`; session stopped; balance **254.05** (this session **3.70 units**, ~37 min) |

Readout path: the model's bf16 logits (`bf16_logits`). Quantized to 0.0625–0.125 logit steps,
deterministic at batch size 1 (§11 of the results). The default is now `fp32_head` (G3).

## 2026-10-29: G1 readout diagnostic (approved; cap 4 units)

| time | event |
|---|---|
| 19:49 | balance 254.05; hard stop at 250.05; session `diag` (A100); code `18969e0` (verdict logic committed before the run) |
| 19:51 | setup verified; build hash verified and unpacked; `readout_diagnostic.py run --max-minutes 20` launched detached |
| 19:58 | one status call returned no output (CLI connection); later calls fine |
| ~20:02 | complete: 680 reads (default and eager attention, both heads), 378 s of reading |
| 20:04 | `diag_out.tgz` (SHA-256 `ae771979…`) downloaded, hash matched, copied to `mas_backup/next_commit/`; session stopped; balance **252.71** (this session **1.34 units**, ~15 min) |

## 2026-10-29: G2 development timing pilot, attempt 1 (approved; cap 13 units)

| time | event |
|---|---|
| 21:41 | balance 252.71 (job stop at 241.21, hard cap 239.71); off-device copy ALL VERIFIED beforehand; session `pilot` (A100); code `856b314` |
| 21:42 | setup verified; `scripts/pilot_g2.py` launched detached |
| 21:44 | dataset loaded, model loaded in 97 s; **stress FAILED**: generation batch 8 at the worst 5-agent length (7,119 prompt + 1,280 new tokens) ran out of memory (peak 37.6 GB allocated / 41.0 reserved); readout batch 1 at 8,402 tokens passed (29.8 GB). The driver stopped before generating anything (as designed). `pilot_g2.py` then crashed on the missing status file (fixed since) |
| 21:53 | records downloaded (`pilot_attempt1.tgz`, SHA-256 `cbb80000…`); session stopped; balance **251.65** (attempt 1: **1.06 units**; 11.94 left under the G2 cap) |

Fix (CPU, before any relaunch): token-budget generation batching (`model.gen_batch_tokens: 24000`
in `configs/main_250.yaml`; worst case 2 per batch, round-0 8 per batch); the stress test now
checks the budget-derived worst-case batch.

## 2026-10-29: G2 development timing pilot, attempt 2 (approved; remaining cap 11.94 units)

| time | event |
|---|---|
| 22:14 | balance 251.65 (cumulative G2 job stop at 241.21, hard cap 239.71); session `pilot2` (A100); code `671f438` |
| 22:15 | setup verified; `scripts/pilot_g2.py` launched detached |
| ~22:24 | model loaded; stress PASSED: generation worst-length batch 2 (7,119 + 1,280 tokens; 33.7 GB reserved), budget-filling 3 × 6,720 (37.7 GB) and 8 × 1,720 (33.3 GB), readout batch 1 at 8,402 tokens (29.6 GB) |
| 22:24–23:15 | parent: 3 / 3 chunks finished (chunk seconds 491 / 906 / 1,285; peak reserved 37.3 / **41.3** / 36.0 GB) |
| 23:15–23:32 | child (second rounds + no-peer control on all 4 prespecified subset questions): 3 / 3 chunks finished; its stress at the round-2 worst case (9,813 readout tokens) passed |
| 23:33 | `pilot_attempt2.tar` (823 MB, SHA-256 `f85c7f77…`) downloaded in ~2 min, hash matched; session stopped; balance **242.98** (attempt 2: **8.67 units**; G2 total **9.73** of 13) |
| 23:37 | both archives verified (15 chunk directories each); copies to `mas_backup/pilot250` and Google Drive started |
| 23:42 | pilot backups verified: `mas_backup/pilot250` (both jobs, 15 chunk directories each, `verify_archive`) and Google Drive `mas_research_backup/pilot250` (313 files, 0 differences, `rclone check`); ALL VERIFIED |

## 2026-10-30: G4 main collection (approved; hard cap 113 units, stop line 129.98)

Settings: `configs/main_250.yaml` with `gen_batch_tokens: 20000` (was 24,000; G2 chunk 2 reached
41.3 of 42.4 GB reserved) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the job's
environment (recorded in the job's execution record). Conditions pool, debate3, structured3,
debate5, structured5; chunks of 8; `--max-hours 13.5` as a backstop. The job refuses to start
unless its question ids hash to the split manifest's `question_ids_sha256` (`9a33c0ea…`) and the
manifest's dataset revision equals the job's (`b189ec76…`); a CPU check of the real selection
(no model) gives `9a33c0ea…`. Pre-launch: pilot250 laptop archive verified and Google Drive copy
ALL VERIFIED (23:42 on 10-29). Supervisor: `scripts/g4_supervisor.py` on the laptop (per-chunk
backup to `runs/main250` + Google Drive `mas_research_backup/main250`; stops the runtime at
balance ≤ 129.98, after a final backup from ≤ 130.98, and whenever the job process exits).

| time | event |
|---|---|
