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
| 23:57 | balance 242.42 (the brief's 242.98 less a late-posted 0.56 from G2; stop line kept at 129.98 = 242.98 − 113, so G4 can use at most 112.44); session `main` (A100); code `b33b916` |
| 23:58 | setup verified (code `b33b916`, torch 2.11.0+cu130, A100 40 GB, token present); `run_experiment.py` launched detached (pid 1526); supervisor started on the laptop |
| 23:58 (03:58 UTC) | question ids verified on the runtime: sha256 `9a33c0ea…`, 200 questions; model loaded in 96 s; worst-case round-1 prompt 7,779 tokens (G2's pilot set: 7,119), readout 9,064 |
| 00:06 | stress PASSED (396 s) at gen_batch_tokens 20,000 + expandable segments: worst-length batch 2 (7,779 + 1,280; 33.5 GB reserved), budget-filling 3 × 5,386 (33.8 GB) and 8 × 1,220 (29.8 GB), readout batch 1 at 9,062 tokens (28.9 GB) |
| 00:07–00:22 | one supervisor cycle took 15 min (a CLI call hung until its timeout; recovered); chunk c000 round 0, pool and debate3 backed up and on Google Drive (verified). Reported balance lags use (241.47 at 00:22) |
| 00:40 | chunk c000 complete in all conditions and backed up (laptop + Google Drive verified): 1,790 s (224 s/question), peak 31.9 GB allocated / 32.7 GB reserved; balance 238.10 |
| 01:40–08:39 | laptop suspended (supervisor paused; the runtime job continued); 08:44: job running, 20 / 25 chunks, 8.62 h, balance 183.70; supervisor resumed at 08:39 and is backing up the 17 chunks finished overnight |
| ~09:2x | WSL ended (supervisor killed mid-catch-up). 09:34: job alive, 22 / 25 chunks, 9.50 h; balance 178.08; supervisor restarted (job not touched); local archive at that point: chunks c000–c002 |
| ~09:50 | WSL rebooted again (the supervisor's single backlog batch -- a 7.36 GB tar of 109 items -- was being downloaded; the CLI download's memory grew to ~19.8 GB) |
| 09:59 | status only: runtime up, job running, 23 / 25 chunks, 9.79 h; balance 175.29. Nothing lost: chunks c000–c022 complete on the runtime (0.41 GB each), c000–c002 on the laptop and Google Drive |
| 10:01–10:08 | fix (`6114054`): one chunk per batch (`cli_backup.prepare(per_chunk=True)`; the 7.36 GB pending tar discarded, its directories untouched), Drive copy verified before the next fetch, final backup bounded by lack of progress, runtime-side backup code at `/content/backup_code` loaded by path (the runtime kernel had the job checkout's `mad` cached); 332 tests pass |
| 10:09–10:10 | c003 backed up by hand under MemoryMax=6G: fetch + hash check + install 40 s, Google Drive verified (34 s), peak RSS 1.7 GB |
| 10:10 | supervisor restarted under `systemd-run --user --scope -p MemoryMax=6G` (stop line 129.98); c004 backed up and Drive-verified at 10:11 |
| 10:11–10:47 | backlog c004–c023 backed up one chunk per batch, each Google Drive-verified before the next fetch; no failed cycles |
| 10:47 | job process exited: state `finished`, 25 / 25 chunks, 10.75 h, no failures |
| 10:54–11:12 | final backup: c024 (10:54) and the five condition views (11:10), each Drive-verified; `verify_archive` on the complete local archive passed (11:12:31); **runtime stopped 11:12:33** (first attempt); final Google Drive check: 1,066 files, 10.43 GB, 0 differences; nothing incomplete |
| 11:13 | balance 166.86, no active assignments. G4: 75.56 units from the 242.42 at launch (76.12 from the brief's 242.98, which includes G2's late-posted 0.56); cap 113. Operations report: `artifacts/main250/run/g4_operations.json` |

**Standing rule (from G4):** for any supervised run, **the laptop must stay awake** — set the
Windows power settings so it never sleeps (plugged in) for the whole session, and keep a WSL
terminal open. G4's supervisor was paused for 7 h by sleep (01:40–08:39) and killed when WSL went
down (~09:50); the runtime job was unaffected, but nothing would have stopped an idle runtime.

## 2026-10-30: revision-gate judge smoke test (authorized; cap 6 units, stop line 160.86)

| time | event |
|---|---|
| 17:06 | balance 166.86; session `judge` (A100); code `20a4391`; setup verified; `scripts/judge_smoke.py` launched (gemma, qwen3, qwen3_think, llama; 10 events + 10 questions each) |
| 17:08–17:27 | gemma (judge 4 s; own 30 in 235 s), qwen3 (2 s; 68 s), qwen3_think (judge 299 s; own 10 in 128 s), llama (2 s; 90 s): all exit 0; peak reserved ≤ 28.3 GB |
| 17:27 | `judge_smoke.tgz` (39 KB, SHA-256 `45eb9847…`) downloaded, hash matched; session stopped; balance **164.92** (smoke: **1.94 units**, may still post a little more) |

## 2026-10-30: cross-model judge, full run (approved; hard cap 18 units, stop line 146.92)

| time | event |
|---|---|
| 18:34 | balance 164.92; session `judgefull` (A100); code `7ef8e31` (decision bar committed before any output); setup verified; `judge_sequence.py` launched (pid 1234); `judge_supervisor.py` started under MemoryMax=6G |
