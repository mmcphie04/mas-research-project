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
