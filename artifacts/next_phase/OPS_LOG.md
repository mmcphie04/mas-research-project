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
