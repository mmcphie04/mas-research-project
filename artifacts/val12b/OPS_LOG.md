# Validation job: operations log (times EDT)

| time | event |
|---|---|
| 10-13 23:40 | quota checked: 281.34 units, no assignment; plan committed and pushed (`e4f39aa`) |
| 23:44 | session `val` created (A100); attached rate verified at **5.30 units/h**, so the 5.0-h cap is 26.5 units (stop at a balance of 254.84) |
| 23:45 | setup: code at `e4f39aa` from a git bundle (clean checkout), torch 2.11.0+cu130, transformers 5.17.0 (as in the pilot), NVIDIA A100-SXM4-40GB, 194 GB free runtime disk; HF token placed (never printed) |
| 23:47 | `colab drivemount` failed (`ValueError: mount failed`): Drive cannot be mounted unattended from the CLI, as in the pilot. Fallback per docs/VALIDATION_JOB.md: laptop archive via the acknowledged backup, plus a hash-verified second copy on another filesystem |
| 23:48–23:53 | **smoke test passed** (2 validation questions, `debate` + `independent`, batches 8/4, pinned MMLU-Pro revision). Stress pass at the worst-case lengths passed; all 49 layers × 3840 finite and non-degenerate; position fixtures, replay and batched-vs-single readout passed; exposure semantics checked. Peak 27.85 GB allocated / 29.94 GB reserved of 42.4 GB. Driver estimate 38 s/question with 2 conditions (it flags this as an overestimate: 2 questions batch poorly) |
| 23:55–00:15 | first live run of `scripts/colab_backup.py` on the smoke job: the batch was downloaded, checksum-verified and installed; `verify_archive` later passed on it (2 chunk directories, 49 layers × 3840). A following `colab exec` call then hung until the 900 s subprocess timeout, and `TimeoutExpired` was not caught. **Fixed** afterwards (hung calls become retryable errors; a lost acknowledgment reply is confirmed from the runtime's state; tests added) |
| 00:16 | launching the 200-question collection was **blocked by the agent's permission classifier** (not by any check of the job). No collection started |
| 00:17 | session stopped to avoid idle charges. Usage so far in this phase: **2.47 units** (281.34 → 278.87, ≈ 0.47 h) |
| 10-14 08:21 | user approved; interrupted tasks checked (only wait loops; nothing running; tree clean at `2621a96`); balance 278.42 |
| 08:22–08:31 | A100 "Service Unavailable" three times; session created on the fourth try (rate 5.30/h) |
| 08:31 | setup at `2621a96` (same versions and GPU); collection `val12b` launched (`--max-hours 3.3`, pinned dataset revision). The copy of the pilot job file into the runtime out-root failed (missing helper), so the runtime's own overlap guard ran without it; disjointness was verified beforehand against the actual pilot IDs and afterwards against the collected list |
| 08:49 | chunk 1: 20 questions in 863 s (43 s/q, including model load and stress) → projected 2.4 h < 3.0 h rule; continue. Peak 41.64 GB reserved / 34.5 allocated of 42.4 |
| 08:46–11:52 | laptop acknowledged backup ran live: every chunk downloaded, checksum-verified, acknowledged. One DNS failure (retried); one `colab download` hung 49 min and was ended by hand, after which the batch was resent and verified |
| 11:14 | core job finished: 10/10 chunks, 2.70 h job time, 0 failures; chunk compute 9,540 s = **47.7 s/question** (round 0 18.2, debate generation 10.5, independent generation 17.0) |
| 11:20 | child rule (resources only): usage 17.62 units ≤ 21 → `val12b-self` launched (first 100, `self_reconsider`, `--max-hours 0.6`) |
| by 11:50 | child finished: 5/5 chunks, 0.44 h, 14.8 s/question |
| 11:54 | complete laptop archive verified (`val12b`: 20 chunk directories, 200 questions, 49 × 3840); `val12b-self` verified (5 directories) |
| 12:05–12:24 | second copy `C:\Users\mmcph\mas_backup\val12b` verified independently (both jobs) |
| 12:25 | runtime stop requested; the session had already ended. **Final balance 258.86: this phase used 22.48 units ≈ 4.24 A100-hours (cap 26.5)**, including setup, both smoke sessions, the core job, the child and backups |
| 12:28 | first frozen-analysis attempt stopped by the host for low memory (no result computed or viewed); cause: dense bootstrap pair matrices; fixed (`fe09318`, identical results); rerun on approval |
| 13:28 | frozen analysis run once (46 s, 0.9 GB); reconsideration analysis run once |

Archive locations: laptop `runs/val12b/` (4.1 GB, primary) and `C:\Users\mmcph\mas_backup\val12b\`
(verified copy); frozen scorers `runs/frozen/pilot12b_scorers_3857a72.pkl` and
`C:\Users\mmcph\mas_backup\frozen\` (SHA-256 `8e5785fe…`, rechecked before analysis). Drive was
not available (mount failed); the runtime copy no longer exists.

