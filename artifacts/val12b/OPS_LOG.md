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
