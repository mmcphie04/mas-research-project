# GPU job: bounded fresh-question validation (validation/development, not confirmation)

Specification dated **2026-10-13**, fixed before any validation question is generated. Analysis:
docs/preregistration.md §10. Decision language: docs/NEXT_STEP_MEMO.md. The tranche is
development data: claims chosen using it need a later, untouched confirmation pool.

## Questions

- **Pool:** the 200 MMLU-Pro questions at master-order positions 121–320
  (`--pool validation --pilot-size 120 --validation-size 200`): subject-interleaved, 14–15 per
  subject over all 14 subjects.
- **Disjointness, checked against the actual pilot IDs** (`artifacts/val12b/plan/check_disjoint.py`):
  - the locally rebuilt master order has the pilot job's recorded SHA-256 (`c4c67294e44a…`);
  - the pilot's 120 IDs are positions 1–120;
  - the 200 validation IDs are disjoint from them (`validation_question_ids.json`, IDs SHA-256
    `20c9bd92a061…`);
  - the final confirmation pool starts at position 321.

  On the runtime the driver also refuses overlap with any job of another pool in its output root.
  A copy of the pilot job file is placed there so that check sees the pilot too.
- **Data version:** the job pins `data.dataset_revision` to MMLU-Pro commit
  `b189ec765aa7ed75c8acfea42df31fdae71f97be`, the snapshot whose master order matched the pilot's.
  The pilot itself ran unpinned. After collection, the job's recorded master-order SHA-256 and its
  question IDs are checked against the plan.

## Conditions

| job | conditions | questions |
|---|---|---|
| `val12b` | shared round 0 + `debate` + `independent` | 200 |
| `val12b-self` (optional child) | `self_reconsider`, using the parent's stored round 0 | first 100 of `val12b` |

`independent`: three fresh samples of the original prompt (new seeds; no exposure to any
first answer). Its "revised" states are resamples, so its old/new transitions are resampling
comparisons, not persuasion. Its stored `peer_exposure` state is an original-prompt state: no peer
was shown. All other settings are the pilot's:

- prompts and their hashes, decoding (temperature 0.7, top-p 0.95, 768 tokens);
- the option permutation (seed 12345), the master order (seed 7);
- the activation positions, and all 49 layers in bf16 storage;
- batches 8/4.

## Cost basis and caps

- **Measured pilot seconds per question:** round 0 19.3, debate 11.8, self-reconsideration 17.6.
  `independent` is unmeasured (estimate 20).
- **Raw chunk compute:** `val12b` ≈ 200 × 51 s = 2.84 h; the child ≈ 100 × 17.6 s = 0.49 h.
- **Measured projection rule:** after `val12b`'s first 20-question chunk, project core chunk
  compute as 200 × that chunk's seconds per question. If it exceeds **3.0 h**, stop before the next
  chunk and report the bottleneck (from the per-stage timings). Generation settings are not
  changed.
- **Driver cap:** `--max-hours 3.3`, cumulative across sessions. The cap is checked before each
  chunk, so a chunk that starts just under it completes (overrun ≤ 1 chunk ≈ 17 min). Resume only
  within the same cumulative cap, never to extend collection.
- **Account cap for the whole phase:** 5.0 h at the verified rate, including setup, smoke, idle
  time and the child. At the previously observed 5.30 units/h that is 26.5 units. The rate is
  read from `colab usage` once the A100 is attached.
- **Child rule:** decided from resource use only, before any outcome is seen. The child runs only
  if core-job usage ≤ 21 units (4.0 h at 5.30/h), with its own `--max-hours 0.6`, inside the same
  5.0-h total.
- **Disk:** core activations ≈ 3.6 GB, the child ≈ 0.7 GB (round 0 referenced, not recopied).
  Runtime disk, Drive free space and laptop space are checked before launch.

## Persistence

1. The job writes to the runtime's local disk (`/content/mas_runs`), the route proven in the pilot.
   Nothing relies on it alone.
2. **Drive (primary persistent copy):** Drive is mounted at `/content/drive` (`colab drivemount`).
   `artifacts/val12b/ops/drive_mirror.py` runs on the runtime. It copies every completed chunk and
   round-0 directory to `/content/drive/MyDrive/mas_runs`, checks each copy against its completion
   checksums, and copies the job's small files.
3. **Laptop (independent copy):** `scripts/colab_backup.py watch` (prepare → download → local
   checksum verification → acknowledge) writes into `runs/val12b`. It is first exercised on the
   smoke job. If Drive cannot be mounted unattended, the laptop copy plus a second verified laptop
   copy on a different filesystem is the fallback, and the report says so.
4. The runtime is stopped only after the complete laptop archive passes `verify_archive.py`:
   - every checksum matches;
   - every state is finite and aligned;
   - round-0 reuse is exact;
   - chunks are complete.

   The Drive mirror must also report every directory verified.
5. **Frozen scorers:** `runs/frozen/pilot12b_scorers_3857a72.pkl`, SHA-256 `8e5785fe62c222c036df888fa9c28b2fccfdce766be33e0047face64461a67fb`
   (frozen at `3857a72` from a clean tree, reproduced byte-identically by a second freeze), with a
   hash-verified copy on a second filesystem (`C:\Users\mmcph\mas_backup\frozen\`). The hash is
   checked again before analysis.

## Procedure

1. Commit everything. The runtime checks out the pinned full commit hash from a git bundle.
2. **Smoke:** a throwaway job over two validation questions, conditions `debate` and
   `independent`, batches 8/4. It includes the long-input stress pass, and must pass every check:
   - all 49 layers, finite;
   - position fixtures, replay, batched vs single readout.
3. **Collect** `val12b` with the caps above. No accuracy or scorer output is displayed during
   collection. Operational checks only: parse and cap rates, memory, timing, archive integrity.
4. **Optional child**, under the child rule.
5. **Verify** both copies, then stop the runtime. Read the account usage.
6. **Offline:** `probe_search.py cache` (conditions discovered; checks shared round 0), then
   `probe_validate.py apply` with the frozen scorers, once. Then the review bundle, a commit, and
   the decision memo.
7. On technical failure or the usage cap: keep and analyse the completed chunks, and report the
   shortfall.
