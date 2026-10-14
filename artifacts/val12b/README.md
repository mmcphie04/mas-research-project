# Fresh-question validation (val12b): portable record

No activation arrays. The archive (4.1 GB) is on the laptop (`runs/val12b/`) with a verified copy
on a second filesystem; see `OPS_LOG.md`. Results: docs/VALIDATION_RESULTS.md. Plan:
docs/preregistration.md §10, docs/VALIDATION_JOB.md, docs/NEXT_STEP_MEMO.md (all dated before
collection).

| path | content |
|---|---|
| `plan/` | disjointness check against the actual pilot IDs; the 200 planned question IDs (the collected job's list equals it) |
| `ops/` | runtime setup, launch, status and Drive-mirror scripts (Drive was unavailable), laptop monitor |
| `smoke/` | smoke and stress reports for the validation conditions (passed) |
| `frozen/` | frozen-scorer application to the 200 questions (debate + independent): `states.csv` (stable state keys, answers, gold, correctness, position availability); `scores_R*.csv.gz` (raw, training-normalized and Platt scores of every frozen scorer); `decisions_R*.csv` (per question and population: every rule's answer, correctness, origin, gate counts, random-tie expectations); `source_comparison_*.csv` (debate vs independent six per question); `frozen_application.json` (ranking metrics with counts, paired differences on shared draws, paired decisions with wins/losses and IDs, development readings, calibration, shift, operational rates) |
| `frozen_self/` | the same for the 100-question self-reconsideration child (secondary) |
| `OPS_LOG.md` | timeline, measured resources, problems, backup locations |

Reproduce: `probe_search.py cache --root runs/val12b --prefix val12b --cache runs/cache/val12b
--conditions debate independent`, then `probe_validate.py apply --frozen
runs/frozen/pilot12b_scorers_3857a72.pkl --cache runs/cache/val12b --out artifacts/val12b/frozen
--n-boot 10000 --secs-per-question 47.7` (the frozen file's SHA-256 is in §10).
