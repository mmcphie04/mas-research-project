# Corrected probe comparison: portable exports (no activation arrays)

Produced by `scripts/probe_validate.py` at the commit in `provenance.json` (3857a72, clean tree).
`leave_condition_out.json` is carried over from 3e0eea0: it uses only the activation scorer, whose
code did not change. The earlier version of this directory is in `../superseded/probe_validation_3e0eea0/`. With
these files the result can be audited without the 7 GB activation cache. Interpretation:
docs/PROBE_AUDIT.md. Exploration history: `EXPLORATION_LOG.md`.

| file | content |
|---|---|
| `states.csv` | one row per answer state (1,800): stable `row_id` (`question|condition|phase|a<agent>`; round 0 once, as `round0`), question, agent, phase, condition, displayed answer, gold, correctness, subject, and `present:<position>` (whether that token-position state exists) |
| `scores_R1.csv.gz`, `scores_R2.csv.gz` | out-of-fold scores for every scorer: `row_id`, `seed`, `fold`, then `<scorer>:raw`, `<scorer>:z` (training-normalized) and `<scorer>:p` (Platt probability from training references), six partitions |
| `decisions_R*.csv.gz` | per question, partition, condition and population: the answer chosen by every rule (votes, best first answer, best of six, gates), correctness, origin (round 0 or revision), new-option flags, agent-level gate counts, random tie-break expectations |
| `metrics_R*.json` | per partition and population: ranking metrics with question-bootstrap intervals and exact pair and question counts; per-(fold) revision cells; calibration; decision summaries with wins, losses and question ids; held-out alignment; orthogonality diagnostics |
| `params_R*.json` | per partition and fold: every scorer's fitted description: layers, per-layer and ensemble normalizers, Platt parameters, chosen C and its inner-CV log-loss, exact output feature names and logistic coefficients; fold of every question; gate thresholds |
| `summary.md`, `summary.json` | generated tables, and paired metric differences on shared bootstrap draws |
| `leave_condition_out.json` | revision and ranking metrics with each condition's revised states left out of training |
| `frozen_scorers.json` | the scorers fitted on the whole pilot for application to new questions: SHA-256 of the local pickle, training population, parameters |
| `provenance.json` | code commit and clean flag; command; Python and package versions; SHA-256 of the archive's verification record, question snapshot and every chunk completion record; SHA-256 of the cache files used (state table, keys, the 9 layer files the two recipes read) |

To reproduce from the archive: `probe_search.py cache`, then `probe_validate.py fit`,
`evaluate`, `summarize`, `loco`, `freeze` (defaults: `runs/cache/probe12b`,
`artifacts/pilot12b/probe_validation`).
