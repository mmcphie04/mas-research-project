# Exploration log: probes on the 12B pilot (all on the same 120 pilot questions)

Every step below inspected the same 120 questions. Later steps were chosen after seeing earlier
ones. Nothing here is a confirmatory sample. Dates are 2026.

| # | date | step | selection it involved | status |
|---|---|---|---|---|
| 1 | 10-12 | `reanalysis.py --mode cv`: per-time probes on 360 debate events (docs/PILOT_ANALYSIS.md §2) | model selection changed from AUROC to log-loss after seeing near-constant models | exploratory; superseded selection rule archived in `artifacts/pilot12b/superseded/` |
| 2 | 10-12 | `probe_search.py screen`: 11,766 configurations (49 layers × 4 positions × 4 transforms × 5 methods × 3 CV seeds) on 1,800 pooled states | the screen itself is a search; its maxima are optimistic | descriptive only |
| 3 | 10-12 | `nested` over 896 single-layer candidates (2 outer seeds) | selection inside outer folds | honest for that procedure; within-panel figures stand, revision figures used the cross-fold metric (superseded) |
| 4 | 10-12 | `nested-ens` over 48 layer-band ensembles (3 outer seeds) | ensembles proposed after step 3 | used fold-own standardization (superseded) |
| 5 | 10-12 | R1 `readout_last:letter:meandiff:15-39` declared after the screen, before steps 4 and 6 | chosen after inspecting the screen (template position, simplest method, broad band) | post-screen fixed recipe |
| 6 | 10-12 | `followup` controls on R1 (option-text and all-output residualization, length-matched pairs, leave-condition-out) | none beyond step 5 | cross-fold revision metric (superseded); within-panel figures stand |
| 7 | 10-12 | equal-weight activation + output sum | **post hoc**, proposed after seeing the stacked model dilute the revision signal | old version used fold-own standardization (superseded) |
| 8 | 10-13 | audit corrections (docs/PROBE_AUDIT.md §1) | none | — |
| 9 | 10-13 | corrected comparison of R1 and R2 `before_stated_answer:letter:pca16:21-33` (R2 = the configuration step 4's selections converged on), six partitions (seeds 0, 1, 2, 10, 11, 12), matched comparators, natural debate primary | the two recipes were fixed by the brief before this run. No third recipe was chosen from its results. The output C grid was widened once, after a smoke run, for the comparator | this directory |
| 10 | 10-13 | frozen scorers fitted on the whole pilot at 8a59105 | none | superseded |
| 11 | 10-13 | second correction round at 3857a72: frozen application by schema, tuning preprocessing inside folds, pure combined-scorer references; bounded comparison re-run (same recipes, seeds) | none | ranking metrics and decisions unchanged except R2's learned combination; this directory |
| 12 | 10-13 | final frozen scorers at 3857a72, SHA-256 8e5785fe… (`frozen_scorers.json`) | none (fixed recipes and comparators) | for the validation tranche, before its data exist |
