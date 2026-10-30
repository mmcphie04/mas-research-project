# Output-feature gate: follow-ups (EXPLORATORY)

Same data, folds and tuning as `feasibility.md`; out-of-fold throughout; question-grouped 95% intervals (10,000 resamples).

## (i) Within-question AUROC

Questions with both a helpful and a harmful event (same question id, any source): every (helpful, harmful) pair; share in which the helpful revision gets the higher out-of-fold P(helpful), ties ½.

| model | questions | pairs | within-question | ordinary AUROC, same questions |
|---|---|---|---|---|
| logistic | 80 | 809 | 0.600 [0.460, 0.745] | 0.633 [0.518, 0.745] |
| boosted_tree | 80 | 809 | 0.646 [0.548, 0.751] | 0.665 [0.581, 0.747] |

## (ii) Feature-group ablation (cumulative)

| features | logistic AUROC | logistic pooled panel gain | boosted-tree AUROC | boosted-tree pooled panel gain |
|---|---|---|---|---|
| all features | 0.707 [0.618, 0.794] | 0.018 [-0.000, 0.036] | 0.727 [0.659, 0.795] | 0.017 [0.002, 0.032] |
| minus panel-level | 0.715 [0.632, 0.793] | 0.020 [0.003, 0.038] | 0.729 [0.661, 0.795] | 0.016 [0.000, 0.031] |
| minus panel-level, confidence | 0.686 [0.618, 0.753] | 0.016 [0.001, 0.031] | 0.705 [0.637, 0.773] | 0.016 [0.000, 0.031] |
| minus panel-level, confidence, exposure readout | 0.536 [0.481, 0.595] | 0.006 [-0.012, 0.023] | 0.550 [0.495, 0.606] | 0.006 [-0.012, 0.023] |

Panel-level: panel_unanimous, n_peers_new, n_peers_old, new_is_peer_majority. Confidence: conf_before, conf_after, conf_change. Exposure readout: p_old_exposure, p_new_exposure, p_diff_exposure. Panel gain: simulated panel accuracy minus accept-all, pooled over all panels.

## (iii) Pooled panel gain excluding misleading_one

896 panels (pilot12b misleading_one excluded from the evaluation; the gates were trained with it).

| model | gain, excluding misleading_one | gain, all panels |
|---|---|---|
| logistic | 0.011 [-0.008, 0.030] | 0.018 [-0.000, 0.036] |
| boosted_tree | 0.010 [-0.005, 0.025] | 0.017 [0.002, 0.032] |
