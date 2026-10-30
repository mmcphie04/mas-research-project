# Output-feature revision gate: feasibility (EXPLORATORY)

Revisions after seeing peers, all Gemma-3-12B datasets except main250 test: 284 helpful, 287 harmful, 397 wrong→wrong, 422 questions. Target: helpful among helpful/harmful (majority-class accuracy 0.503). Features (fixed): conf_before, conf_after, conf_change, p_old_exposure, p_new_exposure, p_diff_exposure, n_peers_new, n_peers_old, new_is_peer_majority, panel_unanimous, len_before, len_after, lp_before, lp_after, panel_k, structured, protocol_250. 5-fold question-grouped cross-validation; C (logistic, [0.01, 0.1, 1.0, 10.0]) or depth (tree, [1, 2, 3]) and the acceptance threshold tuned in 4 inner grouped folds (threshold: most helpful-minus-harmful accepted on inner out-of-fold predictions). The gate decides every changed answer of the held-out questions.

## Prediction (out-of-fold, helpful/harmful events)

| model | AUROC | AUROC, 150-word data | AUROC, 250-word data | accuracy (tuned threshold) | accepts helpful/harmful | accepts wrong→wrong |
|---|---|---|---|---|---|---|
| logistic | 0.707 [0.618, 0.794] | 0.743 [0.656, 0.823] | 0.662 [0.501, 0.815] | 0.671 [0.601, 0.738] | 55% | 48% |
| boosted_tree | 0.727 [0.659, 0.795] | 0.728 [0.648, 0.804] | 0.725 [0.612, 0.838] | 0.648 [0.595, 0.701] | 59% | 52% |

## Simulated agent and panel accuracy

### all panels (1016 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.598 [0.552, 0.645] | 0.616 [0.568, 0.665] | 0.000 [0.000, 0.000] |
| reject all | 0.600 [0.555, 0.648] | 0.622 [0.572, 0.673] | 0.006 [-0.019, 0.031] |
| (a) accept iff new = peer majority | 0.617 [0.569, 0.666] | 0.623 [0.573, 0.673] | 0.007 [-0.016, 0.029] |
| (b) conviction: reject if confidence before ≥ 90 | 0.603 [0.556, 0.651] | 0.623 [0.572, 0.674] | 0.006 [-0.015, 0.027] |
| (c) MAD-Conformist: keep if old = previous majority | 0.623 [0.573, 0.673] | 0.623 [0.572, 0.673] | 0.006 [-0.019, 0.030] |
| logistic gate (CV) | 0.629 [0.580, 0.678] | 0.634 [0.584, 0.684] | 0.018 [-0.000, 0.036] |
| boosted-tree gate (CV) | 0.624 [0.575, 0.672] | 0.633 [0.583, 0.683] | 0.017 [0.002, 0.032] |
| perfect gate (oracle) | 0.683 [0.635, 0.730] | 0.694 [0.645, 0.742] | 0.078 [0.059, 0.097] |

### main250 debate3 (102 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.667 [0.582, 0.748] | 0.693 [0.601, 0.778] | 0.000 [0.000, 0.000] |
| reject all | 0.657 [0.575, 0.739] | 0.676 [0.585, 0.765] | -0.016 [-0.075, 0.039] |
| (a) accept iff new = peer majority | 0.676 [0.585, 0.765] | 0.676 [0.585, 0.765] | -0.016 [-0.075, 0.039] |
| (b) conviction: reject if confidence before ≥ 90 | 0.670 [0.585, 0.752] | 0.686 [0.598, 0.775] | -0.007 [-0.059, 0.046] |
| (c) MAD-Conformist: keep if old = previous majority | 0.683 [0.592, 0.771] | 0.690 [0.598, 0.778] | -0.003 [-0.059, 0.049] |
| logistic gate (CV) | 0.670 [0.582, 0.755] | 0.686 [0.598, 0.775] | -0.007 [-0.052, 0.039] |
| boosted-tree gate (CV) | 0.676 [0.592, 0.761] | 0.683 [0.592, 0.768] | -0.010 [-0.052, 0.033] |
| perfect gate (oracle) | 0.742 [0.657, 0.824] | 0.739 [0.650, 0.824] | 0.046 [0.010, 0.088] |

### main250 debate5 (102 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.663 [0.580, 0.743] | 0.696 [0.608, 0.784] | 0.000 [0.000, 0.000] |
| reject all | 0.649 [0.569, 0.729] | 0.676 [0.588, 0.765] | -0.020 [-0.074, 0.034] |
| (a) accept iff new = peer majority | 0.669 [0.578, 0.757] | 0.676 [0.588, 0.765] | -0.020 [-0.074, 0.034] |
| (b) conviction: reject if confidence before ≥ 90 | 0.655 [0.573, 0.739] | 0.667 [0.578, 0.755] | -0.029 [-0.083, 0.020] |
| (c) MAD-Conformist: keep if old = previous majority | 0.673 [0.582, 0.759] | 0.676 [0.588, 0.765] | -0.020 [-0.074, 0.034] |
| logistic gate (CV) | 0.669 [0.580, 0.755] | 0.676 [0.588, 0.765] | -0.020 [-0.069, 0.029] |
| boosted-tree gate (CV) | 0.678 [0.592, 0.761] | 0.696 [0.608, 0.784] | 0.000 [-0.049, 0.049] |
| perfect gate (oracle) | 0.733 [0.651, 0.812] | 0.755 [0.667, 0.833] | 0.059 [0.020, 0.108] |

### main250 structured3 (102 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.650 [0.565, 0.735] | 0.673 [0.582, 0.761] | 0.000 [0.000, 0.000] |
| reject all | 0.657 [0.575, 0.739] | 0.676 [0.585, 0.765] | 0.003 [-0.049, 0.052] |
| (a) accept iff new = peer majority | 0.680 [0.592, 0.765] | 0.676 [0.585, 0.765] | 0.003 [-0.049, 0.052] |
| (b) conviction: reject if confidence before ≥ 90 | 0.663 [0.578, 0.745] | 0.676 [0.585, 0.765] | 0.003 [-0.039, 0.046] |
| (c) MAD-Conformist: keep if old = previous majority | 0.680 [0.592, 0.765] | 0.680 [0.588, 0.768] | 0.007 [-0.042, 0.056] |
| logistic gate (CV) | 0.683 [0.595, 0.771] | 0.690 [0.601, 0.778] | 0.016 [-0.020, 0.056] |
| boosted-tree gate (CV) | 0.676 [0.588, 0.765] | 0.683 [0.592, 0.771] | 0.010 [-0.026, 0.049] |
| perfect gate (oracle) | 0.719 [0.634, 0.801] | 0.722 [0.637, 0.807] | 0.049 [0.016, 0.088] |

### main250 structured5 (102 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.653 [0.569, 0.735] | 0.676 [0.588, 0.765] | 0.000 [0.000, 0.000] |
| reject all | 0.649 [0.569, 0.729] | 0.676 [0.588, 0.765] | 0.000 [-0.039, 0.034] |
| (a) accept iff new = peer majority | 0.676 [0.588, 0.763] | 0.676 [0.588, 0.765] | 0.000 [-0.039, 0.034] |
| (b) conviction: reject if confidence before ≥ 90 | 0.657 [0.575, 0.739] | 0.676 [0.588, 0.765] | 0.000 [-0.029, 0.029] |
| (c) MAD-Conformist: keep if old = previous majority | 0.676 [0.588, 0.763] | 0.676 [0.588, 0.765] | 0.000 [-0.039, 0.034] |
| logistic gate (CV) | 0.667 [0.580, 0.751] | 0.667 [0.578, 0.755] | -0.010 [-0.049, 0.025] |
| boosted-tree gate (CV) | 0.669 [0.582, 0.753] | 0.672 [0.578, 0.760] | -0.005 [-0.039, 0.029] |
| perfect gate (oracle) | 0.702 [0.618, 0.784] | 0.706 [0.618, 0.794] | 0.029 [0.005, 0.064] |

### pilot12b debate (120 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.561 [0.481, 0.642] | 0.558 [0.469, 0.647] | 0.000 [0.000, 0.000] |
| reject all | 0.561 [0.483, 0.639] | 0.589 [0.503, 0.672] | 0.031 [-0.031, 0.092] |
| (a) accept iff new = peer majority | 0.589 [0.503, 0.672] | 0.597 [0.511, 0.681] | 0.039 [-0.014, 0.094] |
| (b) conviction: reject if confidence before ≥ 90 | 0.564 [0.483, 0.642] | 0.586 [0.500, 0.672] | 0.028 [-0.025, 0.081] |
| (c) MAD-Conformist: keep if old = previous majority | 0.592 [0.506, 0.675] | 0.589 [0.503, 0.675] | 0.031 [-0.028, 0.092] |
| logistic gate (CV) | 0.592 [0.506, 0.675] | 0.586 [0.497, 0.672] | 0.028 [-0.022, 0.078] |
| boosted-tree gate (CV) | 0.581 [0.497, 0.664] | 0.600 [0.514, 0.686] | 0.042 [-0.000, 0.089] |
| perfect gate (oracle) | 0.650 [0.567, 0.731] | 0.658 [0.572, 0.742] | 0.100 [0.050, 0.156] |

### pilot12b debate_answer_only (120 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.578 [0.497, 0.658] | 0.586 [0.500, 0.672] | 0.000 [0.000, 0.000] |
| reject all | 0.561 [0.483, 0.639] | 0.589 [0.503, 0.672] | 0.003 [-0.056, 0.061] |
| (a) accept iff new = peer majority | 0.586 [0.500, 0.669] | 0.597 [0.511, 0.681] | 0.011 [-0.039, 0.064] |
| (b) conviction: reject if confidence before ≥ 90 | 0.567 [0.489, 0.644] | 0.597 [0.511, 0.681] | 0.011 [-0.039, 0.064] |
| (c) MAD-Conformist: keep if old = previous majority | 0.589 [0.506, 0.672] | 0.592 [0.506, 0.678] | 0.006 [-0.050, 0.064] |
| logistic gate (CV) | 0.600 [0.517, 0.681] | 0.608 [0.522, 0.692] | 0.022 [-0.019, 0.067] |
| boosted-tree gate (CV) | 0.575 [0.492, 0.658] | 0.583 [0.497, 0.672] | -0.003 [-0.036, 0.031] |
| perfect gate (oracle) | 0.650 [0.569, 0.731] | 0.661 [0.578, 0.744] | 0.075 [0.033, 0.125] |

### pilot12b misleading_one (120 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.481 [0.406, 0.556] | 0.531 [0.442, 0.617] | 0.000 [0.000, 0.000] |
| reject all | 0.561 [0.483, 0.639] | 0.589 [0.503, 0.672] | 0.058 [0.000, 0.117] |
| (a) accept iff new = peer majority | 0.558 [0.481, 0.636] | 0.583 [0.497, 0.669] | 0.053 [-0.000, 0.108] |
| (b) conviction: reject if confidence before ≥ 90 | 0.558 [0.481, 0.636] | 0.594 [0.508, 0.681] | 0.064 [0.014, 0.114] |
| (c) MAD-Conformist: keep if old = previous majority | 0.594 [0.508, 0.678] | 0.586 [0.500, 0.675] | 0.056 [0.000, 0.114] |
| logistic gate (CV) | 0.594 [0.511, 0.675] | 0.597 [0.511, 0.683] | 0.067 [0.011, 0.122] |
| boosted-tree gate (CV) | 0.586 [0.503, 0.667] | 0.600 [0.517, 0.692] | 0.069 [0.019, 0.122] |
| perfect gate (oracle) | 0.644 [0.564, 0.725] | 0.658 [0.572, 0.742] | 0.128 [0.075, 0.183] |

### pilot250 debate3 (12 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.556 [0.306, 0.778] | 0.667 [0.417, 0.917] | 0.000 [0.000, 0.000] |
| reject all | 0.500 [0.278, 0.722] | 0.528 [0.250, 0.778] | -0.139 [-0.333, 0.000] |
| (a) accept iff new = peer majority | 0.528 [0.250, 0.778] | 0.528 [0.250, 0.778] | -0.139 [-0.333, 0.000] |
| (b) conviction: reject if confidence before ≥ 90 | 0.472 [0.222, 0.722] | 0.500 [0.250, 0.750] | -0.167 [-0.417, 0.000] |
| (c) MAD-Conformist: keep if old = previous majority | 0.556 [0.278, 0.806] | 0.583 [0.333, 0.833] | -0.083 [-0.250, 0.000] |
| logistic gate (CV) | 0.639 [0.389, 0.889] | 0.667 [0.417, 0.917] | 0.000 [0.000, 0.000] |
| boosted-tree gate (CV) | 0.611 [0.361, 0.861] | 0.667 [0.417, 0.917] | 0.000 [0.000, 0.000] |
| perfect gate (oracle) | 0.667 [0.417, 0.917] | 0.667 [0.417, 0.917] | 0.000 [0.000, 0.000] |

### pilot250 debate5 (12 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.483 [0.250, 0.717] | 0.500 [0.250, 0.750] | 0.000 [0.000, 0.000] |
| reject all | 0.500 [0.267, 0.733] | 0.583 [0.333, 0.833] | 0.083 [-0.167, 0.333] |
| (a) accept iff new = peer majority | 0.533 [0.267, 0.783] | 0.583 [0.333, 0.833] | 0.083 [-0.167, 0.333] |
| (b) conviction: reject if confidence before ≥ 90 | 0.500 [0.267, 0.733] | 0.542 [0.250, 0.792] | 0.042 [-0.208, 0.292] |
| (c) MAD-Conformist: keep if old = previous majority | 0.533 [0.267, 0.783] | 0.583 [0.333, 0.833] | 0.083 [-0.167, 0.333] |
| logistic gate (CV) | 0.567 [0.317, 0.800] | 0.667 [0.417, 0.917] | 0.167 [0.000, 0.417] |
| boosted-tree gate (CV) | 0.567 [0.317, 0.800] | 0.667 [0.417, 0.917] | 0.167 [0.000, 0.417] |
| perfect gate (oracle) | 0.617 [0.367, 0.867] | 0.667 [0.417, 0.917] | 0.167 [0.000, 0.417] |

### pilot250 structured3 (12 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.556 [0.306, 0.806] | 0.583 [0.333, 0.833] | 0.000 [0.000, 0.000] |
| reject all | 0.500 [0.278, 0.722] | 0.528 [0.250, 0.778] | -0.056 [-0.167, 0.000] |
| (a) accept iff new = peer majority | 0.528 [0.250, 0.778] | 0.528 [0.250, 0.778] | -0.056 [-0.167, 0.000] |
| (b) conviction: reject if confidence before ≥ 90 | 0.472 [0.222, 0.722] | 0.500 [0.250, 0.750] | -0.083 [-0.250, 0.000] |
| (c) MAD-Conformist: keep if old = previous majority | 0.556 [0.278, 0.806] | 0.583 [0.333, 0.833] | 0.000 [0.000, 0.000] |
| logistic gate (CV) | 0.611 [0.333, 0.861] | 0.583 [0.333, 0.833] | 0.000 [0.000, 0.000] |
| boosted-tree gate (CV) | 0.528 [0.278, 0.750] | 0.583 [0.333, 0.833] | 0.000 [0.000, 0.000] |
| perfect gate (oracle) | 0.639 [0.389, 0.889] | 0.667 [0.417, 0.917] | 0.083 [0.000, 0.250] |

### pilot250 structured5 (12 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.500 [0.250, 0.750] | 0.542 [0.250, 0.792] | 0.000 [0.000, 0.000] |
| reject all | 0.500 [0.267, 0.733] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |
| (a) accept iff new = peer majority | 0.533 [0.267, 0.783] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |
| (b) conviction: reject if confidence before ≥ 90 | 0.517 [0.283, 0.733] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |
| (c) MAD-Conformist: keep if old = previous majority | 0.533 [0.267, 0.783] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |
| logistic gate (CV) | 0.517 [0.250, 0.767] | 0.542 [0.250, 0.792] | 0.000 [0.000, 0.000] |
| boosted-tree gate (CV) | 0.533 [0.267, 0.783] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |
| perfect gate (oracle) | 0.567 [0.317, 0.800] | 0.583 [0.333, 0.833] | 0.042 [0.000, 0.125] |

### val12b debate (200 panels)

| gate | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| accept all | 0.600 [0.538, 0.660] | 0.592 [0.525, 0.658] | 0.000 [0.000, 0.000] |
| reject all | 0.588 [0.527, 0.648] | 0.588 [0.522, 0.653] | -0.003 [-0.048, 0.042] |
| (a) accept iff new = peer majority | 0.590 [0.523, 0.655] | 0.585 [0.518, 0.652] | -0.007 [-0.052, 0.038] |
| (b) conviction: reject if confidence before ≥ 90 | 0.585 [0.522, 0.647] | 0.588 [0.520, 0.655] | -0.003 [-0.043, 0.037] |
| (c) MAD-Conformist: keep if old = previous majority | 0.585 [0.518, 0.650] | 0.575 [0.505, 0.640] | -0.017 [-0.060, 0.027] |
| logistic gate (CV) | 0.612 [0.548, 0.675] | 0.612 [0.545, 0.677] | 0.020 [-0.020, 0.062] |
| boosted-tree gate (CV) | 0.612 [0.548, 0.673] | 0.603 [0.535, 0.670] | 0.012 [-0.018, 0.043] |
| perfect gate (oracle) | 0.677 [0.615, 0.737] | 0.693 [0.630, 0.755] | 0.102 [0.063, 0.145] |

## Logistic coefficients (standardized; fitted on all events above)

C = 0.01 (grouped CV AUROC by C: {'0.01': 0.702, '0.1': 0.6989, '1.0': 0.6878, '10.0': 0.6835}); intercept -0.039. Positive = more likely helpful.

| feature | coefficient |
|---|---|
| conf_after | +0.218 |
| panel_unanimous | -0.211 |
| p_new_exposure | +0.163 |
| p_diff_exposure | +0.133 |
| n_peers_old | -0.126 |
| conf_change | +0.088 |
| p_old_exposure | -0.086 |
| n_peers_new | +0.081 |
| len_before | +0.079 |
| new_is_peer_majority | +0.060 |
| missing:conf_after | +0.059 |
| lp_after | +0.056 |
| len_after | +0.046 |
| conf_before | +0.042 |
| protocol_250 | +0.039 |
| missing:conf_change | +0.036 |
| missing:conf_before | +0.030 |
| panel_k | +0.029 |
| lp_before | -0.024 |
| structured | -0.014 |
