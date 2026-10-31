# Cross-model judge of revisions: results (EXPLORATORY)

As committed in `docs/REVISION_JUDGE_PLAN.md`. Question-grouped 95% intervals (10,000 resamples). Limitations: possible MMLU-Pro contamination in Qwen/Llama; events from Gemma's revisions only; the revision is identifiable from its content (98.4% mention the panel); the 149 questions skew hard and disputed; events mix 150- and 250-word protocols.

## Per judge (helpful/harmful events)

| judge | events | pooled accuracy | on helpful | on harmful | AUROC | mean P(revision): all / helpful / harmful / wrong→wrong | position bias |
|---|---|---|---|---|---|---|---|
| gemma | 391 | 0.621 [0.567, 0.677] | 0.815 [0.727, 0.889] | 0.429 [0.344, 0.518] | 0.705 [0.626, 0.780] | 0.636 / 0.740 / 0.524 / 0.642 | -0.198 [-0.227, -0.169] |
| qwen3 | 391 | 0.675 [0.600, 0.747] | 0.708 [0.607, 0.798] | 0.643 [0.551, 0.733] | 0.766 [0.681, 0.842] | 0.493 / 0.636 / 0.354 / 0.492 | -0.240 [-0.266, -0.213] |
| qwen3_think | 88 | 0.852 [0.707, 0.962] | 0.860 [0.689, 0.976] | 0.844 [0.689, 0.961] | 0.924 [0.806, 0.989] | 0.482 / 0.791 / 0.210 / 0.468 | -0.122 [-0.152, -0.092] |
| llama | 391 | 0.575 [0.517, 0.635] | 0.846 [0.768, 0.918] | 0.306 [0.226, 0.390] | 0.657 [0.568, 0.743] | 0.654 / 0.719 / 0.599 / 0.647 | -0.031 [-0.049, -0.014] |

## Decision bar

| judge | (a) accuracy ≥ 0.75 | judge − logistic gate | judge − boosted-tree gate | judge − own-answer rule | (b) | (c) | met |
|---|---|---|---|---|---|---|---|
| gemma | no (0.621) | -0.079 [-0.168, 0.013] | -0.023 [-0.099, 0.053] | -0.013 [-0.121, 0.095] | no | no | **not met** |
| qwen3 | no (0.675) | -0.026 [-0.128, 0.077] | 0.031 [-0.056, 0.118] | -0.086 [-0.176, 0.006] | no | no | **not met** |
| qwen3_think | yes (0.852) | 0.227 [0.014, 0.423] | 0.239 [0.060, 0.388] | 0.148 [-0.050, 0.345] | yes | no | **not met** |
| llama | no (0.575) | -0.125 [-0.225, -0.021] | -0.069 [-0.153, 0.016] | -0.097 [-0.195, 0.010] | no | no | **not met** |

Comparator accuracy on the same events: gemma: logistic 0.701, boosted 0.645, own-answer 0.634, combined 0.680; qwen3: logistic 0.701, boosted 0.645, own-answer 0.761, combined 0.731; qwen3_think: logistic 0.625, boosted 0.614, own-answer 0.705, combined 0.818; llama: logistic 0.701, boosted 0.645, own-answer 0.673, combined 0.650.

Thinking subsample (same events): Qwen3 thinking 0.852 [0.707, 0.962], Qwen3 non-thinking 0.716 [0.537, 0.860].

## Simulated gated panels (complete sampled panels; panel accuracy − accept-all)

| gate | gemma | qwen3 | qwen3_think | llama |
|---|---|---|---|---|
| accept_all | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| reject_all | 0.013 [-0.033, 0.057] | 0.013 [-0.033, 0.057] | -0.006 [-0.083, 0.067] | 0.013 [-0.033, 0.057] |
| a_new_is_peer_majority | 0.015 [-0.027, 0.057] | 0.015 [-0.027, 0.057] | -0.009 [-0.087, 0.065] | 0.015 [-0.027, 0.057] |
| b_conviction_reject_if_conf_before_ge_90 | 0.016 [-0.025, 0.058] | 0.016 [-0.025, 0.058] | -0.003 [-0.076, 0.069] | 0.016 [-0.025, 0.058] |
| c_mad_conformist_keep_if_old_is_prev_majority | 0.014 [-0.030, 0.058] | 0.014 [-0.030, 0.058] | -0.009 [-0.075, 0.055] | 0.014 [-0.030, 0.058] |
| logistic | 0.036 [-0.001, 0.073] | 0.036 [-0.001, 0.073] | 0.021 [-0.045, 0.086] | 0.036 [-0.001, 0.073] |
| boosted_tree | 0.031 [0.000, 0.062] | 0.031 [0.000, 0.062] | 0.018 [-0.038, 0.079] | 0.031 [0.000, 0.062] |
| judge | 0.025 [0.003, 0.049] | 0.045 [0.015, 0.075] | 0.106 [0.050, 0.178] | 0.016 [-0.014, 0.047] |
| combined | 0.039 [0.011, 0.069] | 0.056 [0.024, 0.091] | 0.088 [0.039, 0.151] | 0.031 [0.003, 0.061] |
| own_answer | 0.007 [-0.032, 0.045] | 0.056 [0.023, 0.091] | 0.061 [-0.007, 0.137] | 0.034 [0.001, 0.067] |
| perfect | 0.128 [0.094, 0.163] | 0.128 [0.094, 0.163] | 0.130 [0.068, 0.208] | 0.128 [0.094, 0.163] |
| panels | 397 | 397 | 110 | 397 |

## Cross-model (own answers, 250-word protocol)

Own accuracy on these questions: gemma 0.432 [0.365, 0.499] (parse 0.99); qwen3 0.488 [0.418, 0.557] (parse 1.00); qwen3_think 0.500 [0.364, 0.659] (parse 1.00); llama 0.351 [0.291, 0.409] (parse 0.94).

| vs Gemma | questions | correlation of per-question accuracy | same-wrong-answer rate | uniform expectation |
|---|---|---|---|---|
| qwen3 | 149 | 0.326 [0.170, 0.472] | 0.302 [0.213, 0.395] (464 pairs) | 0.118 |
| llama | 149 | 0.435 [0.294, 0.567] | 0.313 [0.250, 0.381] (517 pairs) | 0.119 |
| qwen3_think | 44 | 0.510 [0.262, 0.732] | 0.574 [0.393, 0.750] (54 pairs) | 0.118 |

Mixed 3-answer panels (149 questions; exact expected accuracy averaged over sample choices): GGG 0.453 [0.376, 0.530], GGQ 0.499 [0.430, 0.568], GGL 0.445 [0.376, 0.515], GQL 0.475 [0.416, 0.536]; differences from GGG: GGQ 0.045 [0.010, 0.082], GGL -0.008 [-0.041, 0.024], GQL 0.022 [-0.038, 0.083].
