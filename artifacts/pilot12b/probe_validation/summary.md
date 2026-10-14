# Corrected comparison: generated summary

Produced by `scripts/probe_validate.py summarize`. Partitions are re-splits of the same 120 questions, not replications. Intervals are conditional on the fitted scores and do not account for the search that chose the recipes.

## R1: `readout_last:letter:meandiff:15-39` (layers [15, 18, 21, 24, 27, 30, 33, 36, 39])

### debate|eligible (120 questions; absent states 0)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.652 (0.63–0.68) | 0.754 (0.74–0.79) | 0.637 (0.58–0.66) | 0.688 (0.62–0.71) | 0.619 (0.59–0.63) | 0.716 (0.69–0.73) |
| act_rout | 0.559 (0.52–0.61) | 0.555 (0.54–0.59) | 0.637 (0.55–0.71) | 0.604 (0.56–0.67) | 0.611 (0.56–0.65) | 0.704 (0.65–0.79) |
| comb_lr[full+act] | 0.768 (0.73–0.78) | 0.817 (0.79–0.82) | 0.653 (0.60–0.69) | 0.604 (0.54–0.67) | 0.643 (0.62–0.70) | 0.665 (0.63–0.71) |
| eq[act+out_lr_full] | 0.754 (0.73–0.78) | 0.808 (0.79–0.82) | 0.653 (0.63–0.66) | 0.698 (0.67–0.75) | 0.655 (0.64–0.67) | 0.713 (0.68–0.74) |
| eq[act+out_lr_of+ot] | 0.732 (0.71–0.78) | 0.798 (0.78–0.81) | 0.637 (0.58–0.66) | 0.646 (0.62–0.69) | 0.627 (0.62–0.64) | 0.686 (0.62–0.75) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.764 (0.73–0.79) | 0.613 (0.58–0.73) | 0.625 (0.60–0.65) | 0.633 (0.59–0.69) | 0.660 (0.59–0.70) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.813 (0.79–0.82) | 0.653 (0.60–0.71) | 0.573 (0.50–0.62) | 0.645 (0.63–0.69) | 0.646 (0.62–0.71) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.782 (0.77–0.80) | 0.581 (0.55–0.63) | 0.469 (0.46–0.54) | 0.560 (0.55–0.59) | 0.575 (0.52–0.62) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6234 pairs; panel_pre 62 pairs / 31 questions; panel_post 48 pairs / 24 questions; six 252 pairs / 33 questions; revision_auc 221 pairs / 31 questions; helpful 32, harmful 32, wrong-to-wrong 46

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.116 (0/4) | -0.050 (0/0) | -0.032 (0/0) | +0.115 (0/0) | -0.048 (0/0) | +0.061 (0/0) |
| act - out_hgb[full] | -0.049 (0/0) | -0.015 (0/0) | +0.024 (0/0) | +0.062 (0/0) | -0.008 (0/0) | +0.057 (0/0) |
| comb_lr[full+act] - out_lr[full] | -0.001 (0/0) | +0.000 (0/0) | +0.000 (0/0) | +0.031 (0/0) | -0.002 (0/0) | +0.008 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.006 (0/0) | -0.005 (0/0) | +0.000 (0/0) | +0.125 (3/0) | +0.004 (0/0) | +0.056 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | -0.007 (0/0) | +0.014 (0/0) | +0.040 (2/0) | +0.167 (4/0) | +0.063 (0/0) | +0.116 (3/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 73 | 69 | 70 | 71 | 73 | 72 |
| best_original[comb_lr[full+act]] | 71 | 71 | 72 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 70 | 71 | 71 | 71 | 71 | 70 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 72 | 71 | 71 | 72 | 70 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act] | 75 | 71 | 72 | 73 | 74 | 72 |
| best_six[comb_lr[full+act]] | 71 | 71 | 72 | 72 | 71 | 69 |
| best_six[eq[act+out_lr_full]] | 72 | 72 | 71 | 72 | 71 | 71 |
| best_six[eq[act+out_lr_of+ot]] | 71 | 72 | 72 | 71 | 72 | 71 |
| best_six[out_hgb[full]] | 70 | 72 | 68 | 71 | 68 | 70 |
| best_six[out_lr[full]] | 71 | 71 | 72 | 72 | 71 | 69 |
| best_six[out_lr[of+ot]] | 69 | 68 | 69 | 69 | 68 | 70 |
| gate0[act] | 70 | 70 | 70 | 70 | 70 | 70 |
| gate0[comb_lr[full+act]] | 72 | 71 | 72 | 71 | 71 | 69 |
| gate0[eq[act+out_lr_full]] | 70 | 71 | 70 | 71 | 70 | 70 |
| gate0[eq[act+out_lr_of+ot]] | 71 | 70 | 71 | 70 | 71 | 71 |
| gate0[out_hgb[full]] | 71 | 71 | 70 | 72 | 69 | 71 |
| gate0[out_lr[full]] | 72 | 71 | 72 | 71 | 70 | 69 |
| gate0[out_lr[of+ot]] | 69 | 68 | 69 | 69 | 68 | 69 |
| gate_tuned[act] | 68 | 71 | 70 | 70 | 72 | 71 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 71 | 71 | 71 | 71 | 71 | 71 |
| revised_vote | 67 | 67 | 67 | 67 | 67 | 67 |

Random tie-break expectation: original 70.67, revised 67.00, pooled six 70.17. Questions with a correct round-0 answer 82, with a correct revised answer 80, correct only after revision 1.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.013 | 4/0, 0/0, 2/1, 2/0, 3/0, 2/1 |
| best_six[act] − best_original[act] | +0.017 | 2/0, 2/0, 2/0, 2/0, 1/0, 1/1 |
| best_original[act] − original_vote | +0.021 | 7/3, 3/3, 5/4, 5/3, 6/2, 6/3 |
| gate0[act] − original_vote | +0.008 | 2/1, 2/1, 2/1, 2/1, 2/1, 2/1 |
| gate0[act] − pooled6_vote | -0.008 | 0/1, 0/1, 0/1, 0/1, 0/1, 0/1 |
| best_six[act] − best_six[out_lr[full]] | +0.017 | 6/2, 2/2, 4/4, 4/3, 6/3, 6/3 |
| best_original[act] − best_original[out_lr[full]] | +0.004 | 6/4, 3/4, 5/6, 3/5, 7/3, 7/4 |
| gate0[act] − gate0[out_lr[full]] | -0.008 | 1/3, 1/2, 1/3, 2/3, 2/2, 3/2 |
| best_six[out_lr[full]] − pooled6_vote | +0.000 | 3/3, 2/2, 3/2, 3/2, 3/3, 2/4 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.004 | 1/1, 1/0, 1/0, 0/1, 2/0, 1/1 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | +0.017 | 5/2, 4/2, 5/2, 4/2, 3/2, 3/3 |
| gate0[out_lr[full]] − pooled6_vote | +0.000 | 2/1, 1/1, 2/1, 2/2, 1/2, 1/3 |
| best_six[out_hgb[full]] − pooled6_vote | -0.008 | 2/3, 3/2, 3/6, 3/3, 1/4, 2/3 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.004 | 1/1, 2/1, 0/1, 0/1, 0/2, 0/0 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.017 | 3/1, 4/2, 4/3, 4/1, 2/2, 3/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 2/2, 3/3, 3/4, 3/2, 1/3, 2/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.008 | 0/1, 2/1, 0/4, 0/1, 0/3, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.004 | 1/2, 2/2, 1/3, 1/0, 1/2, 3/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.017 | 2/4, 2/5, 2/4, 2/4, 2/5, 2/3 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.004 | 1/2, 2/2, 1/0, 0/2, 1/2, 2/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 3/3, 3/4, 3/3, 3/3, 3/4, 3/3 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.017 | 1/3, 1/4, 1/3, 1/3, 1/4, 1/3 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.025 | 0/2, 1/4, 1/4, 0/3, 0/3, 3/2 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.021 | 0/3, 1/4, 1/4, 1/3, 0/2, 2/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | +0.000 | 3/3, 2/2, 3/2, 3/2, 3/3, 2/4 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.000 | 1/1, 1/1, 1/1, 1/1, 2/0, 1/1 |
| best_original[comb_lr[full+act]] − original_vote | +0.017 | 3/1, 2/0, 3/0, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.017 | 5/2, 4/2, 5/2, 4/2, 4/2, 3/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/1, 1/1, 2/1, 2/2, 1/1, 1/3 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/0, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 1/0, 1/0, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/0, 0/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.004 | 1/0, 1/0, 1/1, 1/0, 1/1, 1/1 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.008 | 2/0, 1/0, 0/0, 1/0, 0/0, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.017 | 3/2, 4/2, 3/1, 3/1, 3/1, 3/2 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.008 | 2/1, 3/1, 2/1, 3/1, 3/2, 2/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.008 | 0/1, 0/0, 0/1, 1/1, 0/1, 0/1 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.004 | 3/2, 2/1, 2/3, 2/2, 2/2, 3/1 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.004 | 2/3, 3/2, 2/2, 0/2, 3/1, 2/1 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.000 | 1/3, 1/1, 1/3, 2/2, 1/1, 3/2 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.004 | 2/2, 1/0, 2/1, 1/1, 2/1, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.000 | 0/0, 0/0, 1/0, 0/0, 0/0, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 4/2, 4/1, 4/2, 3/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 3/1, 2/1, 3/1, 2/1, 3/1, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 1/1, 0/1, 1/1, 0/1, 1/1, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.004 | 1/1, 2/1, 2/2, 1/2, 2/1, 4/2 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.004 | 1/1, 3/1, 2/2, 0/2, 4/1, 3/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | -0.008 | 1/2, 1/2, 1/2, 2/3, 2/1, 3/1 |
| gate_tuned[act] − original_vote | +0.013 | 2/3, 3/1, 3/2, 2/1, 4/1, 4/2 |

Origin of best-of-six picks (partition 0): act: 74 from revisions, 1 new options (0 correct); out_lr[full]: 76 from revisions, 0 new options (0 correct); out_hgb[full]: 80 from revisions, 0 new options (0 correct); out_lr[of+ot]: 59 from revisions, 0 new options (0 correct); comb_lr[full+act]: 76 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 75 from revisions, 1 new options (0 correct); eq[act+out_lr_of+ot]: 63 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 12, helpful_blocked 20, harmful_accepted 6, harmful_blocked 26, wrong_to_wrong_accepted 10, wrong_to_wrong_blocked 36; gate_tuned[act]: helpful_accepted 16, helpful_blocked 16, harmful_accepted 14, harmful_blocked 18, wrong_to_wrong_accepted 17, wrong_to_wrong_blocked 29; gate0[out_lr[full]]: helpful_accepted 22, helpful_blocked 10, harmful_accepted 8, harmful_blocked 24, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 33; gate0[out_hgb[full]]: helpful_accepted 16, helpful_blocked 16, harmful_accepted 6, harmful_blocked 26, wrong_to_wrong_accepted 16, wrong_to_wrong_blocked 30; gate0[out_lr[of+ot]]: helpful_accepted 17, helpful_blocked 15, harmful_accepted 11, harmful_blocked 21, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 31; gate0[comb_lr[full+act]]: helpful_accepted 22, helpful_blocked 10, harmful_accepted 7, harmful_blocked 25, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 34; gate0[eq[act+out_lr_full]]: helpful_accepted 14, helpful_blocked 18, harmful_accepted 5, harmful_blocked 27, wrong_to_wrong_accepted 9, wrong_to_wrong_blocked 37; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 13, helpful_blocked 19, harmful_accepted 5, harmful_blocked 27, wrong_to_wrong_accepted 8, wrong_to_wrong_blocked 38

Calibration of revised-state probabilities (median over partitions): act Brier 0.202 (constant 0.248), ECE 0.081; act_rout Brier 0.250 (constant 0.248), ECE 0.047; comb_lr[full+act] Brier 0.184 (constant 0.248), ECE 0.069; eq[act+out_lr_full] Brier 0.186 (constant 0.248), ECE 0.082; eq[act+out_lr_of+ot] Brier 0.190 (constant 0.248), ECE 0.098; out_hgb[full] Brier 0.203 (constant 0.248), ECE 0.049; out_lr[full] Brier 0.184 (constant 0.248), ECE 0.068; out_lr[of+ot] Brier 0.197 (constant 0.248), ECE 0.069

### debate_answer_only|eligible (120 questions; absent states 0)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.652 (0.63–0.68) | 0.742 (0.71–0.77) | 0.637 (0.58–0.66) | 0.643 (0.62–0.71) | 0.598 (0.57–0.62) | 0.682 (0.65–0.73) |
| act_rout | 0.559 (0.52–0.61) | 0.483 (0.46–0.53) | 0.637 (0.55–0.71) | 0.476 (0.43–0.50) | 0.550 (0.52–0.59) | 0.598 (0.56–0.64) |
| comb_lr[full+act] | 0.768 (0.73–0.78) | 0.813 (0.78–0.83) | 0.653 (0.60–0.69) | 0.643 (0.60–0.71) | 0.654 (0.61–0.67) | 0.637 (0.62–0.74) |
| eq[act+out_lr_full] | 0.754 (0.73–0.78) | 0.811 (0.78–0.83) | 0.653 (0.63–0.66) | 0.714 (0.69–0.74) | 0.662 (0.66–0.67) | 0.680 (0.67–0.76) |
| eq[act+out_lr_of+ot] | 0.732 (0.71–0.78) | 0.787 (0.76–0.82) | 0.637 (0.58–0.66) | 0.679 (0.64–0.76) | 0.632 (0.62–0.65) | 0.682 (0.61–0.74) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.764 (0.73–0.79) | 0.613 (0.58–0.73) | 0.655 (0.55–0.71) | 0.636 (0.60–0.70) | 0.691 (0.58–0.81) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.813 (0.78–0.82) | 0.653 (0.60–0.71) | 0.667 (0.57–0.69) | 0.652 (0.61–0.67) | 0.640 (0.60–0.72) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.787 (0.77–0.82) | 0.581 (0.55–0.63) | 0.655 (0.62–0.67) | 0.614 (0.58–0.62) | 0.622 (0.56–0.68) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6170 pairs; panel_pre 62 pairs / 31 questions; panel_post 42 pairs / 21 questions; six 250 pairs / 35 questions; revision_auc 207 pairs / 33 questions; helpful 32, harmful 26, wrong-to-wrong 45

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.116 (0/4) | -0.066 (0/0) | -0.032 (0/0) | -0.024 (0/0) | -0.058 (0/0) | +0.042 (0/0) |
| act - out_hgb[full] | -0.049 (0/0) | -0.015 (0/0) | +0.024 (0/0) | -0.000 (0/0) | -0.030 (0/0) | -0.010 (0/0) |
| comb_lr[full+act] - out_lr[full] | -0.001 (0/0) | +0.002 (0/0) | +0.000 (0/0) | +0.000 (0/0) | +0.002 (0/0) | +0.017 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.006 (0/0) | +0.003 (0/0) | +0.000 (0/0) | +0.060 (0/0) | +0.014 (0/0) | +0.059 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | -0.007 (0/0) | +0.003 (0/0) | +0.040 (2/0) | +0.048 (0/0) | +0.026 (0/0) | +0.067 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 73 | 69 | 70 | 71 | 73 | 72 |
| best_original[comb_lr[full+act]] | 71 | 71 | 72 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 70 | 71 | 71 | 71 | 71 | 70 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 72 | 71 | 71 | 72 | 70 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act] | 75 | 71 | 72 | 73 | 74 | 73 |
| best_six[comb_lr[full+act]] | 70 | 70 | 72 | 72 | 68 | 68 |
| best_six[eq[act+out_lr_full]] | 72 | 72 | 71 | 72 | 71 | 71 |
| best_six[eq[act+out_lr_of+ot]] | 72 | 72 | 72 | 72 | 73 | 71 |
| best_six[out_hgb[full]] | 70 | 73 | 69 | 72 | 71 | 72 |
| best_six[out_lr[full]] | 69 | 70 | 72 | 72 | 69 | 69 |
| best_six[out_lr[of+ot]] | 68 | 69 | 69 | 70 | 68 | 68 |
| gate0[act] | 70 | 69 | 69 | 69 | 70 | 69 |
| gate0[comb_lr[full+act]] | 71 | 70 | 71 | 69 | 70 | 70 |
| gate0[eq[act+out_lr_full]] | 70 | 70 | 70 | 70 | 70 | 70 |
| gate0[eq[act+out_lr_of+ot]] | 70 | 70 | 70 | 70 | 70 | 70 |
| gate0[out_hgb[full]] | 70 | 71 | 68 | 71 | 68 | 70 |
| gate0[out_lr[full]] | 72 | 69 | 71 | 69 | 69 | 70 |
| gate0[out_lr[of+ot]] | 69 | 69 | 68 | 69 | 66 | 68 |
| gate_tuned[act] | 69 | 70 | 71 | 71 | 69 | 71 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 70 | 70 | 70 | 70 | 70 | 70 |
| revised_vote | 69 | 69 | 69 | 69 | 69 | 69 |

Random tie-break expectation: original 70.67, revised 70.33, pooled six 71.00. Questions with a correct round-0 answer 82, with a correct revised answer 80, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.025 | 5/0, 2/1, 3/1, 3/0, 4/0, 4/1 |
| best_six[act] − best_original[act] | +0.017 | 2/0, 2/0, 2/0, 2/0, 1/0, 1/0 |
| best_original[act] − original_vote | +0.021 | 7/3, 3/3, 5/4, 5/3, 6/2, 6/3 |
| gate0[act] − original_vote | +0.000 | 2/1, 2/2, 2/2, 2/2, 2/1, 2/2 |
| gate0[act] − pooled6_vote | -0.008 | 1/1, 1/2, 1/2, 1/2, 1/1, 1/2 |
| best_six[act] − best_six[out_lr[full]] | +0.021 | 8/2, 3/2, 4/4, 4/3, 8/3, 8/4 |
| best_original[act] − best_original[out_lr[full]] | +0.004 | 6/4, 3/4, 5/6, 3/5, 7/3, 7/4 |
| gate0[act] − gate0[out_lr[full]] | -0.004 | 1/3, 3/3, 1/3, 2/2, 3/2, 3/4 |
| best_six[out_lr[full]] − pooled6_vote | -0.004 | 3/4, 2/2, 3/1, 4/2, 3/4, 3/4 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 0/2, 1/1, 1/0, 0/1, 1/1, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | +0.004 | 4/1, 2/2, 3/1, 2/2, 2/2, 3/2 |
| gate0[out_lr[full]] − pooled6_vote | -0.004 | 3/1, 2/3, 2/1, 1/2, 2/3, 3/3 |
| best_six[out_hgb[full]] − pooled6_vote | +0.013 | 4/4, 4/1, 3/4, 4/2, 4/3, 4/2 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | +0.004 | 1/1, 2/0, 1/1, 1/1, 2/1, 3/1 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.008 | 2/1, 3/1, 2/3, 3/1, 1/2, 2/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 2/2, 3/2, 2/4, 2/1, 1/3, 2/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | +0.013 | 3/2, 4/1, 0/3, 1/1, 3/1, 4/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.004 | 1/3, 3/1, 0/3, 2/0, 1/2, 2/2 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.013 | 2/4, 2/3, 2/3, 3/3, 2/4, 2/4 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.008 | 0/2, 3/2, 3/2, 0/1, 0/1, 1/2 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | -0.004 | 2/2, 3/3, 3/4, 2/2, 1/4, 2/3 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.013 | 2/3, 1/2, 1/3, 2/3, 1/5, 2/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.008 | 0/1, 1/2, 0/3, 0/2, 0/1, 0/1 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.021 | 1/4, 3/3, 0/3, 1/1, 0/3, 1/3 |
| best_six[comb_lr[full+act]] − pooled6_vote | +0.000 | 3/3, 2/2, 3/1, 4/2, 2/4, 2/4 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | -0.008 | 0/1, 1/2, 1/1, 1/1, 0/1, 1/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.017 | 3/1, 2/0, 3/0, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.008 | 4/2, 3/2, 3/1, 2/2, 3/2, 3/2 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/1, 1/1, 2/1, 1/2, 2/2, 3/3 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 1/0, 0/0, 0/0, 0/0, 0/1, 0/1 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 1/0, 1/0, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/1, 2/1, 0/0, 0/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.013 | 3/1, 3/1, 2/1, 3/1, 2/1, 3/2 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.008 | 2/0, 1/0, 0/0, 1/0, 1/1, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.017 | 3/2, 4/2, 3/1, 3/1, 3/1, 3/2 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.008 | 2/1, 2/1, 2/1, 2/1, 2/1, 2/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | +0.000 | 1/1, 1/1, 1/1, 1/1, 1/1, 1/1 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.017 | 5/2, 3/1, 1/2, 2/2, 4/2, 4/2 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.004 | 2/3, 3/2, 2/2, 0/2, 3/1, 2/1 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.004 | 1/3, 3/2, 1/2, 2/1, 3/2, 3/3 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.017 | 4/2, 3/1, 4/2, 3/1, 4/1, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.008 | 1/0, 0/0, 1/0, 1/0, 1/0, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 4/2, 4/1, 4/2, 3/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.008 | 2/1, 2/1, 2/1, 2/1, 2/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 1/1, 1/1, 1/1, 1/1, 1/1, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.017 | 4/1, 3/1, 2/2, 2/2, 5/1, 5/3 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.004 | 1/1, 3/1, 2/2, 0/2, 4/1, 3/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.004 | 1/3, 3/2, 1/2, 2/1, 3/2, 3/3 |
| gate_tuned[act] − original_vote | +0.013 | 3/3, 3/2, 3/1, 3/1, 3/3, 4/2 |

Origin of best-of-six picks (partition 0): act: 70 from revisions, 1 new options (0 correct); out_lr[full]: 77 from revisions, 0 new options (0 correct); out_hgb[full]: 71 from revisions, 0 new options (0 correct); out_lr[of+ot]: 62 from revisions, 0 new options (0 correct); comb_lr[full+act]: 75 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 71 from revisions, 1 new options (0 correct); eq[act+out_lr_of+ot]: 68 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 10, helpful_blocked 22, harmful_accepted 2, harmful_blocked 24, wrong_to_wrong_accepted 9, wrong_to_wrong_blocked 36; gate_tuned[act]: helpful_accepted 16, helpful_blocked 16, harmful_accepted 8, harmful_blocked 18, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 33; gate0[out_lr[full]]: helpful_accepted 18, helpful_blocked 14, harmful_accepted 4, harmful_blocked 22, wrong_to_wrong_accepted 14, wrong_to_wrong_blocked 31; gate0[out_hgb[full]]: helpful_accepted 15, helpful_blocked 17, harmful_accepted 5, harmful_blocked 21, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 34; gate0[out_lr[of+ot]]: helpful_accepted 15, helpful_blocked 17, harmful_accepted 7, harmful_blocked 19, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 33; gate0[comb_lr[full+act]]: helpful_accepted 18, helpful_blocked 14, harmful_accepted 5, harmful_blocked 21, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 32; gate0[eq[act+out_lr_full]]: helpful_accepted 14, helpful_blocked 18, harmful_accepted 3, harmful_blocked 23, wrong_to_wrong_accepted 4, wrong_to_wrong_blocked 41; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 12, helpful_blocked 20, harmful_accepted 3, harmful_blocked 23, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 40

Calibration of revised-state probabilities (median over partitions): act Brier 0.200 (constant 0.247), ECE 0.058; act_rout Brier 0.251 (constant 0.247), ECE 0.053; comb_lr[full+act] Brier 0.182 (constant 0.247), ECE 0.080; eq[act+out_lr_full] Brier 0.183 (constant 0.247), ECE 0.083; eq[act+out_lr_of+ot] Brier 0.184 (constant 0.247), ECE 0.075; out_hgb[full] Brier 0.199 (constant 0.247), ECE 0.062; out_lr[full] Brier 0.183 (constant 0.247), ECE 0.081; out_lr[of+ot] Brier 0.186 (constant 0.247), ECE 0.063

### self_reconsider|eligible (120 questions; absent states 0)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.652 (0.63–0.68) | 0.792 (0.79–0.80) | 0.637 (0.58–0.66) | 0.780 (0.76–0.80) | 0.714 (0.69–0.73) | 0.627 (0.61–0.68) |
| act_rout | 0.559 (0.52–0.61) | 0.550 (0.53–0.59) | 0.637 (0.55–0.71) | 0.518 (0.50–0.57) | 0.605 (0.59–0.65) | 0.531 (0.46–0.58) |
| comb_lr[full+act] | 0.768 (0.73–0.78) | 0.807 (0.79–0.84) | 0.653 (0.60–0.69) | 0.762 (0.72–0.82) | 0.704 (0.70–0.73) | 0.759 (0.73–0.79) |
| eq[act+out_lr_full] | 0.754 (0.73–0.78) | 0.833 (0.82–0.85) | 0.653 (0.63–0.66) | 0.805 (0.77–0.84) | 0.722 (0.72–0.73) | 0.745 (0.70–0.75) |
| eq[act+out_lr_of+ot] | 0.732 (0.71–0.78) | 0.837 (0.83–0.84) | 0.637 (0.58–0.66) | 0.823 (0.80–0.84) | 0.729 (0.72–0.74) | 0.727 (0.69–0.76) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.762 (0.75–0.81) | 0.613 (0.58–0.73) | 0.726 (0.65–0.74) | 0.702 (0.67–0.72) | 0.727 (0.63–0.77) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.806 (0.79–0.83) | 0.653 (0.60–0.71) | 0.756 (0.73–0.79) | 0.705 (0.69–0.73) | 0.753 (0.72–0.79) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.813 (0.80–0.83) | 0.581 (0.55–0.63) | 0.780 (0.77–0.83) | 0.733 (0.73–0.74) | 0.753 (0.70–0.78) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6332 pairs; panel_pre 62 pairs / 31 questions; panel_post 82 pairs / 41 questions; six 494 pairs / 65 questions; revision_auc 575 pairs / 60 questions; helpful 35, harmful 79, wrong-to-wrong 67

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.116 (0/4) | -0.011 (0/0) | -0.032 (0/0) | +0.024 (0/0) | +0.007 (0/0) | -0.115 (0/0) |
| act - out_hgb[full] | -0.049 (0/0) | +0.032 (0/0) | +0.024 (0/0) | +0.055 (0/0) | +0.014 (0/0) | -0.082 (0/0) |
| comb_lr[full+act] - out_lr[full] | -0.001 (0/0) | +0.003 (0/0) | +0.000 (0/0) | +0.000 (0/0) | +0.000 (0/0) | +0.005 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.006 (0/0) | +0.028 (0/0) | +0.000 (0/0) | +0.049 (0/0) | +0.015 (0/0) | -0.025 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | -0.007 (0/0) | +0.027 (0/0) | +0.040 (2/0) | +0.030 (0/0) | -0.003 (0/0) | -0.020 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 73 | 69 | 70 | 71 | 73 | 72 |
| best_original[comb_lr[full+act]] | 71 | 71 | 72 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 70 | 71 | 71 | 71 | 71 | 70 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 72 | 71 | 71 | 72 | 70 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act] | 74 | 70 | 71 | 73 | 73 | 74 |
| best_six[comb_lr[full+act]] | 70 | 71 | 73 | 72 | 69 | 70 |
| best_six[eq[act+out_lr_full]] | 72 | 72 | 70 | 72 | 72 | 73 |
| best_six[eq[act+out_lr_of+ot]] | 73 | 72 | 72 | 73 | 74 | 72 |
| best_six[out_hgb[full]] | 70 | 68 | 71 | 68 | 71 | 69 |
| best_six[out_lr[full]] | 71 | 70 | 73 | 72 | 69 | 70 |
| best_six[out_lr[of+ot]] | 69 | 69 | 69 | 71 | 69 | 69 |
| gate0[act] | 68 | 68 | 68 | 68 | 68 | 68 |
| gate0[comb_lr[full+act]] | 67 | 69 | 68 | 67 | 67 | 67 |
| gate0[eq[act+out_lr_full]] | 69 | 69 | 69 | 69 | 68 | 69 |
| gate0[eq[act+out_lr_of+ot]] | 69 | 69 | 69 | 68 | 69 | 69 |
| gate0[out_hgb[full]] | 67 | 67 | 70 | 69 | 69 | 69 |
| gate0[out_lr[full]] | 67 | 69 | 69 | 68 | 67 | 67 |
| gate0[out_lr[of+ot]] | 71 | 71 | 71 | 70 | 68 | 71 |
| gate_tuned[act] | 67 | 68 | 69 | 68 | 67 | 71 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 72 | 72 | 72 | 72 | 72 | 72 |
| revised_vote | 58 | 58 | 58 | 58 | 58 | 58 |

Random tie-break expectation: original 70.67, revised 54.33, pooled six 69.67. Questions with a correct round-0 answer 82, with a correct revised answer 74, correct only after revision 9.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.008 | 6/4, 5/7, 5/6, 5/4, 5/4, 6/4 |
| best_six[act] − best_original[act] | +0.008 | 1/0, 1/0, 1/0, 2/0, 1/1, 2/0 |
| best_original[act] − original_vote | +0.021 | 7/3, 3/3, 5/4, 5/3, 6/2, 6/3 |
| gate0[act] − original_vote | -0.008 | 0/1, 0/1, 0/1, 0/1, 0/1, 0/1 |
| gate0[act] − pooled6_vote | -0.033 | 2/6, 2/6, 2/6, 2/6, 2/6, 2/6 |
| best_six[act] − best_six[out_lr[full]] | +0.017 | 9/6, 5/5, 6/8, 7/6, 9/5, 9/5 |
| best_original[act] − best_original[out_lr[full]] | +0.004 | 6/4, 3/4, 5/6, 3/5, 7/3, 7/4 |
| gate0[act] − gate0[out_lr[full]] | +0.004 | 3/2, 2/3, 3/4, 3/3, 3/2, 3/2 |
| best_six[out_lr[full]] − pooled6_vote | -0.013 | 6/7, 5/7, 7/6, 7/7, 5/8, 5/7 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 3/3, 3/3, 3/1, 3/4, 3/3, 4/3 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | -0.013 | 1/3, 2/2, 3/3, 2/3, 1/3, 1/3 |
| gate0[out_lr[full]] − pooled6_vote | -0.037 | 2/7, 2/5, 3/6, 2/6, 2/7, 2/7 |
| best_six[out_hgb[full]] − pooled6_vote | -0.021 | 7/9, 5/9, 8/9, 6/10, 7/8, 6/9 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.004 | 4/4, 2/5, 7/5, 3/7, 4/3, 3/4 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.000 | 1/3, 2/4, 5/4, 4/4, 4/4, 3/3 |
| gate0[out_hgb[full]] − pooled6_vote | -0.025 | 1/6, 2/7, 4/6, 4/7, 4/7, 4/7 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.013 | 5/6, 5/7, 6/8, 1/5, 6/4, 5/6 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.008 | 1/1, 1/3, 4/3, 2/1, 5/3, 3/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.025 | 3/6, 2/5, 3/6, 5/6, 3/6, 3/6 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.000 | 2/3, 2/1, 2/1, 2/2, 2/2, 1/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.017 | 3/1, 2/0, 2/0, 2/1, 2/3, 2/0 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.008 | 3/4, 3/4, 3/4, 3/5, 2/6, 3/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.008 | 2/4, 4/5, 3/7, 3/4, 5/5, 3/4 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.017 | 4/0, 2/0, 4/2, 3/1, 3/2, 4/0 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.013 | 5/7, 6/7, 7/6, 6/6, 4/7, 5/7 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.000 | 2/3, 3/3, 3/2, 3/3, 2/2, 4/3 |
| best_original[comb_lr[full+act]] − original_vote | +0.017 | 3/1, 2/0, 3/0, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | -0.017 | 1/3, 2/2, 2/3, 1/3, 1/3, 1/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.042 | 2/7, 2/5, 3/7, 2/7, 2/7, 2/7 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/1, 1/0, 0/0, 1/1, 1/1, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 1/0, 1/0, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/1, 0/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.000 | 6/6, 6/6, 4/6, 6/6, 5/5, 6/5 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.008 | 2/0, 1/0, 0/1, 1/0, 2/1, 3/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.017 | 3/2, 4/2, 3/1, 3/1, 3/1, 3/2 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/1, 0/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.025 | 2/5, 2/5, 2/5, 2/5, 1/5, 2/5 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.013 | 6/5, 5/3, 2/5, 5/5, 6/3, 7/4 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.004 | 2/3, 3/2, 2/2, 0/2, 3/1, 2/1 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.008 | 3/1, 2/2, 3/3, 3/2, 2/1, 3/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.004 | 7/6, 6/6, 7/7, 6/5, 7/5, 5/5 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.017 | 2/0, 0/0, 1/0, 2/0, 2/0, 2/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 4/2, 4/1, 4/2, 3/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.000 | 0/0, 0/0, 0/0, 0/1, 0/0, 0/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.025 | 2/5, 2/5, 2/5, 2/6, 2/5, 2/5 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.017 | 6/4, 5/3, 4/5, 5/4, 8/3, 7/5 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.004 | 1/1, 3/1, 2/2, 0/2, 4/1, 3/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.008 | 3/1, 2/2, 3/3, 3/3, 3/1, 3/1 |
| gate_tuned[act] − original_vote | -0.008 | 1/3, 0/1, 2/2, 1/2, 0/2, 4/2 |

Origin of best-of-six picks (partition 0): act: 15 from revisions, 0 new options (0 correct); out_lr[full]: 38 from revisions, 4 new options (1 correct); out_hgb[full]: 48 from revisions, 8 new options (1 correct); out_lr[of+ot]: 24 from revisions, 2 new options (0 correct); comb_lr[full+act]: 38 from revisions, 4 new options (1 correct); eq[act+out_lr_full]: 26 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 15 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 0, helpful_blocked 35, harmful_accepted 3, harmful_blocked 76, wrong_to_wrong_accepted 9, wrong_to_wrong_blocked 58; gate_tuned[act]: helpful_accepted 4, helpful_blocked 31, harmful_accepted 10, harmful_blocked 69, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 52; gate0[out_lr[full]]: helpful_accepted 10, helpful_blocked 25, harmful_accepted 8, harmful_blocked 71, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 52; gate0[out_hgb[full]]: helpful_accepted 10, helpful_blocked 25, harmful_accepted 11, harmful_blocked 68, wrong_to_wrong_accepted 18, wrong_to_wrong_blocked 49; gate0[out_lr[of+ot]]: helpful_accepted 12, helpful_blocked 23, harmful_accepted 4, harmful_blocked 75, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 56; gate0[comb_lr[full+act]]: helpful_accepted 9, helpful_blocked 26, harmful_accepted 8, harmful_blocked 71, wrong_to_wrong_accepted 14, wrong_to_wrong_blocked 53; gate0[eq[act+out_lr_full]]: helpful_accepted 1, helpful_blocked 34, harmful_accepted 2, harmful_blocked 77, wrong_to_wrong_accepted 8, wrong_to_wrong_blocked 59; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 1, helpful_blocked 34, harmful_accepted 1, harmful_blocked 78, wrong_to_wrong_accepted 6, wrong_to_wrong_blocked 61

Calibration of revised-state probabilities (median over partitions): act Brier 0.188 (constant 0.254), ECE 0.062; act_rout Brier 0.253 (constant 0.254), ECE 0.088; comb_lr[full+act] Brier 0.177 (constant 0.254), ECE 0.074; eq[act+out_lr_full] Brier 0.167 (constant 0.254), ECE 0.061; eq[act+out_lr_of+ot] Brier 0.169 (constant 0.254), ECE 0.061; out_hgb[full] Brier 0.195 (constant 0.254), ECE 0.053; out_lr[full] Brier 0.178 (constant 0.254), ECE 0.071; out_lr[of+ot] Brier 0.177 (constant 0.254), ECE 0.049

### misleading_one|eligible (120 questions; absent states 0)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.652 (0.63–0.68) | 0.829 (0.82–0.85) | 0.637 (0.58–0.66) | 0.857 (0.85–0.88) | 0.752 (0.73–0.76) | 0.831 (0.76–0.84) |
| act_rout | 0.559 (0.52–0.61) | 0.555 (0.54–0.59) | 0.637 (0.55–0.71) | 0.601 (0.57–0.68) | 0.636 (0.59–0.66) | 0.760 (0.71–0.80) |
| comb_lr[full+act] | 0.768 (0.73–0.78) | 0.852 (0.84–0.87) | 0.653 (0.60–0.69) | 0.780 (0.77–0.82) | 0.763 (0.74–0.79) | 0.741 (0.70–0.78) |
| eq[act+out_lr_full] | 0.754 (0.73–0.78) | 0.869 (0.86–0.88) | 0.653 (0.63–0.66) | 0.845 (0.83–0.86) | 0.768 (0.76–0.78) | 0.810 (0.77–0.83) |
| eq[act+out_lr_of+ot] | 0.732 (0.71–0.78) | 0.859 (0.85–0.88) | 0.637 (0.58–0.66) | 0.863 (0.83–0.88) | 0.767 (0.75–0.78) | 0.818 (0.78–0.83) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.823 (0.79–0.85) | 0.613 (0.58–0.73) | 0.798 (0.73–0.83) | 0.756 (0.72–0.78) | 0.759 (0.70–0.82) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.853 (0.83–0.86) | 0.653 (0.60–0.71) | 0.774 (0.75–0.80) | 0.748 (0.74–0.78) | 0.740 (0.69–0.77) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.843 (0.83–0.86) | 0.581 (0.55–0.63) | 0.792 (0.76–0.82) | 0.753 (0.74–0.77) | 0.729 (0.70–0.76) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6429 pairs; panel_pre 62 pairs / 31 questions; panel_post 84 pairs / 42 questions; six 375 pairs / 56 questions; revision_auc 331 pairs / 56 questions; helpful 30, harmful 59, wrong-to-wrong 75

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.116 (0/4) | -0.017 (0/0) | -0.032 (0/0) | +0.083 (2/0) | +0.004 (0/0) | +0.082 (0/0) |
| act - out_hgb[full] | -0.049 (0/0) | +0.012 (0/0) | +0.024 (0/0) | +0.065 (1/0) | -0.008 (0/0) | +0.045 (0/0) |
| comb_lr[full+act] - out_lr[full] | -0.001 (0/0) | +0.003 (0/0) | +0.000 (0/0) | +0.006 (0/0) | +0.011 (1/0) | +0.004 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.006 (0/0) | +0.021 (1/0) | +0.000 (0/0) | +0.071 (2/0) | +0.020 (0/0) | +0.075 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | -0.007 (0/0) | +0.018 (0/0) | +0.040 (2/0) | +0.071 (3/0) | +0.012 (0/0) | +0.074 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 73 | 69 | 70 | 71 | 73 | 72 |
| best_original[comb_lr[full+act]] | 71 | 71 | 72 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 70 | 71 | 71 | 71 | 71 | 70 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 72 | 71 | 71 | 72 | 70 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act] | 74 | 72 | 71 | 73 | 74 | 74 |
| best_six[comb_lr[full+act]] | 70 | 70 | 73 | 73 | 70 | 69 |
| best_six[eq[act+out_lr_full]] | 71 | 72 | 71 | 71 | 71 | 71 |
| best_six[eq[act+out_lr_of+ot]] | 71 | 72 | 73 | 72 | 72 | 72 |
| best_six[out_hgb[full]] | 68 | 70 | 68 | 69 | 70 | 70 |
| best_six[out_lr[full]] | 71 | 69 | 72 | 71 | 70 | 69 |
| best_six[out_lr[of+ot]] | 72 | 71 | 71 | 73 | 72 | 72 |
| gate0[act] | 69 | 71 | 69 | 70 | 70 | 70 |
| gate0[comb_lr[full+act]] | 68 | 69 | 70 | 72 | 69 | 68 |
| gate0[eq[act+out_lr_full]] | 69 | 69 | 68 | 70 | 68 | 69 |
| gate0[eq[act+out_lr_of+ot]] | 70 | 69 | 70 | 69 | 70 | 70 |
| gate0[out_hgb[full]] | 68 | 70 | 70 | 71 | 69 | 71 |
| gate0[out_lr[full]] | 68 | 68 | 69 | 71 | 68 | 68 |
| gate0[out_lr[of+ot]] | 69 | 70 | 69 | 70 | 69 | 69 |
| gate_tuned[act] | 70 | 71 | 72 | 74 | 72 | 72 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 72 | 72 | 72 | 72 | 72 | 72 |
| revised_vote | 66 | 66 | 66 | 66 | 66 | 66 |

Random tie-break expectation: original 70.67, revised 63.67, pooled six 72.00. Questions with a correct round-0 answer 82, with a correct revised answer 77, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.013 | 4/2, 2/2, 2/3, 2/1, 3/1, 3/1 |
| best_six[act] − best_original[act] | +0.013 | 1/0, 3/0, 1/0, 2/0, 1/0, 2/0 |
| best_original[act] − original_vote | +0.021 | 7/3, 3/3, 5/4, 5/3, 6/2, 6/3 |
| gate0[act] − original_vote | +0.008 | 2/2, 3/1, 2/2, 2/1, 2/1, 2/1 |
| gate0[act] − pooled6_vote | -0.017 | 0/3, 1/2, 0/3, 0/2, 0/2, 0/2 |
| best_six[act] − best_six[out_lr[full]] | +0.025 | 6/3, 5/2, 4/5, 5/3, 6/2, 7/2 |
| best_original[act] − best_original[out_lr[full]] | +0.004 | 6/4, 3/4, 5/6, 3/5, 7/3, 7/4 |
| gate0[act] − gate0[out_lr[full]] | +0.013 | 3/2, 5/2, 2/2, 2/3, 3/1, 3/1 |
| best_six[out_lr[full]] − pooled6_vote | -0.013 | 3/4, 2/5, 2/2, 3/4, 1/3, 2/5 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 2/2, 1/2, 2/1, 0/2, 1/0, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | -0.008 | 1/2, 1/2, 1/1, 3/1, 1/2, 2/3 |
| gate0[out_lr[full]] − pooled6_vote | -0.033 | 0/4, 1/5, 0/3, 2/3, 0/4, 1/5 |
| best_six[out_hgb[full]] − pooled6_vote | -0.021 | 2/6, 3/5, 3/7, 3/6, 3/5, 2/4 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.008 | 0/2, 3/4, 1/2, 0/3, 1/1, 1/1 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.008 | 2/3, 4/3, 3/2, 3/1, 2/2, 3/1 |
| gate0[out_hgb[full]] − pooled6_vote | -0.017 | 1/5, 3/5, 2/4, 2/3, 1/4, 1/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.008 | 1/4, 5/4, 2/6, 1/3, 2/2, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.008 | 1/1, 3/1, 2/1, 1/1, 1/0, 3/0 |
| best_six[out_lr[of+ot]] − pooled6_vote | +0.000 | 2/2, 1/2, 2/3, 3/2, 2/2, 2/2 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.025 | 2/0, 3/0, 3/0, 2/0, 3/0, 3/0 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 2/2, 2/1, 2/2, 3/2, 2/2, 2/2 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.025 | 1/4, 1/3, 1/4, 2/4, 1/4, 1/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | +0.017 | 3/2, 4/2, 1/2, 3/1, 3/1, 5/2 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.008 | 3/2, 3/1, 2/2, 1/2, 2/1, 3/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.017 | 2/4, 2/4, 2/1, 3/2, 1/3, 2/5 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.004 | 1/2, 1/2, 1/0, 1/0, 1/0, 2/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.017 | 3/1, 2/0, 3/0, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.000 | 1/2, 2/2, 2/1, 3/0, 1/1, 2/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.025 | 0/4, 1/4, 1/3, 2/2, 0/3, 1/5 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.004 | 0/1, 1/0, 1/0, 2/0, 0/0, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 1/0, 1/0, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.008 | 0/0, 1/0, 1/0, 1/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | -0.008 | 2/3, 2/2, 1/2, 2/3, 2/3, 2/3 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.004 | 1/0, 1/0, 0/0, 0/0, 0/0, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.017 | 3/2, 4/2, 3/1, 3/1, 3/1, 3/2 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.000 | 1/1, 2/2, 1/2, 1/0, 1/2, 1/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.025 | 0/3, 0/3, 0/4, 0/2, 0/4, 0/3 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.004 | 3/3, 4/1, 2/3, 2/2, 3/2, 4/2 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.004 | 2/3, 3/2, 2/2, 0/2, 3/1, 2/1 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.004 | 2/1, 3/2, 0/1, 1/2, 1/1, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 3/4, 2/2, 3/2, 2/2, 3/3, 2/2 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.004 | 0/0, 0/0, 2/0, 1/0, 0/0, 2/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 4/2, 4/1, 4/2, 3/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.008 | 2/1, 1/1, 2/1, 1/1, 2/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.017 | 1/3, 0/3, 1/3, 0/3, 1/3, 1/3 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.013 | 2/2, 4/1, 3/2, 3/2, 4/2, 5/2 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.004 | 1/1, 3/1, 2/2, 0/2, 4/1, 3/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.013 | 3/1, 3/2, 2/1, 1/3, 3/1, 3/1 |
| gate_tuned[act] − original_vote | +0.025 | 3/2, 3/1, 4/1, 6/1, 4/1, 5/2 |

Origin of best-of-six picks (partition 0): act: 22 from revisions, 2 new options (0 correct); out_lr[full]: 53 from revisions, 4 new options (0 correct); out_hgb[full]: 46 from revisions, 2 new options (0 correct); out_lr[of+ot]: 41 from revisions, 3 new options (0 correct); comb_lr[full+act]: 50 from revisions, 4 new options (0 correct); eq[act+out_lr_full]: 30 from revisions, 2 new options (0 correct); eq[act+out_lr_of+ot]: 28 from revisions, 1 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 6, helpful_blocked 24, harmful_accepted 4, harmful_blocked 55, wrong_to_wrong_accepted 10, wrong_to_wrong_blocked 65; gate_tuned[act]: helpful_accepted 14, helpful_blocked 16, harmful_accepted 11, harmful_blocked 48, wrong_to_wrong_accepted 24, wrong_to_wrong_blocked 51; gate0[out_lr[full]]: helpful_accepted 15, helpful_blocked 15, harmful_accepted 10, harmful_blocked 49, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 60; gate0[out_hgb[full]]: helpful_accepted 13, helpful_blocked 17, harmful_accepted 10, harmful_blocked 49, wrong_to_wrong_accepted 19, wrong_to_wrong_blocked 56; gate0[out_lr[of+ot]]: helpful_accepted 14, helpful_blocked 16, harmful_accepted 7, harmful_blocked 52, wrong_to_wrong_accepted 17, wrong_to_wrong_blocked 58; gate0[comb_lr[full+act]]: helpful_accepted 14, helpful_blocked 16, harmful_accepted 9, harmful_blocked 50, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 60; gate0[eq[act+out_lr_full]]: helpful_accepted 10, helpful_blocked 20, harmful_accepted 5, harmful_blocked 54, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 63; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 9, helpful_blocked 21, harmful_accepted 6, harmful_blocked 53, wrong_to_wrong_accepted 10, wrong_to_wrong_blocked 65

Calibration of revised-state probabilities (median over partitions): act Brier 0.181 (constant 0.252), ECE 0.086; act_rout Brier 0.252 (constant 0.252), ECE 0.045; comb_lr[full+act] Brier 0.165 (constant 0.252), ECE 0.063; eq[act+out_lr_full] Brier 0.158 (constant 0.252), ECE 0.073; eq[act+out_lr_of+ot] Brier 0.163 (constant 0.252), ECE 0.092; out_hgb[full] Brier 0.185 (constant 0.252), ECE 0.064; out_lr[full] Brier 0.166 (constant 0.252), ECE 0.063; out_lr[of+ot] Brier 0.170 (constant 0.252), ECE 0.077

Alignment: held-out z SD (1 = reference scale), median (range) over folds × partitions: act 0.98 (0.80–1.09); out_lr[of+ot] 0.92 (0.79–1.23); out_lr[full] 0.98 (0.83–1.26); out_hgb[full] 0.92 (0.80–1.02); comb_lr[full+act] 1.00 (0.85–1.24); eq[act+out_lr_full] 1.83 (1.52–2.17); eq[act+out_lr_of+ot] 1.76 (1.49–1.97); act_rout 0.90 (0.71–1.13)

Orthogonality, act: held-out R² from all output features 0.82 (0.57–0.86); largest |r| with one output feature 0.68 (0.56–0.79).
Orthogonality, act_rout: held-out R² from all output features -0.22 (-1.93–-0.03); largest |r| with one output feature 0.28 (0.19–0.59).

## R2: `before_stated_answer:letter:pca16:21-33` (layers [21, 24, 27, 30, 33])

### debate|eligible (109 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.672 (0.65–0.72) | 0.763 (0.75–0.78) | 0.670 (0.64–0.74) | 0.726 (0.69–0.76) | 0.672 (0.64–0.68) | 0.782 (0.70–0.82) |
| comb_lr[full+act] | 0.769 (0.74–0.79) | 0.817 (0.80–0.82) | 0.640 (0.60–0.72) | 0.571 (0.50–0.62) | 0.643 (0.60–0.70) | 0.631 (0.60–0.70) |
| eq[act+out_lr_full] | 0.765 (0.75–0.78) | 0.812 (0.80–0.83) | 0.690 (0.64–0.76) | 0.690 (0.64–0.76) | 0.682 (0.65–0.72) | 0.711 (0.64–0.74) |
| eq[act+out_lr_of+ot] | 0.747 (0.72–0.78) | 0.792 (0.78–0.81) | 0.670 (0.60–0.72) | 0.595 (0.50–0.67) | 0.648 (0.60–0.66) | 0.706 (0.62–0.72) |
| out_hgb[full] | 0.688 (0.64–0.74) | 0.754 (0.72–0.78) | 0.600 (0.56–0.68) | 0.595 (0.57–0.62) | 0.609 (0.56–0.65) | 0.616 (0.52–0.66) |
| out_lr[full] | 0.764 (0.73–0.78) | 0.810 (0.80–0.82) | 0.650 (0.60–0.74) | 0.548 (0.48–0.60) | 0.636 (0.61–0.69) | 0.619 (0.60–0.70) |
| out_lr[of+ot] | 0.744 (0.71–0.79) | 0.775 (0.76–0.79) | 0.600 (0.56–0.66) | 0.429 (0.40–0.50) | 0.556 (0.53–0.60) | 0.563 (0.48–0.60) |

Counts (partition 0): state_auc_pre 5052 pairs; state_auc_post 5129 pairs; panel_pre 50 pairs / 25 questions; panel_post 42 pairs / 21 questions; six 206 pairs / 27 questions; revision_auc 166 pairs / 26 questions; helpful 27, harmful 29, wrong-to-wrong 38

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.094 (0/1) | -0.039 (0/0) | +0.040 (0/0) | +0.179 (0/0) | +0.015 (0/0) | +0.123 (0/0) |
| act - out_hgb[full] | -0.027 (0/0) | +0.012 (0/0) | +0.080 (0/0) | +0.131 (0/0) | +0.063 (0/0) | +0.168 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.006 (0/0) | +0.003 (0/0) | +0.000 (0/0) | +0.024 (0/0) | +0.000 (0/0) | +0.006 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | +0.005 (0/0) | +0.080 (1/0) | +0.143 (4/0) | +0.029 (0/0) | +0.076 (1/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.009 (0/0) | +0.021 (0/0) | +0.040 (1/0) | +0.155 (1/0) | +0.070 (1/0) | +0.134 (2/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 68 | 67 | 67 | 68 | 67 | 66 |
| best_original[comb_lr[full+act]] | 66 | 65 | 65 | 67 | 64 | 64 |
| best_original[eq[act+out_lr_full]] | 68 | 67 | 65 | 67 | 66 | 66 |
| best_original[eq[act+out_lr_of+ot]] | 67 | 65 | 64 | 66 | 68 | 66 |
| best_original[out_hgb[full]] | 65 | 65 | 64 | 67 | 65 | 65 |
| best_original[out_lr[full]] | 66 | 65 | 66 | 68 | 64 | 64 |
| best_original[out_lr[of+ot]] | 66 | 64 | 64 | 67 | 65 | 65 |
| best_six[act] | 67 | 66 | 67 | 68 | 66 | 67 |
| best_six[comb_lr[full+act]] | 66 | 66 | 66 | 67 | 65 | 64 |
| best_six[eq[act+out_lr_full]] | 68 | 66 | 65 | 67 | 66 | 67 |
| best_six[eq[act+out_lr_of+ot]] | 68 | 66 | 64 | 67 | 67 | 67 |
| best_six[out_hgb[full]] | 65 | 66 | 63 | 66 | 63 | 65 |
| best_six[out_lr[full]] | 66 | 66 | 67 | 67 | 66 | 64 |
| best_six[out_lr[of+ot]] | 65 | 64 | 65 | 65 | 64 | 66 |
| gate0[act] | 64 | 65 | 64 | 65 | 65 | 64 |
| gate0[comb_lr[full+act]] | 67 | 66 | 67 | 67 | 65 | 64 |
| gate0[eq[act+out_lr_full]] | 65 | 65 | 65 | 67 | 64 | 66 |
| gate0[eq[act+out_lr_of+ot]] | 67 | 66 | 67 | 66 | 66 | 67 |
| gate0[out_hgb[full]] | 66 | 66 | 65 | 67 | 64 | 66 |
| gate0[out_lr[full]] | 67 | 66 | 67 | 67 | 66 | 65 |
| gate0[out_lr[of+ot]] | 65 | 64 | 65 | 65 | 64 | 65 |
| gate_tuned[act] | 61 | 66 | 64 | 66 | 66 | 67 |
| original_vote | 65 | 65 | 65 | 65 | 65 | 65 |
| pooled6_vote | 66 | 66 | 66 | 66 | 66 | 66 |
| revised_vote | 62 | 62 | 62 | 62 | 62 | 62 |

Random tie-break expectation: original 66.33, revised 61.67, pooled six 64.83. Questions with a correct round-0 answer 75, with a correct revised answer 74, correct only after revision 1.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.009 | 4/3, 2/2, 3/2, 3/1, 3/3, 4/3 |
| best_six[act] − best_original[act] | -0.005 | 0/1, 0/1, 0/0, 0/0, 0/1, 1/0 |
| best_original[act] − original_vote | +0.018 | 6/3, 4/2, 4/2, 4/1, 4/2, 5/4 |
| gate0[act] − original_vote | -0.005 | 0/1, 1/1, 0/1, 1/1, 1/1, 0/1 |
| gate0[act] − pooled6_vote | -0.014 | 0/2, 1/2, 0/2, 1/2, 1/2, 0/2 |
| best_six[act] − best_six[out_lr[full]] | +0.005 | 5/4, 3/3, 4/4, 4/3, 5/5, 7/4 |
| best_original[act] − best_original[out_lr[full]] | +0.018 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act] − gate0[out_lr[full]] | -0.014 | 1/4, 2/3, 1/4, 2/4, 2/3, 2/3 |
| best_six[out_lr[full]] − pooled6_vote | +0.000 | 2/2, 1/1, 2/1, 2/1, 2/2, 1/3 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.005 | 1/1, 1/0, 1/0, 0/1, 2/0, 1/1 |
| best_original[out_lr[full]] − original_vote | +0.005 | 2/1, 1/1, 2/1, 3/0, 1/2, 1/2 |
| gate0[out_lr[full]] − original_vote | +0.014 | 3/1, 2/1, 3/1, 3/1, 2/1, 2/2 |
| gate0[out_lr[full]] − pooled6_vote | +0.005 | 2/1, 1/1, 2/1, 2/1, 1/1, 1/2 |
| best_six[out_hgb[full]] − pooled6_vote | -0.009 | 1/2, 2/2, 2/5, 2/2, 0/3, 1/2 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.005 | 1/1, 2/1, 0/1, 0/1, 0/2, 0/0 |
| best_original[out_hgb[full]] − original_vote | +0.000 | 3/3, 2/2, 3/4, 3/1, 2/2, 2/2 |
| gate0[out_hgb[full]] − original_vote | +0.009 | 2/1, 3/2, 3/3, 3/1, 1/2, 2/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 1/1, 2/2, 2/3, 2/1, 0/2, 1/1 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.009 | 0/1, 1/1, 0/4, 0/1, 0/3, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | -0.005 | 2/3, 2/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.005 | 0/1, 1/1, 0/2, 0/0, 0/2, 2/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.009 | 2/3, 2/4, 2/3, 2/3, 2/4, 2/2 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.005 | 1/2, 2/2, 1/0, 0/2, 1/2, 2/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 2/1, 1/2, 2/3, 3/1, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 2/2, 2/3, 2/2, 2/2, 2/3, 2/2 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.009 | 1/2, 1/3, 1/2, 1/2, 1/3, 1/2 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.018 | 0/1, 1/3, 1/3, 0/2, 0/2, 3/1 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.005 | 0/0, 1/2, 1/3, 0/1, 2/1, 1/0 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.018 | 0/2, 1/3, 1/3, 1/3, 0/2, 2/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/2, 1/1, 2/2, 2/1, 1/2, 1/3 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.005 | 1/1, 1/0, 1/0, 1/1, 1/0, 1/1 |
| best_original[comb_lr[full+act]] − original_vote | +0.000 | 2/1, 1/1, 2/2, 2/0, 1/2, 1/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.014 | 3/1, 2/1, 3/1, 3/1, 1/1, 1/2 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.005 | 2/1, 1/1, 2/1, 2/1, 0/1, 0/2 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/0, 0/1, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/1, 0/1 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.005 | 2/0, 1/1, 1/2, 1/0, 1/1, 1/0 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.000 | 0/0, 0/1, 0/0, 0/0, 1/1, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.014 | 3/0, 2/0, 1/1, 2/0, 1/0, 2/1 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.000 | 1/1, 1/1, 1/1, 2/0, 1/2, 1/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.009 | 0/1, 0/1, 0/1, 1/0, 0/2, 0/0 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.000 | 2/0, 0/0, 0/2, 1/1, 2/2, 4/1 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.018 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | -0.014 | 0/2, 0/1, 0/2, 1/1, 0/2, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.009 | 3/1, 2/2, 2/4, 2/1, 3/2, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.009 | 1/0, 1/0, 0/0, 1/0, 0/1, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.009 | 3/1, 2/2, 2/3, 2/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.014 | 2/0, 1/0, 2/0, 1/0, 2/1, 2/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.005 | 1/0, 0/0, 1/0, 0/0, 1/1, 1/0 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.005 | 3/1, 2/2, 1/4, 2/2, 2/1, 5/2 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.005 | 2/1, 2/2, 1/3, 0/2, 5/1, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.000 | 1/1, 1/1, 1/1, 1/2, 0/0, 2/0 |
| gate_tuned[act] − original_vote | +0.009 | 0/4, 2/1, 0/1, 2/1, 3/2, 4/2 |

Origin of best-of-six picks (partition 0): act: 73 from revisions, 0 new options (0 correct); out_lr[full]: 74 from revisions, 0 new options (0 correct); out_hgb[full]: 78 from revisions, 0 new options (0 correct); out_lr[of+ot]: 56 from revisions, 0 new options (0 correct); comb_lr[full+act]: 73 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 68 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 70 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 5, helpful_blocked 22, harmful_accepted 1, harmful_blocked 28, wrong_to_wrong_accepted 1, wrong_to_wrong_blocked 37; gate_tuned[act]: helpful_accepted 15, helpful_blocked 12, harmful_accepted 13, harmful_blocked 16, wrong_to_wrong_accepted 16, wrong_to_wrong_blocked 22; gate0[out_lr[full]]: helpful_accepted 17, helpful_blocked 10, harmful_accepted 7, harmful_blocked 22, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 26; gate0[out_hgb[full]]: helpful_accepted 13, helpful_blocked 14, harmful_accepted 6, harmful_blocked 23, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 23; gate0[out_lr[of+ot]]: helpful_accepted 13, helpful_blocked 14, harmful_accepted 9, harmful_blocked 20, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 23; gate0[comb_lr[full+act]]: helpful_accepted 17, helpful_blocked 10, harmful_accepted 6, harmful_blocked 23, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 27; gate0[eq[act+out_lr_full]]: helpful_accepted 9, helpful_blocked 18, harmful_accepted 2, harmful_blocked 27, wrong_to_wrong_accepted 6, wrong_to_wrong_blocked 32; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 8, helpful_blocked 19, harmful_accepted 2, harmful_blocked 27, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 33

Calibration of revised-state probabilities (median over partitions): act Brier 0.199 (constant 0.247), ECE 0.086; comb_lr[full+act] Brier 0.185 (constant 0.247), ECE 0.082; eq[act+out_lr_full] Brier 0.184 (constant 0.247), ECE 0.095; eq[act+out_lr_of+ot] Brier 0.190 (constant 0.247), ECE 0.092; out_hgb[full] Brier 0.207 (constant 0.247), ECE 0.064; out_lr[full] Brier 0.187 (constant 0.247), ECE 0.077; out_lr[of+ot] Brier 0.202 (constant 0.247), ECE 0.088

### debate|all_fallback (120 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb | 0.681 (0.67–0.72) | 0.759 (0.74–0.78) | 0.653 (0.63–0.71) | 0.677 (0.65–0.73) | 0.651 (0.62–0.67) | 0.785 (0.72–0.81) |
| comb_lr[full+act] | 0.773 (0.74–0.79) | 0.818 (0.79–0.82) | 0.637 (0.61–0.69) | 0.583 (0.52–0.62) | 0.647 (0.61–0.69) | 0.654 (0.62–0.72) |
| eq[act+out_lr_full] | 0.770 (0.75–0.78) | 0.808 (0.79–0.82) | 0.685 (0.65–0.74) | 0.688 (0.65–0.75) | 0.681 (0.65–0.71) | 0.727 (0.67–0.77) |
| eq[act+out_lr_of+ot] | 0.753 (0.72–0.78) | 0.790 (0.78–0.81) | 0.645 (0.60–0.68) | 0.604 (0.52–0.67) | 0.645 (0.62–0.65) | 0.705 (0.65–0.73) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.764 (0.73–0.79) | 0.613 (0.58–0.73) | 0.625 (0.60–0.65) | 0.633 (0.59–0.69) | 0.660 (0.59–0.70) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.813 (0.79–0.82) | 0.653 (0.60–0.71) | 0.573 (0.50–0.62) | 0.645 (0.63–0.69) | 0.646 (0.62–0.71) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.782 (0.77–0.80) | 0.581 (0.55–0.63) | 0.469 (0.46–0.54) | 0.560 (0.55–0.59) | 0.575 (0.52–0.62) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6234 pairs; panel_pre 62 pairs / 31 questions; panel_post 48 pairs / 24 questions; six 252 pairs / 33 questions; revision_auc 221 pairs / 31 questions; helpful 32, harmful 32, wrong-to-wrong 46

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb - out_lr[full] | -0.089 (0/1) | -0.047 (0/0) | +0.016 (0/0) | +0.115 (0/0) | -0.014 (0/0) | +0.096 (0/0) |
| act_fb - out_hgb[full] | -0.024 (0/0) | -0.006 (0/0) | +0.040 (0/0) | +0.062 (0/0) | +0.020 (0/0) | +0.129 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.007 (0/0) | +0.002 (0/0) | +0.000 (0/0) | +0.010 (0/0) | -0.004 (0/0) | +0.016 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | -0.001 (0/0) | +0.056 (0/0) | +0.115 (3/0) | +0.020 (0/0) | +0.071 (1/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.011 (0/0) | +0.012 (0/0) | +0.048 (1/0) | +0.125 (1/0) | +0.063 (1/0) | +0.126 (3/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act_fb] | 73 | 72 | 72 | 73 | 72 | 71 |
| best_original[comb_lr[full+act]] | 71 | 70 | 69 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 73 | 72 | 70 | 72 | 71 | 71 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 69 | 69 | 70 | 72 | 71 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act_fb] | 72 | 71 | 72 | 73 | 71 | 72 |
| best_six[comb_lr[full+act]] | 71 | 71 | 70 | 72 | 70 | 69 |
| best_six[eq[act+out_lr_full]] | 73 | 71 | 70 | 72 | 71 | 72 |
| best_six[eq[act+out_lr_of+ot]] | 72 | 70 | 68 | 71 | 71 | 72 |
| best_six[out_hgb[full]] | 70 | 72 | 68 | 71 | 68 | 70 |
| best_six[out_lr[full]] | 71 | 71 | 72 | 72 | 71 | 69 |
| best_six[out_lr[of+ot]] | 69 | 68 | 69 | 69 | 68 | 70 |
| gate0[act_fb] | 68 | 69 | 68 | 69 | 69 | 67 |
| gate0[comb_lr[full+act]] | 72 | 71 | 72 | 71 | 69 | 68 |
| gate0[eq[act+out_lr_full]] | 70 | 70 | 70 | 71 | 68 | 70 |
| gate0[eq[act+out_lr_of+ot]] | 71 | 71 | 72 | 70 | 70 | 71 |
| gate0[out_hgb[full]] | 71 | 71 | 70 | 72 | 69 | 71 |
| gate0[out_lr[full]] | 72 | 71 | 72 | 71 | 70 | 69 |
| gate0[out_lr[of+ot]] | 69 | 68 | 69 | 69 | 68 | 69 |
| gate_tuned[act_fb] | 65 | 71 | 68 | 70 | 70 | 72 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 71 | 71 | 71 | 71 | 71 | 71 |
| revised_vote | 67 | 67 | 67 | 67 | 67 | 67 |

Random tie-break expectation: original 70.67, revised 67.00, pooled six 70.17. Questions with a correct round-0 answer 82, with a correct revised answer 80, correct only after revision 1.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[out_lr[full]] − pooled6_vote | +0.000 | 3/3, 2/2, 3/2, 3/2, 3/3, 2/4 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.004 | 1/1, 1/0, 1/0, 0/1, 2/0, 1/1 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | +0.017 | 5/2, 4/2, 5/2, 4/2, 3/2, 3/3 |
| gate0[out_lr[full]] − pooled6_vote | +0.000 | 2/1, 1/1, 2/1, 2/2, 1/2, 1/3 |
| best_six[out_hgb[full]] − pooled6_vote | -0.008 | 2/3, 3/2, 3/6, 3/3, 1/4, 2/3 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.004 | 1/1, 2/1, 0/1, 0/1, 0/2, 0/0 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.017 | 3/1, 4/2, 4/3, 4/1, 2/2, 3/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 2/2, 3/3, 3/4, 3/2, 1/3, 2/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.008 | 0/1, 2/1, 0/4, 0/1, 0/3, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.004 | 1/2, 2/2, 1/3, 1/0, 1/2, 3/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.017 | 2/4, 2/5, 2/4, 2/4, 2/5, 2/3 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.004 | 1/2, 2/2, 1/0, 0/2, 1/2, 2/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 3/3, 3/4, 3/3, 3/3, 3/4, 3/3 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.017 | 1/3, 1/4, 1/3, 1/3, 1/4, 1/3 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.025 | 0/2, 1/4, 1/4, 0/3, 0/3, 3/2 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.021 | 0/3, 1/4, 1/4, 1/3, 0/2, 2/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.004 | 3/3, 2/2, 2/3, 3/2, 2/3, 2/4 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.004 | 1/1, 1/0, 1/0, 1/1, 1/0, 1/1 |
| best_original[comb_lr[full+act]] − original_vote | +0.004 | 3/1, 2/1, 3/3, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.017 | 5/2, 4/2, 5/2, 4/2, 2/2, 2/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/1, 1/1, 2/1, 2/2, 0/2, 0/3 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/0, 0/1, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/1, 0/1 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.004 | 3/1, 2/2, 2/3, 2/1, 2/2, 2/1 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.000 | 0/0, 0/1, 0/0, 0/0, 1/1, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.021 | 4/0, 3/0, 2/1, 3/0, 2/0, 3/1 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.008 | 3/2, 3/2, 3/2, 3/1, 2/3, 2/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.008 | 0/1, 0/1, 0/1, 1/1, 0/3, 0/1 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.000 | 2/0, 0/0, 0/2, 1/1, 2/2, 4/1 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.017 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | -0.013 | 0/2, 0/1, 0/2, 1/1, 0/2, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 3/2, 2/3, 2/5, 2/2, 3/3, 3/2 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.008 | 1/0, 1/0, 0/1, 1/0, 0/1, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.013 | 4/2, 3/3, 3/3, 3/2, 5/2, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 3/1, 3/1, 4/1, 2/1, 3/2, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 1/1, 0/0, 1/0, 0/1, 1/2, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | -0.004 | 3/2, 2/3, 1/5, 2/3, 2/2, 5/2 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | -0.004 | 2/2, 2/3, 1/3, 0/3, 5/2, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.000 | 1/2, 1/1, 1/1, 1/2, 0/0, 2/0 |
| best_six[act_fb] − pooled6_vote | +0.008 | 5/4, 3/3, 4/3, 4/2, 4/4, 5/4 |
| best_six[act_fb] − best_original[act_fb] | -0.004 | 0/1, 0/1, 0/0, 0/0, 0/1, 1/0 |
| best_original[act_fb] − original_vote | +0.025 | 7/3, 5/2, 5/2, 5/1, 5/2, 6/4 |
| gate0[act_fb] − original_vote | -0.004 | 2/3, 3/3, 2/3, 3/3, 3/3, 1/3 |
| gate0[act_fb] − pooled6_vote | -0.021 | 0/3, 1/3, 0/3, 1/3, 1/3, 0/4 |
| best_six[act_fb] − best_six[out_lr[full]] | +0.004 | 5/4, 3/3, 4/4, 4/3, 5/5, 7/4 |
| best_original[act_fb] − best_original[out_lr[full]] | +0.017 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act_fb] − gate0[out_lr[full]] | -0.017 | 1/5, 2/4, 1/5, 3/5, 3/4, 2/4 |
| gate_tuned[act_fb] − original_vote | +0.008 | 1/5, 4/2, 2/3, 4/3, 4/3, 6/3 |

Origin of best-of-six picks (partition 0): out_lr[full]: 76 from revisions, 0 new options (0 correct); out_hgb[full]: 80 from revisions, 0 new options (0 correct); out_lr[of+ot]: 59 from revisions, 0 new options (0 correct); comb_lr[full+act]: 75 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 69 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 71 from revisions, 0 new options (0 correct); act_fb: 74 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[out_lr[full]]: helpful_accepted 22, helpful_blocked 10, harmful_accepted 8, harmful_blocked 24, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 33; gate0[out_hgb[full]]: helpful_accepted 16, helpful_blocked 16, harmful_accepted 6, harmful_blocked 26, wrong_to_wrong_accepted 16, wrong_to_wrong_blocked 30; gate0[out_lr[of+ot]]: helpful_accepted 17, helpful_blocked 15, harmful_accepted 11, harmful_blocked 21, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 31; gate0[comb_lr[full+act]]: helpful_accepted 22, helpful_blocked 10, harmful_accepted 7, harmful_blocked 25, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 34; gate0[eq[act+out_lr_full]]: helpful_accepted 14, helpful_blocked 18, harmful_accepted 3, harmful_blocked 29, wrong_to_wrong_accepted 7, wrong_to_wrong_blocked 39; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 12, helpful_blocked 20, harmful_accepted 3, harmful_blocked 29, wrong_to_wrong_accepted 6, wrong_to_wrong_blocked 40; gate0[act_fb]: helpful_accepted 9, helpful_blocked 23, harmful_accepted 3, harmful_blocked 29, wrong_to_wrong_accepted 4, wrong_to_wrong_blocked 42; gate_tuned[act_fb]: helpful_accepted 19, helpful_blocked 13, harmful_accepted 16, harmful_blocked 16, wrong_to_wrong_accepted 22, wrong_to_wrong_blocked 24

Calibration of revised-state probabilities (median over partitions): act_fb Brier 0.202 (constant 0.248), ECE 0.080; comb_lr[full+act] Brier 0.183 (constant 0.248), ECE 0.069; eq[act+out_lr_full] Brier 0.185 (constant 0.248), ECE 0.086; eq[act+out_lr_of+ot] Brier 0.190 (constant 0.248), ECE 0.089; out_hgb[full] Brier 0.203 (constant 0.248), ECE 0.049; out_lr[full] Brier 0.184 (constant 0.248), ECE 0.068; out_lr[of+ot] Brier 0.197 (constant 0.248), ECE 0.069

### debate_answer_only|eligible (109 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.672 (0.65–0.72) | 0.718 (0.70–0.74) | 0.670 (0.64–0.74) | 0.671 (0.61–0.71) | 0.634 (0.60–0.65) | 0.747 (0.68–0.81) |
| comb_lr[full+act] | 0.769 (0.74–0.79) | 0.809 (0.77–0.82) | 0.640 (0.60–0.72) | 0.632 (0.58–0.66) | 0.624 (0.60–0.66) | 0.604 (0.54–0.67) |
| eq[act+out_lr_full] | 0.765 (0.75–0.78) | 0.795 (0.76–0.81) | 0.690 (0.64–0.76) | 0.658 (0.63–0.74) | 0.641 (0.61–0.65) | 0.702 (0.64–0.80) |
| eq[act+out_lr_of+ot] | 0.747 (0.72–0.78) | 0.764 (0.74–0.79) | 0.670 (0.60–0.72) | 0.632 (0.58–0.68) | 0.654 (0.61–0.67) | 0.692 (0.60–0.78) |
| out_hgb[full] | 0.688 (0.64–0.74) | 0.748 (0.72–0.77) | 0.600 (0.56–0.68) | 0.645 (0.53–0.74) | 0.639 (0.58–0.68) | 0.640 (0.54–0.78) |
| out_lr[full] | 0.764 (0.73–0.78) | 0.812 (0.77–0.82) | 0.650 (0.60–0.74) | 0.632 (0.55–0.66) | 0.632 (0.60–0.66) | 0.606 (0.54–0.66) |
| out_lr[of+ot] | 0.744 (0.71–0.79) | 0.775 (0.75–0.81) | 0.600 (0.56–0.66) | 0.618 (0.58–0.63) | 0.607 (0.58–0.64) | 0.614 (0.53–0.68) |

Counts (partition 0): state_auc_pre 5052 pairs; state_auc_post 5056 pairs; panel_pre 50 pairs / 25 questions; panel_post 38 pairs / 19 questions; six 205 pairs / 29 questions; revision_auc 148 pairs / 28 questions; helpful 26, harmful 23, wrong-to-wrong 36

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.094 (0/1) | -0.078 (0/2) | +0.040 (0/0) | +0.026 (0/0) | -0.005 (0/0) | +0.146 (0/0) |
| act - out_hgb[full] | -0.027 (0/0) | -0.032 (0/0) | +0.080 (0/0) | +0.013 (0/0) | -0.002 (0/0) | +0.107 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.006 (0/0) | -0.003 (0/0) | +0.000 (0/0) | +0.013 (0/0) | -0.002 (0/0) | -0.006 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | -0.011 (0/0) | +0.080 (1/0) | +0.053 (0/0) | +0.015 (0/0) | +0.098 (2/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.009 (0/0) | -0.012 (0/0) | +0.040 (1/0) | +0.000 (0/0) | +0.032 (0/0) | +0.071 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 68 | 67 | 67 | 68 | 67 | 66 |
| best_original[comb_lr[full+act]] | 66 | 65 | 65 | 67 | 64 | 64 |
| best_original[eq[act+out_lr_full]] | 68 | 67 | 65 | 67 | 66 | 66 |
| best_original[eq[act+out_lr_of+ot]] | 67 | 65 | 64 | 66 | 68 | 66 |
| best_original[out_hgb[full]] | 65 | 65 | 64 | 67 | 65 | 65 |
| best_original[out_lr[full]] | 66 | 65 | 66 | 68 | 64 | 64 |
| best_original[out_lr[of+ot]] | 66 | 64 | 64 | 67 | 65 | 65 |
| best_six[act] | 69 | 68 | 68 | 68 | 67 | 67 |
| best_six[comb_lr[full+act]] | 64 | 64 | 65 | 67 | 64 | 63 |
| best_six[eq[act+out_lr_full]] | 66 | 66 | 66 | 67 | 65 | 65 |
| best_six[eq[act+out_lr_of+ot]] | 67 | 67 | 66 | 68 | 67 | 66 |
| best_six[out_hgb[full]] | 65 | 67 | 64 | 67 | 65 | 67 |
| best_six[out_lr[full]] | 64 | 64 | 66 | 67 | 64 | 64 |
| best_six[out_lr[of+ot]] | 64 | 64 | 64 | 66 | 64 | 64 |
| gate0[act] | 64 | 65 | 66 | 65 | 65 | 65 |
| gate0[comb_lr[full+act]] | 66 | 64 | 65 | 65 | 65 | 65 |
| gate0[eq[act+out_lr_full]] | 64 | 64 | 64 | 64 | 63 | 65 |
| gate0[eq[act+out_lr_of+ot]] | 65 | 66 | 66 | 66 | 65 | 65 |
| gate0[out_hgb[full]] | 65 | 66 | 63 | 66 | 63 | 65 |
| gate0[out_lr[full]] | 66 | 64 | 65 | 65 | 64 | 65 |
| gate0[out_lr[of+ot]] | 65 | 64 | 63 | 65 | 63 | 64 |
| gate_tuned[act] | 65 | 67 | 66 | 66 | 66 | 67 |
| original_vote | 65 | 65 | 65 | 65 | 65 | 65 |
| pooled6_vote | 65 | 65 | 65 | 65 | 65 | 65 |
| revised_vote | 64 | 64 | 64 | 64 | 64 | 64 |

Random tie-break expectation: original 66.33, revised 65.00, pooled six 65.67. Questions with a correct round-0 answer 75, with a correct revised answer 74, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.028 | 6/2, 3/0, 3/0, 4/1, 3/1, 4/2 |
| best_six[act] − best_original[act] | +0.009 | 1/0, 2/1, 2/1, 0/0, 1/1, 2/1 |
| best_original[act] − original_vote | +0.018 | 6/3, 4/2, 4/2, 4/1, 4/2, 5/4 |
| gate0[act] − original_vote | +0.000 | 0/1, 1/1, 1/0, 1/1, 1/1, 1/1 |
| gate0[act] − pooled6_vote | +0.000 | 1/2, 2/2, 2/1, 2/2, 2/2, 1/1 |
| best_six[act] − best_six[out_lr[full]] | +0.028 | 7/2, 5/1, 4/2, 4/3, 6/3, 6/3 |
| best_original[act] − best_original[out_lr[full]] | +0.018 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act] − gate0[out_lr[full]] | +0.005 | 1/3, 2/1, 2/1, 2/2, 2/1, 2/2 |
| best_six[out_lr[full]] − pooled6_vote | -0.009 | 2/3, 1/2, 2/1, 3/1, 2/3, 2/3 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | -0.005 | 0/2, 0/1, 0/0, 0/1, 1/1, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.005 | 2/1, 1/1, 2/1, 3/0, 1/2, 1/2 |
| gate0[out_lr[full]] − original_vote | +0.000 | 2/1, 1/2, 1/1, 1/1, 1/2, 2/2 |
| gate0[out_lr[full]] − pooled6_vote | +0.000 | 2/1, 1/2, 1/1, 1/1, 1/2, 2/2 |
| best_six[out_hgb[full]] − pooled6_vote | +0.009 | 3/3, 3/1, 2/3, 3/1, 3/3, 3/1 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | +0.000 | 1/1, 2/0, 1/1, 1/1, 1/1, 3/1 |
| best_original[out_hgb[full]] − original_vote | +0.000 | 3/3, 2/2, 3/4, 3/1, 2/2, 2/2 |
| gate0[out_hgb[full]] − original_vote | +0.000 | 1/1, 2/1, 1/3, 2/1, 0/2, 1/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 1/1, 2/1, 1/3, 1/0, 0/2, 1/1 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | +0.009 | 3/2, 4/1, 0/2, 1/1, 2/1, 4/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | -0.005 | 2/3, 2/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.005 | 1/2, 3/1, 0/2, 1/0, 0/1, 2/2 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.009 | 2/3, 2/3, 2/3, 3/2, 2/3, 2/3 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.009 | 0/2, 2/2, 2/2, 0/1, 0/1, 1/2 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 2/1, 1/2, 2/3, 3/1, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − original_vote | -0.009 | 1/1, 1/2, 1/3, 1/1, 1/3, 1/2 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.009 | 2/2, 1/2, 1/3, 2/2, 1/3, 2/3 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | +0.000 | 0/0, 1/1, 0/2, 0/1, 0/0, 0/0 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.005 | 0/0, 1/2, 1/3, 0/1, 2/1, 1/0 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.009 | 1/2, 2/2, 0/2, 1/1, 0/1, 1/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.009 | 2/3, 1/2, 2/2, 3/1, 2/3, 1/3 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | -0.005 | 0/2, 0/1, 0/0, 1/1, 1/1, 1/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.000 | 2/1, 1/1, 2/2, 2/0, 1/2, 1/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.000 | 2/1, 1/2, 1/1, 1/1, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/1, 1/2, 1/1, 1/1, 2/2, 2/2 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/0, 0/0, 0/1 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/0, 0/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.009 | 3/2, 2/1, 1/0, 2/0, 1/1, 1/1 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | -0.009 | 0/2, 0/1, 1/0, 1/1, 1/2, 0/1 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.014 | 3/0, 2/0, 1/1, 2/0, 1/0, 2/1 |
| gate0[eq[act+out_lr_full]] − original_vote | -0.009 | 0/1, 0/1, 0/1, 0/1, 0/2, 1/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.009 | 1/2, 1/2, 0/1, 1/2, 0/2, 2/2 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.009 | 2/0, 2/0, 1/1, 1/1, 3/2, 3/2 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.018 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | -0.009 | 1/3, 1/1, 0/1, 1/2, 0/1, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.018 | 3/1, 3/1, 2/1, 3/0, 3/1, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.009 | 1/1, 3/1, 3/1, 2/0, 0/1, 1/1 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.009 | 3/1, 2/2, 2/3, 2/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.005 | 1/1, 1/0, 1/0, 1/0, 1/1, 1/1 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.005 | 2/2, 2/1, 2/1, 2/1, 2/2, 2/2 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.023 | 3/0, 3/0, 1/1, 2/1, 3/0, 3/1 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.005 | 2/1, 2/2, 1/3, 0/2, 5/1, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.009 | 1/2, 3/1, 1/0, 1/0, 1/0, 1/1 |
| gate_tuned[act] − original_vote | +0.009 | 2/2, 5/3, 1/0, 2/1, 3/2, 4/2 |

Origin of best-of-six picks (partition 0): act: 78 from revisions, 0 new options (0 correct); out_lr[full]: 75 from revisions, 0 new options (0 correct); out_hgb[full]: 67 from revisions, 0 new options (0 correct); out_lr[of+ot]: 59 from revisions, 0 new options (0 correct); comb_lr[full+act]: 76 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 77 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 79 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 4, helpful_blocked 22, harmful_accepted 1, harmful_blocked 22, wrong_to_wrong_accepted 2, wrong_to_wrong_blocked 34; gate_tuned[act]: helpful_accepted 17, helpful_blocked 9, harmful_accepted 9, harmful_blocked 14, wrong_to_wrong_accepted 18, wrong_to_wrong_blocked 18; gate0[out_lr[full]]: helpful_accepted 14, helpful_blocked 12, harmful_accepted 4, harmful_blocked 19, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 24; gate0[out_hgb[full]]: helpful_accepted 13, helpful_blocked 13, harmful_accepted 5, harmful_blocked 18, wrong_to_wrong_accepted 10, wrong_to_wrong_blocked 26; gate0[out_lr[of+ot]]: helpful_accepted 12, helpful_blocked 14, harmful_accepted 5, harmful_blocked 18, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 25; gate0[comb_lr[full+act]]: helpful_accepted 14, helpful_blocked 12, harmful_accepted 4, harmful_blocked 19, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 24; gate0[eq[act+out_lr_full]]: helpful_accepted 9, helpful_blocked 17, harmful_accepted 1, harmful_blocked 22, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 31; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 8, helpful_blocked 18, harmful_accepted 1, harmful_blocked 22, wrong_to_wrong_accepted 2, wrong_to_wrong_blocked 34

Calibration of revised-state probabilities (median over partitions): act Brier 0.210 (constant 0.246), ECE 0.070; comb_lr[full+act] Brier 0.187 (constant 0.246), ECE 0.081; eq[act+out_lr_full] Brier 0.192 (constant 0.246), ECE 0.073; eq[act+out_lr_of+ot] Brier 0.195 (constant 0.246), ECE 0.062; out_hgb[full] Brier 0.203 (constant 0.246), ECE 0.059; out_lr[full] Brier 0.185 (constant 0.246), ECE 0.083; out_lr[of+ot] Brier 0.192 (constant 0.246), ECE 0.073

### debate_answer_only|all_fallback (120 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb | 0.681 (0.67–0.72) | 0.721 (0.70–0.74) | 0.653 (0.63–0.71) | 0.690 (0.62–0.74) | 0.634 (0.60–0.65) | 0.732 (0.68–0.81) |
| comb_lr[full+act] | 0.773 (0.74–0.79) | 0.810 (0.78–0.82) | 0.637 (0.61–0.69) | 0.655 (0.60–0.69) | 0.640 (0.61–0.66) | 0.629 (0.58–0.71) |
| eq[act+out_lr_full] | 0.770 (0.75–0.78) | 0.793 (0.77–0.81) | 0.685 (0.65–0.74) | 0.667 (0.67–0.76) | 0.654 (0.63–0.66) | 0.719 (0.69–0.80) |
| eq[act+out_lr_of+ot] | 0.753 (0.72–0.78) | 0.770 (0.75–0.80) | 0.645 (0.60–0.68) | 0.667 (0.62–0.71) | 0.648 (0.62–0.66) | 0.695 (0.59–0.76) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.764 (0.73–0.79) | 0.613 (0.58–0.73) | 0.655 (0.55–0.71) | 0.636 (0.60–0.70) | 0.691 (0.58–0.81) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.813 (0.78–0.82) | 0.653 (0.60–0.71) | 0.667 (0.57–0.69) | 0.652 (0.61–0.67) | 0.640 (0.60–0.72) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.787 (0.77–0.82) | 0.581 (0.55–0.63) | 0.655 (0.62–0.67) | 0.614 (0.58–0.62) | 0.622 (0.56–0.68) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6170 pairs; panel_pre 62 pairs / 31 questions; panel_post 42 pairs / 21 questions; six 250 pairs / 35 questions; revision_auc 207 pairs / 33 questions; helpful 32, harmful 26, wrong-to-wrong 45

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb - out_lr[full] | -0.089 (0/1) | -0.081 (0/3) | +0.016 (0/0) | +0.012 (0/0) | -0.016 (0/0) | +0.111 (0/0) |
| act_fb - out_hgb[full] | -0.024 (0/0) | -0.045 (0/0) | +0.040 (0/0) | +0.036 (0/0) | -0.002 (0/0) | +0.041 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.007 (0/0) | -0.003 (0/0) | +0.000 (0/0) | +0.000 (0/0) | -0.002 (0/0) | -0.001 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | -0.014 (0/0) | +0.056 (0/0) | +0.048 (0/0) | +0.004 (0/0) | +0.093 (1/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.011 (0/0) | -0.017 (0/0) | +0.048 (1/0) | +0.000 (0/0) | +0.036 (1/0) | +0.068 (1/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act_fb] | 73 | 72 | 72 | 73 | 72 | 71 |
| best_original[comb_lr[full+act]] | 71 | 70 | 69 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 73 | 72 | 70 | 72 | 71 | 71 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 69 | 69 | 70 | 72 | 71 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act_fb] | 74 | 73 | 73 | 73 | 72 | 72 |
| best_six[comb_lr[full+act]] | 69 | 70 | 70 | 72 | 69 | 68 |
| best_six[eq[act+out_lr_full]] | 71 | 72 | 71 | 72 | 70 | 70 |
| best_six[eq[act+out_lr_of+ot]] | 71 | 72 | 71 | 72 | 71 | 71 |
| best_six[out_hgb[full]] | 70 | 73 | 69 | 72 | 71 | 72 |
| best_six[out_lr[full]] | 69 | 70 | 72 | 72 | 69 | 69 |
| best_six[out_lr[of+ot]] | 68 | 69 | 69 | 70 | 68 | 68 |
| gate0[act_fb] | 68 | 70 | 71 | 69 | 70 | 69 |
| gate0[comb_lr[full+act]] | 71 | 70 | 70 | 69 | 70 | 70 |
| gate0[eq[act+out_lr_full]] | 69 | 69 | 69 | 68 | 68 | 70 |
| gate0[eq[act+out_lr_of+ot]] | 69 | 71 | 71 | 70 | 69 | 69 |
| gate0[out_hgb[full]] | 70 | 71 | 68 | 71 | 68 | 70 |
| gate0[out_lr[full]] | 72 | 69 | 71 | 69 | 69 | 70 |
| gate0[out_lr[of+ot]] | 69 | 69 | 68 | 69 | 66 | 68 |
| gate_tuned[act_fb] | 69 | 73 | 71 | 71 | 70 | 72 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 70 | 70 | 70 | 70 | 70 | 70 |
| revised_vote | 69 | 69 | 69 | 69 | 69 | 69 |

Random tie-break expectation: original 70.67, revised 70.33, pooled six 71.00. Questions with a correct round-0 answer 82, with a correct revised answer 80, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[out_lr[full]] − pooled6_vote | -0.004 | 3/4, 2/2, 3/1, 4/2, 3/4, 3/4 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 0/2, 1/1, 1/0, 0/1, 1/1, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | +0.004 | 4/1, 2/2, 3/1, 2/2, 2/2, 3/2 |
| gate0[out_lr[full]] − pooled6_vote | -0.004 | 3/1, 2/3, 2/1, 1/2, 2/3, 3/3 |
| best_six[out_hgb[full]] − pooled6_vote | +0.013 | 4/4, 4/1, 3/4, 4/2, 4/3, 4/2 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | +0.004 | 1/1, 2/0, 1/1, 1/1, 2/1, 3/1 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.008 | 2/1, 3/1, 2/3, 3/1, 1/2, 2/1 |
| gate0[out_hgb[full]] − pooled6_vote | +0.000 | 2/2, 3/2, 2/4, 2/1, 1/3, 2/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | +0.013 | 3/2, 4/1, 0/3, 1/1, 3/1, 4/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | -0.004 | 1/3, 3/1, 0/3, 2/0, 1/2, 2/2 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.013 | 2/4, 2/3, 2/3, 3/3, 2/4, 2/4 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | -0.008 | 0/2, 3/2, 3/2, 0/1, 0/1, 1/2 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | -0.004 | 2/2, 3/3, 3/4, 2/2, 1/4, 2/3 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.013 | 2/3, 1/2, 1/3, 2/3, 1/5, 2/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.008 | 0/1, 1/2, 0/3, 0/2, 0/1, 0/1 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | -0.021 | 1/4, 3/3, 0/3, 1/1, 0/3, 1/3 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.004 | 3/4, 2/2, 2/2, 4/2, 3/4, 2/4 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.000 | 0/2, 1/1, 1/0, 1/1, 1/1, 1/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.004 | 3/1, 2/1, 3/3, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | +0.008 | 4/2, 3/2, 3/2, 2/2, 3/2, 3/2 |
| gate0[comb_lr[full+act]] − pooled6_vote | +0.000 | 2/1, 2/2, 1/1, 1/2, 3/3, 3/3 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/0, 0/0, 0/1 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/1, 1/0, 0/1, 0/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.008 | 4/3, 3/1, 2/1, 3/1, 2/2, 2/2 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | -0.004 | 0/2, 1/1, 1/0, 1/1, 1/2, 0/1 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.021 | 4/0, 3/0, 2/1, 3/0, 2/0, 3/1 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.000 | 2/2, 2/2, 2/2, 1/2, 1/2, 2/1 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.008 | 1/2, 1/2, 0/1, 1/3, 1/3, 3/3 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.008 | 2/0, 2/0, 1/2, 1/1, 3/2, 3/2 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.017 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | -0.008 | 1/4, 2/2, 0/2, 1/2, 1/2, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.008 | 3/2, 3/1, 3/2, 3/1, 3/2, 3/2 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.008 | 1/1, 4/1, 3/1, 2/0, 0/1, 1/1 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.013 | 4/2, 3/3, 3/3, 3/2, 5/2, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.004 | 2/2, 3/1, 3/1, 2/1, 2/2, 2/2 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.004 | 2/3, 2/1, 2/1, 2/2, 2/3, 2/3 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.017 | 3/1, 3/1, 1/2, 2/2, 3/1, 3/1 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | -0.004 | 2/2, 2/3, 1/3, 0/3, 5/2, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.000 | 1/4, 4/2, 1/1, 1/0, 2/2, 1/2 |
| best_six[act_fb] − pooled6_vote | +0.025 | 7/3, 4/1, 4/1, 5/2, 4/2, 5/3 |
| best_six[act_fb] − best_original[act_fb] | +0.008 | 1/0, 2/1, 2/1, 0/0, 1/1, 2/1 |
| best_original[act_fb] − original_vote | +0.025 | 7/3, 5/2, 5/2, 5/1, 5/2, 6/4 |
| gate0[act_fb] − original_vote | +0.004 | 1/2, 3/2, 3/1, 2/2, 2/1, 2/2 |
| gate0[act_fb] − pooled6_vote | -0.004 | 1/3, 2/2, 2/1, 2/3, 3/3, 1/2 |
| best_six[act_fb] − best_six[out_lr[full]] | +0.025 | 7/2, 5/2, 4/3, 4/3, 6/3, 6/3 |
| best_original[act_fb] − best_original[out_lr[full]] | +0.017 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act_fb] − gate0[out_lr[full]] | +0.000 | 1/5, 3/2, 2/2, 2/2, 3/2, 2/3 |
| gate_tuned[act_fb] − original_vote | +0.017 | 3/3, 7/3, 3/1, 4/2, 4/3, 6/3 |

Origin of best-of-six picks (partition 0): out_lr[full]: 77 from revisions, 0 new options (0 correct); out_hgb[full]: 71 from revisions, 0 new options (0 correct); out_lr[of+ot]: 62 from revisions, 0 new options (0 correct); comb_lr[full+act]: 78 from revisions, 0 new options (0 correct); eq[act+out_lr_full]: 79 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 82 from revisions, 0 new options (0 correct); act_fb: 82 from revisions, 1 new options (0 correct)

Gate, agent level (partition 0): gate0[out_lr[full]]: helpful_accepted 18, helpful_blocked 14, harmful_accepted 4, harmful_blocked 22, wrong_to_wrong_accepted 14, wrong_to_wrong_blocked 31; gate0[out_hgb[full]]: helpful_accepted 15, helpful_blocked 17, harmful_accepted 5, harmful_blocked 21, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 34; gate0[out_lr[of+ot]]: helpful_accepted 15, helpful_blocked 17, harmful_accepted 7, harmful_blocked 19, wrong_to_wrong_accepted 12, wrong_to_wrong_blocked 33; gate0[comb_lr[full+act]]: helpful_accepted 18, helpful_blocked 14, harmful_accepted 5, harmful_blocked 21, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 32; gate0[eq[act+out_lr_full]]: helpful_accepted 13, helpful_blocked 19, harmful_accepted 2, harmful_blocked 24, wrong_to_wrong_accepted 7, wrong_to_wrong_blocked 38; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 10, helpful_blocked 22, harmful_accepted 2, harmful_blocked 24, wrong_to_wrong_accepted 3, wrong_to_wrong_blocked 42; gate0[act_fb]: helpful_accepted 5, helpful_blocked 27, harmful_accepted 2, harmful_blocked 24, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 40; gate_tuned[act_fb]: helpful_accepted 21, helpful_blocked 11, harmful_accepted 12, harmful_blocked 14, wrong_to_wrong_accepted 25, wrong_to_wrong_blocked 20

Calibration of revised-state probabilities (median over partitions): act_fb Brier 0.213 (constant 0.247), ECE 0.058; comb_lr[full+act] Brier 0.184 (constant 0.247), ECE 0.076; eq[act+out_lr_full] Brier 0.193 (constant 0.247), ECE 0.080; eq[act+out_lr_of+ot] Brier 0.193 (constant 0.247), ECE 0.058; out_hgb[full] Brier 0.199 (constant 0.247), ECE 0.062; out_lr[full] Brier 0.183 (constant 0.247), ECE 0.081; out_lr[of+ot] Brier 0.186 (constant 0.247), ECE 0.063

### self_reconsider|eligible (109 questions; absent states 21)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.672 (0.65–0.72) | 0.770 (0.75–0.80) | 0.670 (0.64–0.74) | 0.757 (0.72–0.78) | 0.735 (0.72–0.75) | 0.704 (0.63–0.73) |
| comb_lr[full+act] | 0.769 (0.74–0.79) | 0.800 (0.79–0.83) | 0.640 (0.60–0.72) | 0.764 (0.72–0.81) | 0.709 (0.71–0.73) | 0.768 (0.72–0.82) |
| eq[act+out_lr_full] | 0.765 (0.75–0.78) | 0.818 (0.81–0.85) | 0.690 (0.64–0.76) | 0.819 (0.79–0.88) | 0.744 (0.73–0.75) | 0.802 (0.74–0.83) |
| eq[act+out_lr_of+ot] | 0.747 (0.72–0.78) | 0.828 (0.82–0.84) | 0.670 (0.60–0.72) | 0.840 (0.82–0.86) | 0.746 (0.73–0.76) | 0.809 (0.74–0.83) |
| out_hgb[full] | 0.688 (0.64–0.74) | 0.755 (0.73–0.80) | 0.600 (0.56–0.68) | 0.715 (0.62–0.75) | 0.705 (0.66–0.72) | 0.712 (0.62–0.75) |
| out_lr[full] | 0.764 (0.73–0.78) | 0.802 (0.79–0.83) | 0.650 (0.60–0.74) | 0.764 (0.72–0.79) | 0.707 (0.70–0.73) | 0.762 (0.72–0.80) |
| out_lr[of+ot] | 0.744 (0.71–0.79) | 0.808 (0.79–0.82) | 0.600 (0.56–0.66) | 0.785 (0.76–0.83) | 0.742 (0.73–0.76) | 0.777 (0.69–0.81) |

Counts (partition 0): state_auc_pre 5052 pairs; state_auc_post 5259 pairs; panel_pre 50 pairs / 25 questions; panel_post 72 pairs / 36 questions; six 447 pairs / 59 questions; revision_auc 522 pairs / 57 questions; helpful 32, harmful 77, wrong-to-wrong 57

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.094 (0/1) | -0.032 (0/0) | +0.040 (0/0) | -0.021 (0/0) | +0.029 (0/0) | -0.053 (0/0) |
| act - out_hgb[full] | -0.027 (0/0) | +0.022 (0/0) | +0.080 (0/0) | +0.056 (0/0) | +0.037 (0/0) | -0.023 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.006 (0/0) | -0.001 (0/0) | +0.000 (0/0) | +0.014 (0/0) | +0.001 (0/0) | +0.006 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | +0.016 (0/0) | +0.080 (1/0) | +0.069 (1/0) | +0.034 (0/0) | +0.023 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.009 (0/0) | +0.022 (0/0) | +0.040 (1/0) | +0.049 (0/0) | -0.001 (0/0) | +0.028 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 68 | 67 | 67 | 68 | 67 | 66 |
| best_original[comb_lr[full+act]] | 66 | 65 | 65 | 67 | 64 | 64 |
| best_original[eq[act+out_lr_full]] | 68 | 67 | 65 | 67 | 66 | 66 |
| best_original[eq[act+out_lr_of+ot]] | 67 | 65 | 64 | 66 | 68 | 66 |
| best_original[out_hgb[full]] | 65 | 65 | 64 | 67 | 65 | 65 |
| best_original[out_lr[full]] | 66 | 65 | 66 | 68 | 64 | 64 |
| best_original[out_lr[of+ot]] | 66 | 64 | 64 | 67 | 65 | 65 |
| best_six[act] | 69 | 67 | 68 | 69 | 68 | 68 |
| best_six[comb_lr[full+act]] | 65 | 66 | 67 | 67 | 65 | 65 |
| best_six[eq[act+out_lr_full]] | 68 | 67 | 66 | 68 | 65 | 66 |
| best_six[eq[act+out_lr_of+ot]] | 69 | 68 | 65 | 68 | 68 | 67 |
| best_six[out_hgb[full]] | 64 | 62 | 65 | 63 | 65 | 64 |
| best_six[out_lr[full]] | 66 | 64 | 68 | 67 | 64 | 65 |
| best_six[out_lr[of+ot]] | 64 | 64 | 64 | 67 | 65 | 65 |
| gate0[act] | 65 | 64 | 64 | 65 | 65 | 65 |
| gate0[comb_lr[full+act]] | 63 | 65 | 64 | 64 | 63 | 63 |
| gate0[eq[act+out_lr_full]] | 64 | 65 | 64 | 65 | 64 | 65 |
| gate0[eq[act+out_lr_of+ot]] | 65 | 65 | 65 | 65 | 65 | 65 |
| gate0[out_hgb[full]] | 63 | 63 | 66 | 65 | 65 | 65 |
| gate0[out_lr[full]] | 63 | 65 | 65 | 64 | 63 | 63 |
| gate0[out_lr[of+ot]] | 67 | 67 | 67 | 67 | 64 | 67 |
| gate_tuned[act] | 63 | 64 | 63 | 66 | 63 | 67 |
| original_vote | 65 | 65 | 65 | 65 | 65 | 65 |
| pooled6_vote | 67 | 67 | 67 | 67 | 67 | 67 |
| revised_vote | 53 | 53 | 53 | 53 | 53 | 53 |

Random tie-break expectation: original 66.33, revised 49.67, pooled six 63.83. Questions with a correct round-0 answer 75, with a correct revised answer 67, correct only after revision 9.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.009 | 5/3, 4/4, 3/2, 3/1, 4/3, 3/2 |
| best_six[act] − best_original[act] | +0.009 | 1/0, 1/1, 1/0, 1/0, 2/1, 2/0 |
| best_original[act] − original_vote | +0.018 | 6/3, 4/2, 4/2, 4/1, 4/2, 5/4 |
| gate0[act] − original_vote | +0.000 | 0/0, 0/1, 0/1, 0/0, 0/0, 0/0 |
| gate0[act] − pooled6_vote | -0.018 | 2/4, 2/5, 2/5, 2/4, 2/4, 2/4 |
| best_six[act] − best_six[out_lr[full]] | +0.028 | 9/6, 9/6, 7/7, 7/5, 8/4, 7/4 |
| best_original[act] − best_original[out_lr[full]] | +0.018 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act] − gate0[out_lr[full]] | +0.014 | 3/1, 2/3, 3/4, 3/2, 3/1, 3/1 |
| best_six[out_lr[full]] − pooled6_vote | -0.014 | 5/6, 3/6, 6/5, 6/6, 4/7, 4/6 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 3/3, 2/3, 3/1, 3/4, 3/3, 4/3 |
| best_original[out_lr[full]] − original_vote | +0.005 | 2/1, 1/1, 2/1, 3/0, 1/2, 1/2 |
| gate0[out_lr[full]] − original_vote | -0.014 | 1/3, 2/2, 3/3, 2/3, 1/3, 1/3 |
| gate0[out_lr[full]] − pooled6_vote | -0.032 | 2/6, 2/4, 3/5, 2/5, 2/6, 2/6 |
| best_six[out_hgb[full]] − pooled6_vote | -0.028 | 5/8, 3/8, 6/8, 5/9, 5/7, 5/8 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.009 | 3/4, 2/5, 6/5, 3/7, 3/3, 3/4 |
| best_original[out_hgb[full]] − original_vote | +0.000 | 3/3, 2/2, 3/4, 3/1, 2/2, 2/2 |
| gate0[out_hgb[full]] − original_vote | +0.000 | 1/3, 2/4, 5/4, 4/4, 4/4, 3/3 |
| gate0[out_hgb[full]] − pooled6_vote | -0.018 | 1/5, 2/6, 4/5, 4/6, 4/6, 4/6 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.018 | 4/6, 5/7, 5/8, 1/5, 5/4, 5/6 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | -0.005 | 2/3, 2/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.009 | 1/1, 1/3, 4/3, 2/1, 5/3, 3/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.023 | 2/5, 1/4, 2/5, 4/4, 2/4, 2/4 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.000 | 1/3, 1/1, 1/1, 2/2, 2/2, 1/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 2/1, 1/2, 2/3, 3/1, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − original_vote | +0.018 | 3/1, 2/0, 2/0, 2/0, 2/3, 2/0 |
| gate0[out_lr[of+ot]] − pooled6_vote | +0.000 | 3/3, 3/3, 3/3, 3/3, 2/5, 3/3 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | +0.000 | 1/3, 3/3, 2/6, 3/3, 5/4, 3/3 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.005 | 0/0, 1/2, 1/3, 0/1, 2/1, 1/0 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.023 | 4/0, 2/0, 4/2, 3/0, 3/2, 4/0 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.014 | 4/6, 4/5, 5/5, 5/5, 4/6, 4/6 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.009 | 2/3, 2/1, 3/1, 3/3, 3/2, 4/3 |
| best_original[comb_lr[full+act]] − original_vote | +0.000 | 2/1, 1/1, 2/2, 2/0, 1/2, 1/2 |
| gate0[comb_lr[full+act]] − original_vote | -0.014 | 1/3, 2/2, 2/3, 1/2, 1/3, 1/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.032 | 2/6, 2/4, 3/6, 2/5, 2/6, 2/6 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/1, 2/0, 1/2, 1/1, 1/0, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 1/1, 0/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | -0.005 | 4/3, 3/3, 2/3, 4/3, 2/4, 2/3 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.000 | 2/2, 2/2, 1/0, 2/1, 1/2, 1/1 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.014 | 3/0, 2/0, 1/1, 2/0, 1/0, 2/1 |
| gate0[eq[act+out_lr_full]] − original_vote | -0.005 | 0/1, 1/1, 0/1, 0/0, 0/1, 0/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.023 | 1/4, 2/4, 1/4, 2/4, 1/4, 2/4 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.009 | 5/3, 6/3, 2/4, 5/4, 4/3, 4/3 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.018 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.009 | 2/1, 1/1, 2/3, 3/2, 2/1, 3/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.009 | 5/3, 4/3, 3/5, 4/3, 4/3, 3/3 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.014 | 2/0, 3/0, 2/1, 2/0, 1/1, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.009 | 3/1, 2/2, 2/3, 2/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/0, 0/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.018 | 2/4, 2/4, 2/4, 2/4, 2/4, 2/4 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.023 | 6/3, 7/3, 3/6, 5/4, 6/2, 5/3 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.005 | 2/1, 2/2, 1/3, 0/2, 5/1, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.014 | 3/1, 2/2, 3/3, 3/2, 3/1, 3/1 |
| gate_tuned[act] − original_vote | -0.014 | 5/7, 7/8, 0/2, 3/2, 3/5, 8/6 |

Origin of best-of-six picks (partition 0): act: 29 from revisions, 1 new options (1 correct); out_lr[full]: 35 from revisions, 4 new options (1 correct); out_hgb[full]: 44 from revisions, 8 new options (1 correct); out_lr[of+ot]: 21 from revisions, 2 new options (0 correct); comb_lr[full+act]: 35 from revisions, 4 new options (1 correct); eq[act+out_lr_full]: 31 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 21 from revisions, 0 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 1, helpful_blocked 31, harmful_accepted 1, harmful_blocked 76, wrong_to_wrong_accepted 1, wrong_to_wrong_blocked 56; gate_tuned[act]: helpful_accepted 17, helpful_blocked 15, harmful_accepted 25, harmful_blocked 52, wrong_to_wrong_accepted 24, wrong_to_wrong_blocked 33; gate0[out_lr[full]]: helpful_accepted 9, helpful_blocked 23, harmful_accepted 8, harmful_blocked 69, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 44; gate0[out_hgb[full]]: helpful_accepted 9, helpful_blocked 23, harmful_accepted 11, harmful_blocked 66, wrong_to_wrong_accepted 17, wrong_to_wrong_blocked 40; gate0[out_lr[of+ot]]: helpful_accepted 11, helpful_blocked 21, harmful_accepted 3, harmful_blocked 74, wrong_to_wrong_accepted 10, wrong_to_wrong_blocked 47; gate0[comb_lr[full+act]]: helpful_accepted 8, helpful_blocked 24, harmful_accepted 8, harmful_blocked 69, wrong_to_wrong_accepted 13, wrong_to_wrong_blocked 44; gate0[eq[act+out_lr_full]]: helpful_accepted 2, helpful_blocked 30, harmful_accepted 2, harmful_blocked 75, wrong_to_wrong_accepted 4, wrong_to_wrong_blocked 53; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 0, helpful_blocked 32, harmful_accepted 0, harmful_blocked 77, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 52

Calibration of revised-state probabilities (median over partitions): act Brier 0.192 (constant 0.254), ECE 0.065; comb_lr[full+act] Brier 0.181 (constant 0.254), ECE 0.074; eq[act+out_lr_full] Brier 0.170 (constant 0.254), ECE 0.060; eq[act+out_lr_of+ot] Brier 0.172 (constant 0.254), ECE 0.060; out_hgb[full] Brier 0.199 (constant 0.254), ECE 0.052; out_lr[full] Brier 0.180 (constant 0.254), ECE 0.081; out_lr[of+ot] Brier 0.180 (constant 0.254), ECE 0.050

### self_reconsider|all_fallback (120 questions; absent states 21)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb | 0.681 (0.67–0.72) | 0.751 (0.75–0.78) | 0.653 (0.63–0.71) | 0.744 (0.72–0.77) | 0.721 (0.71–0.73) | 0.684 (0.62–0.71) |
| comb_lr[full+act] | 0.773 (0.74–0.79) | 0.804 (0.79–0.83) | 0.637 (0.61–0.69) | 0.768 (0.73–0.79) | 0.706 (0.70–0.72) | 0.759 (0.72–0.80) |
| eq[act+out_lr_full] | 0.770 (0.75–0.78) | 0.810 (0.80–0.84) | 0.685 (0.65–0.74) | 0.811 (0.78–0.85) | 0.735 (0.72–0.74) | 0.787 (0.74–0.82) |
| eq[act+out_lr_of+ot] | 0.753 (0.72–0.78) | 0.821 (0.81–0.83) | 0.645 (0.60–0.68) | 0.817 (0.79–0.82) | 0.727 (0.72–0.74) | 0.778 (0.74–0.81) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.762 (0.75–0.81) | 0.613 (0.58–0.73) | 0.726 (0.65–0.74) | 0.702 (0.67–0.72) | 0.727 (0.63–0.77) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.806 (0.79–0.83) | 0.653 (0.60–0.71) | 0.756 (0.73–0.79) | 0.705 (0.69–0.73) | 0.753 (0.72–0.79) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.813 (0.80–0.83) | 0.581 (0.55–0.63) | 0.780 (0.77–0.83) | 0.733 (0.73–0.74) | 0.753 (0.70–0.78) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6332 pairs; panel_pre 62 pairs / 31 questions; panel_post 82 pairs / 41 questions; six 494 pairs / 65 questions; revision_auc 575 pairs / 60 questions; helpful 35, harmful 79, wrong-to-wrong 67

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb - out_lr[full] | -0.089 (0/1) | -0.052 (0/1) | +0.016 (0/0) | -0.018 (0/0) | +0.015 (0/0) | -0.066 (0/0) |
| act_fb - out_hgb[full] | -0.024 (0/0) | -0.006 (0/0) | +0.040 (0/0) | +0.037 (0/0) | +0.019 (0/0) | -0.047 (0/0) |
| comb_lr[full+act] - out_lr[full] | +0.007 (0/0) | -0.002 (0/0) | +0.000 (0/0) | +0.006 (0/0) | +0.001 (0/0) | +0.001 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | +0.007 (0/0) | +0.056 (0/0) | +0.055 (1/0) | +0.028 (0/0) | +0.025 (0/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.011 (0/0) | +0.007 (0/0) | +0.048 (1/0) | +0.030 (0/0) | -0.009 (0/0) | +0.024 (0/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act_fb] | 73 | 72 | 72 | 73 | 72 | 71 |
| best_original[comb_lr[full+act]] | 71 | 70 | 69 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 73 | 72 | 70 | 72 | 71 | 71 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 69 | 69 | 70 | 72 | 71 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act_fb] | 74 | 72 | 73 | 74 | 73 | 73 |
| best_six[comb_lr[full+act]] | 70 | 71 | 71 | 72 | 70 | 70 |
| best_six[eq[act+out_lr_full]] | 73 | 72 | 71 | 73 | 70 | 71 |
| best_six[eq[act+out_lr_of+ot]] | 73 | 72 | 70 | 72 | 72 | 72 |
| best_six[out_hgb[full]] | 70 | 68 | 71 | 68 | 71 | 69 |
| best_six[out_lr[full]] | 71 | 70 | 73 | 72 | 69 | 70 |
| best_six[out_lr[of+ot]] | 69 | 69 | 69 | 71 | 69 | 69 |
| gate0[act_fb] | 69 | 68 | 68 | 69 | 69 | 69 |
| gate0[comb_lr[full+act]] | 67 | 69 | 68 | 68 | 67 | 67 |
| gate0[eq[act+out_lr_full]] | 68 | 69 | 68 | 69 | 68 | 69 |
| gate0[eq[act+out_lr_of+ot]] | 69 | 69 | 69 | 69 | 69 | 69 |
| gate0[out_hgb[full]] | 67 | 67 | 70 | 69 | 69 | 69 |
| gate0[out_lr[full]] | 67 | 69 | 69 | 68 | 67 | 67 |
| gate0[out_lr[of+ot]] | 71 | 71 | 71 | 70 | 68 | 71 |
| gate_tuned[act_fb] | 67 | 68 | 67 | 70 | 67 | 71 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 72 | 72 | 72 | 72 | 72 | 72 |
| revised_vote | 58 | 58 | 58 | 58 | 58 | 58 |

Random tie-break expectation: original 70.67, revised 54.33, pooled six 69.67. Questions with a correct round-0 answer 82, with a correct revised answer 74, correct only after revision 9.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[out_lr[full]] − pooled6_vote | -0.013 | 6/7, 5/7, 7/6, 7/7, 5/8, 5/7 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 3/3, 3/3, 3/1, 3/4, 3/3, 4/3 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | -0.013 | 1/3, 2/2, 3/3, 2/3, 1/3, 1/3 |
| gate0[out_lr[full]] − pooled6_vote | -0.037 | 2/7, 2/5, 3/6, 2/6, 2/7, 2/7 |
| best_six[out_hgb[full]] − pooled6_vote | -0.021 | 7/9, 5/9, 8/9, 6/10, 7/8, 6/9 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.004 | 4/4, 2/5, 7/5, 3/7, 4/3, 3/4 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.000 | 1/3, 2/4, 5/4, 4/4, 4/4, 3/3 |
| gate0[out_hgb[full]] − pooled6_vote | -0.025 | 1/6, 2/7, 4/6, 4/7, 4/7, 4/7 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.013 | 5/6, 5/7, 6/8, 1/5, 6/4, 5/6 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.008 | 1/1, 1/3, 4/3, 2/1, 5/3, 3/1 |
| best_six[out_lr[of+ot]] − pooled6_vote | -0.025 | 3/6, 2/5, 3/6, 5/6, 3/6, 3/6 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.000 | 2/3, 2/1, 2/1, 2/2, 2/2, 1/1 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.017 | 3/1, 2/0, 2/0, 2/1, 2/3, 2/0 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.008 | 3/4, 3/4, 3/4, 3/5, 2/6, 3/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | -0.008 | 2/4, 4/5, 3/7, 3/4, 5/5, 3/4 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.017 | 4/0, 2/0, 4/2, 3/1, 3/2, 4/0 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.013 | 5/7, 5/6, 6/7, 6/6, 5/7, 5/7 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.008 | 2/3, 2/1, 3/1, 3/3, 3/2, 4/3 |
| best_original[comb_lr[full+act]] − original_vote | +0.004 | 3/1, 2/1, 3/3, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | -0.013 | 1/3, 2/2, 2/3, 1/2, 1/3, 1/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.037 | 2/7, 2/5, 3/7, 2/6, 2/7, 2/7 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.000 | 0/1, 2/1, 1/3, 1/1, 1/0, 0/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 1/1, 0/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | -0.004 | 5/4, 4/4, 3/4, 5/4, 3/5, 3/4 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.000 | 2/2, 2/2, 1/0, 2/1, 1/2, 1/1 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.021 | 4/0, 3/0, 2/1, 3/0, 2/0, 3/1 |
| gate0[eq[act+out_lr_full]] − original_vote | -0.004 | 0/1, 1/1, 0/1, 0/0, 0/1, 0/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.029 | 1/5, 2/5, 1/5, 2/5, 1/5, 2/5 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.008 | 5/3, 6/4, 2/4, 5/4, 4/3, 4/3 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.017 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.008 | 2/1, 1/1, 2/3, 3/2, 2/1, 3/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 6/5, 5/5, 4/6, 5/5, 5/5, 4/4 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.013 | 2/0, 3/0, 2/1, 2/0, 1/1, 1/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.013 | 4/2, 3/3, 3/3, 3/2, 5/2, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.000 | 0/0, 0/0, 0/0, 0/0, 0/0, 0/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.025 | 2/5, 2/5, 2/5, 2/5, 2/5, 2/5 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.017 | 6/4, 7/5, 3/6, 5/5, 6/3, 5/3 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | -0.004 | 2/2, 2/3, 1/3, 0/3, 5/2, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.013 | 3/1, 2/2, 3/3, 3/2, 3/1, 3/1 |
| best_six[act_fb] − pooled6_vote | +0.008 | 6/4, 5/5, 4/3, 4/2, 5/4, 4/3 |
| best_six[act_fb] − best_original[act_fb] | +0.008 | 1/0, 1/1, 1/0, 1/0, 2/1, 2/0 |
| best_original[act_fb] − original_vote | +0.025 | 7/3, 5/2, 5/2, 5/1, 5/2, 6/4 |
| gate0[act_fb] − original_vote | +0.000 | 0/0, 0/1, 0/1, 0/0, 0/0, 0/0 |
| gate0[act_fb] − pooled6_vote | -0.025 | 2/5, 2/6, 2/6, 2/5, 2/5, 2/5 |
| best_six[act_fb] − best_six[out_lr[full]] | +0.021 | 9/6, 9/7, 7/7, 7/5, 8/4, 7/4 |
| best_original[act_fb] − best_original[out_lr[full]] | +0.017 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act_fb] − gate0[out_lr[full]] | +0.013 | 3/1, 2/3, 3/4, 3/2, 3/1, 3/1 |
| gate_tuned[act_fb] − original_vote | -0.013 | 5/7, 7/8, 0/2, 3/2, 3/5, 8/6 |

Origin of best-of-six picks (partition 0): out_lr[full]: 38 from revisions, 4 new options (1 correct); out_hgb[full]: 48 from revisions, 8 new options (1 correct); out_lr[of+ot]: 24 from revisions, 2 new options (0 correct); comb_lr[full+act]: 38 from revisions, 4 new options (1 correct); eq[act+out_lr_full]: 35 from revisions, 0 new options (0 correct); eq[act+out_lr_of+ot]: 25 from revisions, 0 new options (0 correct); act_fb: 33 from revisions, 1 new options (1 correct)

Gate, agent level (partition 0): gate0[out_lr[full]]: helpful_accepted 10, helpful_blocked 25, harmful_accepted 8, harmful_blocked 71, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 52; gate0[out_hgb[full]]: helpful_accepted 10, helpful_blocked 25, harmful_accepted 11, harmful_blocked 68, wrong_to_wrong_accepted 18, wrong_to_wrong_blocked 49; gate0[out_lr[of+ot]]: helpful_accepted 12, helpful_blocked 23, harmful_accepted 4, harmful_blocked 75, wrong_to_wrong_accepted 11, wrong_to_wrong_blocked 56; gate0[comb_lr[full+act]]: helpful_accepted 9, helpful_blocked 26, harmful_accepted 8, harmful_blocked 71, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 52; gate0[eq[act+out_lr_full]]: helpful_accepted 2, helpful_blocked 33, harmful_accepted 2, harmful_blocked 77, wrong_to_wrong_accepted 7, wrong_to_wrong_blocked 60; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 1, helpful_blocked 34, harmful_accepted 1, harmful_blocked 78, wrong_to_wrong_accepted 6, wrong_to_wrong_blocked 61; gate0[act_fb]: helpful_accepted 1, helpful_blocked 34, harmful_accepted 1, harmful_blocked 78, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 62; gate_tuned[act_fb]: helpful_accepted 19, helpful_blocked 16, harmful_accepted 27, harmful_blocked 52, wrong_to_wrong_accepted 30, wrong_to_wrong_blocked 37

Calibration of revised-state probabilities (median over partitions): act_fb Brier 0.196 (constant 0.254), ECE 0.062; comb_lr[full+act] Brier 0.179 (constant 0.254), ECE 0.072; eq[act+out_lr_full] Brier 0.174 (constant 0.254), ECE 0.060; eq[act+out_lr_of+ot] Brier 0.174 (constant 0.254), ECE 0.059; out_hgb[full] Brier 0.195 (constant 0.254), ECE 0.053; out_lr[full] Brier 0.178 (constant 0.254), ECE 0.071; out_lr[of+ot] Brier 0.177 (constant 0.254), ECE 0.049

### misleading_one|eligible (109 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act | 0.672 (0.65–0.72) | 0.874 (0.87–0.88) | 0.670 (0.64–0.74) | 0.885 (0.85–0.94) | 0.820 (0.81–0.85) | 0.894 (0.86–0.93) |
| comb_lr[full+act] | 0.769 (0.74–0.79) | 0.849 (0.83–0.86) | 0.640 (0.60–0.72) | 0.776 (0.77–0.82) | 0.763 (0.75–0.79) | 0.756 (0.72–0.78) |
| eq[act+out_lr_full] | 0.765 (0.75–0.78) | 0.884 (0.87–0.89) | 0.690 (0.64–0.76) | 0.885 (0.82–0.90) | 0.816 (0.79–0.83) | 0.841 (0.82–0.91) |
| eq[act+out_lr_of+ot] | 0.747 (0.72–0.78) | 0.874 (0.86–0.89) | 0.670 (0.60–0.72) | 0.878 (0.86–0.90) | 0.816 (0.79–0.83) | 0.871 (0.83–0.88) |
| out_hgb[full] | 0.688 (0.64–0.74) | 0.811 (0.77–0.83) | 0.600 (0.56–0.68) | 0.782 (0.71–0.82) | 0.757 (0.71–0.78) | 0.755 (0.70–0.81) |
| out_lr[full] | 0.764 (0.73–0.78) | 0.842 (0.82–0.85) | 0.650 (0.60–0.74) | 0.763 (0.73–0.78) | 0.751 (0.74–0.78) | 0.738 (0.71–0.77) |
| out_lr[of+ot] | 0.744 (0.71–0.79) | 0.829 (0.82–0.85) | 0.600 (0.56–0.66) | 0.788 (0.76–0.82) | 0.770 (0.75–0.79) | 0.752 (0.72–0.79) |

Counts (partition 0): state_auc_pre 5052 pairs; state_auc_post 5343 pairs; panel_pre 50 pairs / 25 questions; panel_post 78 pairs / 39 questions; six 331 pairs / 50 questions; revision_auc 267 pairs / 50 questions; helpful 26, harmful 55, wrong-to-wrong 61

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act - out_lr[full] | -0.094 (0/1) | +0.034 (0/0) | +0.040 (0/0) | +0.122 (5/0) | +0.073 (0/0) | +0.165 (4/0) |
| act - out_hgb[full] | -0.027 (0/0) | +0.068 (4/0) | +0.080 (0/0) | +0.103 (3/0) | +0.071 (1/0) | +0.138 (1/0) |
| comb_lr[full+act] - out_lr[full] | +0.006 (0/0) | +0.010 (4/0) | +0.000 (0/0) | +0.019 (0/0) | +0.009 (1/0) | +0.017 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | +0.046 (5/0) | +0.080 (1/0) | +0.115 (6/0) | +0.059 (2/0) | +0.115 (6/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.009 (0/0) | +0.043 (6/0) | +0.040 (1/0) | +0.090 (6/0) | +0.039 (3/0) | +0.109 (6/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act] | 68 | 67 | 67 | 68 | 67 | 66 |
| best_original[comb_lr[full+act]] | 66 | 65 | 65 | 67 | 64 | 64 |
| best_original[eq[act+out_lr_full]] | 68 | 67 | 65 | 67 | 66 | 66 |
| best_original[eq[act+out_lr_of+ot]] | 67 | 65 | 64 | 66 | 68 | 66 |
| best_original[out_hgb[full]] | 65 | 65 | 64 | 67 | 65 | 65 |
| best_original[out_lr[full]] | 66 | 65 | 66 | 68 | 64 | 64 |
| best_original[out_lr[of+ot]] | 66 | 64 | 64 | 67 | 65 | 65 |
| best_six[act] | 70 | 68 | 68 | 69 | 66 | 68 |
| best_six[comb_lr[full+act]] | 65 | 66 | 67 | 67 | 65 | 65 |
| best_six[eq[act+out_lr_full]] | 69 | 68 | 67 | 68 | 67 | 67 |
| best_six[eq[act+out_lr_of+ot]] | 69 | 67 | 67 | 68 | 69 | 68 |
| best_six[out_hgb[full]] | 63 | 64 | 63 | 65 | 65 | 65 |
| best_six[out_lr[full]] | 66 | 64 | 67 | 66 | 65 | 64 |
| best_six[out_lr[of+ot]] | 68 | 67 | 67 | 69 | 68 | 68 |
| gate0[act] | 66 | 67 | 66 | 67 | 67 | 67 |
| gate0[comb_lr[full+act]] | 63 | 64 | 64 | 67 | 64 | 64 |
| gate0[eq[act+out_lr_full]] | 64 | 65 | 64 | 66 | 64 | 65 |
| gate0[eq[act+out_lr_of+ot]] | 66 | 65 | 66 | 65 | 66 | 67 |
| gate0[out_hgb[full]] | 63 | 65 | 65 | 66 | 64 | 66 |
| gate0[out_lr[full]] | 63 | 64 | 64 | 66 | 63 | 64 |
| gate0[out_lr[of+ot]] | 65 | 65 | 65 | 66 | 65 | 65 |
| gate_tuned[act] | 65 | 69 | 66 | 69 | 68 | 68 |
| original_vote | 65 | 65 | 65 | 65 | 65 | 65 |
| pooled6_vote | 66 | 66 | 66 | 66 | 66 | 66 |
| revised_vote | 61 | 61 | 61 | 61 | 61 | 61 |

Random tie-break expectation: original 66.33, revised 58.67, pooled six 66.50. Questions with a correct round-0 answer 75, with a correct revised answer 72, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[act] − pooled6_vote | +0.018 | 5/1, 3/1, 3/1, 3/0, 2/2, 4/2 |
| best_six[act] − best_original[act] | +0.009 | 2/0, 1/0, 1/0, 1/0, 0/1, 2/0 |
| best_original[act] − original_vote | +0.018 | 6/3, 4/2, 4/2, 4/1, 4/2, 5/4 |
| gate0[act] − original_vote | +0.018 | 2/1, 2/0, 1/0, 2/0, 2/0, 2/0 |
| gate0[act] − pooled6_vote | +0.009 | 1/1, 1/0, 0/0, 1/0, 1/0, 1/0 |
| best_six[act] − best_six[out_lr[full]] | +0.032 | 6/2, 6/2, 4/3, 5/2, 4/3, 7/3 |
| best_original[act] − best_original[out_lr[full]] | +0.018 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act] − gate0[out_lr[full]] | +0.028 | 4/1, 4/1, 2/0, 3/2, 4/0, 4/1 |
| best_six[out_lr[full]] − pooled6_vote | -0.005 | 3/3, 2/4, 2/1, 3/3, 1/2, 2/4 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 2/2, 1/2, 2/1, 0/2, 1/0, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.005 | 2/1, 1/1, 2/1, 3/0, 1/2, 1/2 |
| gate0[out_lr[full]] − original_vote | -0.009 | 0/2, 1/2, 0/1, 2/1, 0/2, 1/2 |
| gate0[out_lr[full]] − pooled6_vote | -0.018 | 0/3, 1/3, 0/2, 2/2, 0/3, 1/3 |
| best_six[out_hgb[full]] − pooled6_vote | -0.014 | 2/5, 3/5, 3/6, 3/4, 3/4, 2/3 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.009 | 0/2, 3/4, 1/2, 0/2, 1/1, 1/1 |
| best_original[out_hgb[full]] − original_vote | +0.000 | 3/3, 2/2, 3/4, 3/1, 2/2, 2/2 |
| gate0[out_hgb[full]] − original_vote | +0.000 | 1/3, 3/3, 2/2, 2/1, 1/2, 2/1 |
| gate0[out_hgb[full]] − pooled6_vote | -0.009 | 1/4, 3/4, 2/3, 2/2, 1/3, 1/1 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.005 | 1/4, 4/4, 2/6, 1/2, 2/2, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | -0.005 | 2/3, 2/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.009 | 1/1, 2/1, 2/1, 1/1, 1/0, 2/0 |
| best_six[out_lr[of+ot]] − pooled6_vote | +0.018 | 2/0, 1/0, 2/1, 3/0, 2/0, 2/0 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.028 | 2/0, 3/0, 3/0, 2/0, 3/0, 3/0 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 2/1, 1/2, 2/3, 3/1, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 1/1, 1/1, 1/1, 2/1, 1/1, 1/1 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.009 | 1/2, 1/2, 1/2, 2/2, 1/2, 1/2 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | +0.028 | 3/1, 4/1, 1/1, 3/0, 3/0, 5/1 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.005 | 0/0, 1/2, 1/3, 0/1, 2/1, 1/0 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.009 | 3/1, 2/1, 2/1, 1/1, 2/0, 3/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.005 | 2/3, 2/2, 2/1, 3/2, 1/2, 2/3 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.009 | 1/2, 2/1, 2/0, 2/2, 1/0, 3/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.000 | 2/1, 1/1, 2/2, 2/0, 1/2, 1/2 |
| gate0[comb_lr[full+act]] − original_vote | -0.009 | 0/2, 1/2, 1/2, 2/0, 0/1, 1/2 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.018 | 0/3, 1/3, 1/3, 2/1, 0/2, 1/3 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.005 | 0/1, 2/0, 1/1, 1/0, 0/0, 1/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/1, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 1/1, 1/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.014 | 3/0, 2/0, 1/0, 2/0, 1/0, 1/0 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.009 | 1/0, 1/0, 2/0, 1/0, 1/0, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.014 | 3/0, 2/0, 1/1, 2/0, 1/0, 2/1 |
| gate0[eq[act+out_lr_full]] − original_vote | -0.005 | 0/1, 1/1, 0/1, 1/0, 0/1, 0/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.014 | 0/2, 1/2, 0/2, 1/1, 0/2, 0/1 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.023 | 3/0, 4/0, 1/1, 3/1, 3/1, 4/1 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.018 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.009 | 1/0, 1/0, 0/0, 1/1, 1/0, 2/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.018 | 3/0, 2/1, 2/1, 2/0, 3/0, 2/0 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.018 | 2/0, 2/0, 3/0, 2/0, 1/0, 2/0 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.009 | 3/1, 2/2, 2/3, 2/1, 4/1, 2/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.009 | 1/0, 0/0, 1/0, 0/0, 1/0, 2/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 1/1, 0/1, 1/1, 0/1, 1/1, 1/0 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.028 | 3/0, 4/1, 1/1, 3/1, 4/0, 5/1 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | +0.005 | 2/1, 2/2, 1/3, 0/2, 5/1, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.023 | 3/0, 2/1, 2/0, 1/2, 3/0, 4/1 |
| gate_tuned[act] − original_vote | +0.028 | 1/1, 4/0, 2/1, 4/0, 3/0, 4/1 |

Origin of best-of-six picks (partition 0): act: 66 from revisions, 0 new options (0 correct); out_lr[full]: 49 from revisions, 4 new options (0 correct); out_hgb[full]: 44 from revisions, 2 new options (0 correct); out_lr[of+ot]: 38 from revisions, 2 new options (0 correct); comb_lr[full+act]: 48 from revisions, 4 new options (0 correct); eq[act+out_lr_full]: 60 from revisions, 1 new options (0 correct); eq[act+out_lr_of+ot]: 58 from revisions, 2 new options (0 correct)

Gate, agent level (partition 0): gate0[act]: helpful_accepted 9, helpful_blocked 17, harmful_accepted 2, harmful_blocked 53, wrong_to_wrong_accepted 1, wrong_to_wrong_blocked 60; gate_tuned[act]: helpful_accepted 17, helpful_blocked 9, harmful_accepted 16, harmful_blocked 39, wrong_to_wrong_accepted 20, wrong_to_wrong_blocked 41; gate0[out_lr[full]]: helpful_accepted 13, helpful_blocked 13, harmful_accepted 8, harmful_blocked 47, wrong_to_wrong_accepted 14, wrong_to_wrong_blocked 47; gate0[out_hgb[full]]: helpful_accepted 11, helpful_blocked 15, harmful_accepted 9, harmful_blocked 46, wrong_to_wrong_accepted 17, wrong_to_wrong_blocked 44; gate0[out_lr[of+ot]]: helpful_accepted 12, helpful_blocked 14, harmful_accepted 5, harmful_blocked 50, wrong_to_wrong_accepted 16, wrong_to_wrong_blocked 45; gate0[comb_lr[full+act]]: helpful_accepted 13, helpful_blocked 13, harmful_accepted 7, harmful_blocked 48, wrong_to_wrong_accepted 14, wrong_to_wrong_blocked 47; gate0[eq[act+out_lr_full]]: helpful_accepted 8, helpful_blocked 18, harmful_accepted 2, harmful_blocked 53, wrong_to_wrong_accepted 4, wrong_to_wrong_blocked 57; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 10, helpful_blocked 16, harmful_accepted 0, harmful_blocked 55, wrong_to_wrong_accepted 7, wrong_to_wrong_blocked 54

Calibration of revised-state probabilities (median over partitions): act Brier 0.155 (constant 0.251), ECE 0.093; comb_lr[full+act] Brier 0.167 (constant 0.251), ECE 0.048; eq[act+out_lr_full] Brier 0.146 (constant 0.251), ECE 0.069; eq[act+out_lr_of+ot] Brier 0.152 (constant 0.251), ECE 0.071; out_hgb[full] Brier 0.190 (constant 0.251), ECE 0.063; out_lr[full] Brier 0.172 (constant 0.251), ECE 0.070; out_lr[of+ot] Brier 0.177 (constant 0.251), ECE 0.074

### misleading_one|all_fallback (120 questions; absent states 18)

Ranking (median over 6 partitions, range):

| scorer | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb | 0.681 (0.67–0.72) | 0.867 (0.86–0.88) | 0.653 (0.63–0.71) | 0.893 (0.86–0.94) | 0.805 (0.80–0.83) | 0.877 (0.85–0.90) |
| comb_lr[full+act] | 0.773 (0.74–0.79) | 0.858 (0.84–0.87) | 0.637 (0.61–0.69) | 0.786 (0.79–0.83) | 0.760 (0.74–0.78) | 0.751 (0.71–0.78) |
| eq[act+out_lr_full] | 0.770 (0.75–0.78) | 0.885 (0.87–0.89) | 0.685 (0.65–0.74) | 0.887 (0.83–0.90) | 0.804 (0.79–0.82) | 0.832 (0.80–0.87) |
| eq[act+out_lr_of+ot] | 0.753 (0.72–0.78) | 0.875 (0.86–0.89) | 0.645 (0.60–0.68) | 0.881 (0.87–0.90) | 0.797 (0.79–0.81) | 0.842 (0.81–0.85) |
| out_hgb[full] | 0.700 (0.64–0.74) | 0.823 (0.79–0.85) | 0.613 (0.58–0.73) | 0.798 (0.73–0.83) | 0.756 (0.72–0.78) | 0.759 (0.70–0.82) |
| out_lr[full] | 0.769 (0.73–0.78) | 0.853 (0.83–0.86) | 0.653 (0.60–0.71) | 0.774 (0.75–0.80) | 0.748 (0.74–0.78) | 0.740 (0.69–0.77) |
| out_lr[of+ot] | 0.745 (0.71–0.78) | 0.843 (0.83–0.86) | 0.581 (0.55–0.63) | 0.792 (0.76–0.82) | 0.753 (0.74–0.77) | 0.729 (0.70–0.76) |

Counts (partition 0): state_auc_pre 6206 pairs; state_auc_post 6429 pairs; panel_pre 62 pairs / 31 questions; panel_post 84 pairs / 42 questions; six 375 pairs / 56 questions; revision_auc 331 pairs / 56 questions; helpful 30, harmful 59, wrong-to-wrong 75

Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):

| difference | state_auc_pre | state_auc_post | panel_pre | panel_post | six | revision_auc |
|---|---|---|---|---|---|---|
| act_fb - out_lr[full] | -0.089 (0/1) | +0.019 (0/0) | +0.016 (0/0) | +0.113 (5/0) | +0.057 (0/0) | +0.138 (3/0) |
| act_fb - out_hgb[full] | -0.024 (0/0) | +0.051 (2/0) | +0.040 (0/0) | +0.095 (3/0) | +0.053 (1/0) | +0.115 (1/0) |
| comb_lr[full+act] - out_lr[full] | +0.007 (0/0) | +0.008 (3/0) | +0.000 (0/0) | +0.018 (0/0) | +0.008 (1/0) | +0.013 (0/0) |
| eq[act+out_lr_full] - out_lr[full] | -0.001 (0/0) | +0.032 (3/0) | +0.056 (0/0) | +0.107 (6/0) | +0.056 (3/0) | +0.100 (6/0) |
| eq[act+out_lr_of+ot] - out_lr[of+ot] | +0.011 (0/0) | +0.032 (3/0) | +0.048 (1/0) | +0.095 (6/0) | +0.043 (5/0) | +0.101 (6/0) |

Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):

| rule | 0 | 1 | 2 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|
| best_original[act_fb] | 73 | 72 | 72 | 73 | 72 | 71 |
| best_original[comb_lr[full+act]] | 71 | 70 | 69 | 72 | 69 | 69 |
| best_original[eq[act+out_lr_full]] | 73 | 72 | 70 | 72 | 71 | 71 |
| best_original[eq[act+out_lr_of+ot]] | 71 | 69 | 69 | 70 | 72 | 71 |
| best_original[out_hgb[full]] | 70 | 71 | 69 | 72 | 70 | 70 |
| best_original[out_lr[full]] | 71 | 70 | 71 | 73 | 69 | 69 |
| best_original[out_lr[of+ot]] | 70 | 68 | 68 | 71 | 69 | 69 |
| best_six[act_fb] | 75 | 73 | 73 | 74 | 71 | 73 |
| best_six[comb_lr[full+act]] | 70 | 71 | 71 | 72 | 70 | 70 |
| best_six[eq[act+out_lr_full]] | 74 | 73 | 72 | 73 | 72 | 72 |
| best_six[eq[act+out_lr_of+ot]] | 73 | 71 | 71 | 72 | 73 | 72 |
| best_six[out_hgb[full]] | 68 | 70 | 68 | 69 | 70 | 70 |
| best_six[out_lr[full]] | 71 | 69 | 72 | 71 | 70 | 69 |
| best_six[out_lr[of+ot]] | 72 | 71 | 71 | 73 | 72 | 72 |
| gate0[act_fb] | 71 | 71 | 71 | 72 | 72 | 72 |
| gate0[comb_lr[full+act]] | 68 | 68 | 69 | 72 | 69 | 68 |
| gate0[eq[act+out_lr_full]] | 69 | 69 | 69 | 71 | 69 | 70 |
| gate0[eq[act+out_lr_of+ot]] | 71 | 69 | 71 | 69 | 71 | 72 |
| gate0[out_hgb[full]] | 68 | 70 | 70 | 71 | 69 | 71 |
| gate0[out_lr[full]] | 68 | 68 | 69 | 71 | 68 | 68 |
| gate0[out_lr[of+ot]] | 69 | 70 | 69 | 70 | 69 | 69 |
| gate_tuned[act_fb] | 69 | 73 | 71 | 74 | 72 | 72 |
| original_vote | 69 | 69 | 69 | 69 | 69 | 69 |
| pooled6_vote | 72 | 72 | 72 | 72 | 72 | 72 |
| revised_vote | 66 | 66 | 66 | 66 | 66 | 66 |

Random tie-break expectation: original 70.67, revised 63.67, pooled six 72.00. Questions with a correct round-0 answer 82, with a correct revised answer 77, correct only after revision 2.

Paired decision differences (median; wins−losses per partition):

| comparison | median diff | wins/losses by partition |
|---|---|---|
| best_six[out_lr[full]] − pooled6_vote | -0.013 | 3/4, 2/5, 2/2, 3/4, 1/3, 2/5 |
| best_six[out_lr[full]] − best_original[out_lr[full]] | +0.000 | 2/2, 1/2, 2/1, 0/2, 1/0, 2/2 |
| best_original[out_lr[full]] − original_vote | +0.013 | 3/1, 2/1, 3/1, 4/0, 2/2, 2/2 |
| gate0[out_lr[full]] − original_vote | -0.008 | 1/2, 1/2, 1/1, 3/1, 1/2, 2/3 |
| gate0[out_lr[full]] − pooled6_vote | -0.033 | 0/4, 1/5, 0/3, 2/3, 0/4, 1/5 |
| best_six[out_hgb[full]] − pooled6_vote | -0.021 | 2/6, 3/5, 3/7, 3/6, 3/5, 2/4 |
| best_six[out_hgb[full]] − best_original[out_hgb[full]] | -0.008 | 0/2, 3/4, 1/2, 0/3, 1/1, 1/1 |
| best_original[out_hgb[full]] − original_vote | +0.008 | 4/3, 4/2, 4/4, 4/1, 3/2, 3/2 |
| gate0[out_hgb[full]] − original_vote | +0.008 | 2/3, 4/3, 3/2, 3/1, 2/2, 3/1 |
| gate0[out_hgb[full]] − pooled6_vote | -0.017 | 1/5, 3/5, 2/4, 2/3, 1/4, 1/2 |
| best_six[out_hgb[full]] − best_six[out_lr[full]] | -0.008 | 1/4, 5/4, 2/6, 1/3, 2/2, 2/1 |
| best_original[out_hgb[full]] − best_original[out_lr[full]] | +0.000 | 2/3, 3/2, 1/3, 0/1, 2/1, 2/1 |
| gate0[out_hgb[full]] − gate0[out_lr[full]] | +0.008 | 1/1, 3/1, 2/1, 1/1, 1/0, 3/0 |
| best_six[out_lr[of+ot]] − pooled6_vote | +0.000 | 2/2, 1/2, 2/3, 3/2, 2/2, 2/2 |
| best_six[out_lr[of+ot]] − best_original[out_lr[of+ot]] | +0.025 | 2/0, 3/0, 3/0, 2/0, 3/0, 3/0 |
| best_original[out_lr[of+ot]] − original_vote | +0.000 | 3/2, 2/3, 3/4, 4/2, 3/3, 2/2 |
| gate0[out_lr[of+ot]] − original_vote | +0.000 | 2/2, 2/1, 2/2, 3/2, 2/2, 2/2 |
| gate0[out_lr[of+ot]] − pooled6_vote | -0.025 | 1/4, 1/3, 1/4, 2/4, 1/4, 1/4 |
| best_six[out_lr[of+ot]] − best_six[out_lr[full]] | +0.017 | 3/2, 4/2, 1/2, 3/1, 3/1, 5/2 |
| best_original[out_lr[of+ot]] − best_original[out_lr[full]] | -0.013 | 0/1, 1/3, 1/4, 0/2, 2/2, 1/1 |
| gate0[out_lr[of+ot]] − gate0[out_lr[full]] | +0.008 | 3/2, 3/1, 2/2, 1/2, 2/1, 3/2 |
| best_six[comb_lr[full+act]] − pooled6_vote | -0.013 | 2/4, 2/3, 2/3, 3/3, 1/3, 2/4 |
| best_six[comb_lr[full+act]] − best_original[comb_lr[full+act]] | +0.008 | 1/2, 2/1, 2/0, 2/2, 1/0, 3/2 |
| best_original[comb_lr[full+act]] − original_vote | +0.004 | 3/1, 2/1, 3/3, 3/0, 2/2, 2/2 |
| gate0[comb_lr[full+act]] − original_vote | -0.004 | 1/2, 1/2, 2/2, 3/0, 1/1, 2/3 |
| gate0[comb_lr[full+act]] − pooled6_vote | -0.029 | 0/4, 1/5, 1/4, 2/2, 0/3, 1/5 |
| best_six[comb_lr[full+act]] − best_six[out_lr[full]] | +0.004 | 0/1, 2/0, 1/2, 1/0, 0/0, 1/0 |
| best_original[comb_lr[full+act]] − best_original[out_lr[full]] | +0.000 | 0/0, 0/0, 0/2, 0/1, 0/0, 0/0 |
| gate0[comb_lr[full+act]] − gate0[out_lr[full]] | +0.000 | 0/0, 0/0, 1/1, 1/0, 1/0, 0/0 |
| best_six[eq[act+out_lr_full]] − pooled6_vote | +0.004 | 3/1, 2/1, 1/1, 2/1, 1/1, 1/1 |
| best_six[eq[act+out_lr_full]] − best_original[eq[act+out_lr_full]] | +0.008 | 1/0, 1/0, 2/0, 1/0, 1/0, 1/0 |
| best_original[eq[act+out_lr_full]] − original_vote | +0.021 | 4/0, 3/0, 2/1, 3/0, 2/0, 3/1 |
| gate0[eq[act+out_lr_full]] − original_vote | +0.000 | 1/1, 1/1, 1/1, 2/0, 1/1, 1/0 |
| gate0[eq[act+out_lr_full]] − pooled6_vote | -0.025 | 0/3, 1/4, 0/3, 1/2, 0/3, 0/2 |
| best_six[eq[act+out_lr_full]] − best_six[out_lr[full]] | +0.021 | 3/0, 4/0, 1/1, 3/1, 3/1, 4/1 |
| best_original[eq[act+out_lr_full]] − best_original[out_lr[full]] | +0.017 | 2/0, 2/0, 0/1, 0/1, 3/1, 4/2 |
| gate0[eq[act+out_lr_full]] − gate0[out_lr[full]] | +0.008 | 1/0, 1/0, 0/0, 1/1, 1/0, 3/1 |
| best_six[eq[act+out_lr_of+ot]] − pooled6_vote | +0.000 | 3/2, 2/3, 2/3, 2/2, 3/2, 2/2 |
| best_six[eq[act+out_lr_of+ot]] − best_original[eq[act+out_lr_of+ot]] | +0.017 | 2/0, 2/0, 3/1, 2/0, 1/0, 2/1 |
| best_original[eq[act+out_lr_of+ot]] − original_vote | +0.013 | 4/2, 3/3, 3/3, 3/2, 5/2, 3/1 |
| gate0[eq[act+out_lr_of+ot]] − original_vote | +0.017 | 2/0, 0/0, 2/0, 1/1, 2/0, 3/0 |
| gate0[eq[act+out_lr_of+ot]] − pooled6_vote | -0.008 | 1/2, 0/3, 1/2, 0/3, 1/2, 1/1 |
| best_six[eq[act+out_lr_of+ot]] − best_six[out_lr[full]] | +0.017 | 3/1, 4/2, 1/2, 3/2, 4/1, 5/2 |
| best_original[eq[act+out_lr_of+ot]] − best_original[out_lr[full]] | -0.004 | 2/2, 2/3, 1/3, 0/3, 5/2, 4/2 |
| gate0[eq[act+out_lr_of+ot]] − gate0[out_lr[full]] | +0.021 | 3/0, 2/1, 2/0, 1/3, 3/0, 5/1 |
| best_six[act_fb] − pooled6_vote | +0.008 | 5/2, 3/2, 3/2, 3/1, 2/3, 4/3 |
| best_six[act_fb] − best_original[act_fb] | +0.008 | 2/0, 1/0, 1/0, 1/0, 0/1, 2/0 |
| best_original[act_fb] − original_vote | +0.025 | 7/3, 5/2, 5/2, 5/1, 5/2, 6/4 |
| gate0[act_fb] − original_vote | +0.021 | 3/1, 2/0, 2/0, 3/0, 3/0, 3/0 |
| gate0[act_fb] − pooled6_vote | -0.004 | 1/2, 1/2, 0/1, 1/1, 1/1, 1/1 |
| best_six[act_fb] − best_six[out_lr[full]] | +0.029 | 6/2, 6/2, 4/3, 5/2, 4/3, 7/3 |
| best_original[act_fb] − best_original[out_lr[full]] | +0.017 | 6/4, 5/3, 5/4, 3/3, 6/3, 7/5 |
| gate0[act_fb] − gate0[out_lr[full]] | +0.025 | 4/1, 4/1, 2/0, 3/2, 4/0, 5/1 |
| gate_tuned[act_fb] − original_vote | +0.025 | 2/2, 5/1, 3/1, 5/0, 4/1, 5/2 |

Origin of best-of-six picks (partition 0): out_lr[full]: 53 from revisions, 4 new options (0 correct); out_hgb[full]: 46 from revisions, 2 new options (0 correct); out_lr[of+ot]: 41 from revisions, 3 new options (0 correct); comb_lr[full+act]: 51 from revisions, 4 new options (0 correct); eq[act+out_lr_full]: 62 from revisions, 1 new options (0 correct); eq[act+out_lr_of+ot]: 60 from revisions, 3 new options (0 correct); act_fb: 69 from revisions, 1 new options (0 correct)

Gate, agent level (partition 0): gate0[out_lr[full]]: helpful_accepted 15, helpful_blocked 15, harmful_accepted 10, harmful_blocked 49, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 60; gate0[out_hgb[full]]: helpful_accepted 13, helpful_blocked 17, harmful_accepted 10, harmful_blocked 49, wrong_to_wrong_accepted 19, wrong_to_wrong_blocked 56; gate0[out_lr[of+ot]]: helpful_accepted 14, helpful_blocked 16, harmful_accepted 7, harmful_blocked 52, wrong_to_wrong_accepted 17, wrong_to_wrong_blocked 58; gate0[comb_lr[full+act]]: helpful_accepted 15, helpful_blocked 15, harmful_accepted 9, harmful_blocked 50, wrong_to_wrong_accepted 15, wrong_to_wrong_blocked 60; gate0[eq[act+out_lr_full]]: helpful_accepted 10, helpful_blocked 20, harmful_accepted 4, harmful_blocked 55, wrong_to_wrong_accepted 5, wrong_to_wrong_blocked 70; gate0[eq[act+out_lr_of+ot]]: helpful_accepted 12, helpful_blocked 18, harmful_accepted 2, harmful_blocked 57, wrong_to_wrong_accepted 7, wrong_to_wrong_blocked 68; gate0[act_fb]: helpful_accepted 11, helpful_blocked 19, harmful_accepted 4, harmful_blocked 55, wrong_to_wrong_accepted 4, wrong_to_wrong_blocked 71; gate_tuned[act_fb]: helpful_accepted 20, helpful_blocked 10, harmful_accepted 19, harmful_blocked 40, wrong_to_wrong_accepted 29, wrong_to_wrong_blocked 46

Calibration of revised-state probabilities (median over partitions): act_fb Brier 0.158 (constant 0.252), ECE 0.089; comb_lr[full+act] Brier 0.163 (constant 0.252), ECE 0.053; eq[act+out_lr_full] Brier 0.146 (constant 0.252), ECE 0.070; eq[act+out_lr_of+ot] Brier 0.150 (constant 0.252), ECE 0.077; out_hgb[full] Brier 0.185 (constant 0.252), ECE 0.064; out_lr[full] Brier 0.166 (constant 0.252), ECE 0.063; out_lr[of+ot] Brier 0.170 (constant 0.252), ECE 0.077

Alignment: held-out z SD (1 = reference scale), median (range) over folds × partitions: act 0.99 (0.77–1.09); out_lr[of+ot] 0.92 (0.79–1.23); out_lr[full] 0.98 (0.83–1.26); out_hgb[full] 0.92 (0.80–1.02); comb_lr[full+act] 1.00 (0.82–1.25); eq[act+out_lr_full] 1.83 (1.51–2.10); eq[act+out_lr_of+ot] 1.75 (1.38–2.08); act_fb 0.98 (0.77–1.11)

Orthogonality, act: held-out R² from all output features 0.69 (0.54–0.81); largest |r| with one output feature 0.66 (0.57–0.75).

