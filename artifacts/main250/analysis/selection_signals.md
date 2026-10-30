# main250 — selection signals (EXPLORATORY)

**Exploratory.** 200 questions; brackets are 95% percentile intervals from 10,000 question resamples (seed 0). Signal definitions, values and rules were fixed before computing (see `scripts/main250_signals.py`). Scores: exact expected accuracy under random tie-breaking.

## Signals per state

| states | signal | available | median top-1 / value | share > 0.99 | AUROC for correctness |
|---|---|---|---|---|---|
| pool (pre) | (i) before-stated-answer distribution | 99.2% | 1.0000 | 97.2% | 0.667 [0.624, 0.710] |
| pool (pre) | (ii) response mean log-prob (as exp) | 100.0% | 0.7621 | 0.0% | 0.648 [0.585, 0.706] |
| pool (pre) | (iii) verbalized confidence | 99.2% | 0.9500 | 16.5% | 0.674 [0.631, 0.716] |
| debate3 (post) | (i) before-stated-answer distribution | 99.8% | 1.0000 | 99.2% | 0.562 [0.532, 0.596] |
| debate3 (post) | (ii) response mean log-prob (as exp) | 100.0% | 0.6915 | 0.0% | 0.632 [0.572, 0.689] |
| debate3 (post) | (iii) verbalized confidence | 99.8% | 0.9800 | 29.9% | 0.788 [0.733, 0.839] |
| debate3 (post) | (iv) peer-exposure readout | 100.0% | 1.0000 | 91.7% | 0.771 [0.708, 0.828] |
| structured3 (post) | (i) before-stated-answer distribution | 100.0% | 1.0000 | 99.2% | 0.571 [0.534, 0.610] |
| structured3 (post) | (ii) response mean log-prob (as exp) | 100.0% | 0.6825 | 0.0% | 0.600 [0.534, 0.666] |
| structured3 (post) | (iii) verbalized confidence | 100.0% | 0.9800 | 24.2% | 0.788 [0.737, 0.837] |
| structured3 (post) | (iv) peer-exposure readout | 100.0% | 0.9998 | 87.8% | 0.807 [0.753, 0.857] |
| debate5 (post) | (i) before-stated-answer distribution | 100.0% | 1.0000 | 99.3% | 0.593 [0.561, 0.628] |
| debate5 (post) | (ii) response mean log-prob (as exp) | 100.0% | 0.6737 | 0.0% | 0.643 [0.585, 0.696] |
| debate5 (post) | (iii) verbalized confidence | 100.0% | 0.9900 | 45.4% | 0.809 [0.756, 0.859] |
| debate5 (post) | (iv) peer-exposure readout | 100.0% | 1.0000 | 89.2% | 0.787 [0.728, 0.843] |
| structured5 (post) | (i) before-stated-answer distribution | 100.0% | 1.0000 | 99.0% | 0.594 [0.558, 0.629] |
| structured5 (post) | (ii) response mean log-prob (as exp) | 100.0% | 0.6850 | 0.0% | 0.584 [0.519, 0.649] |
| structured5 (post) | (iii) verbalized confidence | 100.0% | 0.9800 | 27.9% | 0.789 [0.743, 0.832] |
| structured5 (post) | (iv) peer-exposure readout | 100.0% | 0.9999 | 87.8% | 0.818 [0.766, 0.866] |

Verbalized-confidence parse rate = its 'available' column. AUROC scores each signal's value for the answer it accompanies (for (iv): the exposure probability of the agent's post answer) against that answer's correctness.

## Rules against plurality

### pool first 3

Plurality 0.667 [0.602, 0.730]; coverage 0.750 [0.690, 0.805]; gap 0.083 [0.048, 0.122]; non-unanimous questions: 67.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.670 [0.605, 0.735] | 0.003 [-0.008, 0.017] | 7 | 0.04 [-0.11, 0.21] | 0.010 [-0.025, 0.050] | 0.04 [-0.11, 0.21] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.662 [0.595, 0.728] | -0.004 [-0.014, 0.006] | 7 | -0.05 [-0.17, 0.07] | -0.012 [-0.040, 0.017] | -0.05 [-0.17, 0.07] |
| (i) before-stated-answer distribution | c_highest_signal | 0.664 [0.599, 0.726] | -0.003 [-0.033, 0.028] | 27 | -0.03 [-0.54, 0.29] | -0.007 [-0.097, 0.085] | -0.03 [-0.51, 0.29] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.675 [0.610, 0.740] | 0.008 [-0.005, 0.023] | 7 | 0.10 [-0.06, 0.29] | 0.025 [-0.015, 0.070] | 0.10 [-0.06, 0.29] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.695 [0.630, 0.755] | 0.028 [-0.008, 0.065] | 19 | 0.34 [-0.14, 0.66] | 0.085 [-0.020, 0.194] | 0.34 [-0.11, 0.66] |
| (iii) verbalized confidence | b_weighted_plurality | 0.665 [0.598, 0.729] | -0.002 [-0.011, 0.008] | 5 | -0.02 [-0.15, 0.10] | -0.005 [-0.032, 0.025] | -0.02 [-0.15, 0.10] |
| (iii) verbalized confidence | c_highest_signal | 0.664 [0.603, 0.724] | -0.003 [-0.028, 0.023] | 30 | -0.03 [-0.46, 0.23] | -0.007 [-0.082, 0.072] | -0.03 [-0.46, 0.23] |

### pool first 5

Plurality 0.657 [0.591, 0.721]; coverage 0.775 [0.715, 0.830]; gap 0.118 [0.075, 0.164]; non-unanimous questions: 79.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.655 [0.590, 0.720] | -0.002 [-0.005, 0.000] | 2 | -0.02 [-0.04, 0.00] | -0.005 [-0.013, 0.000] | -0.02 [-0.04, 0.00] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.655 [0.590, 0.720] | -0.002 [-0.005, 0.000] | 2 | -0.02 [-0.04, 0.00] | -0.005 [-0.013, 0.000] | -0.02 [-0.04, 0.00] |
| (i) before-stated-answer distribution | c_highest_signal | 0.670 [0.607, 0.731] | 0.013 [-0.017, 0.045] | 32 | 0.11 [-0.19, 0.33] | 0.034 [-0.045, 0.115] | 0.11 [-0.19, 0.33] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.660 [0.595, 0.725] | 0.003 [-0.003, 0.012] | 2 | 0.03 [-0.02, 0.11] | 0.008 [-0.008, 0.030] | 0.03 [-0.02, 0.11] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.680 [0.615, 0.745] | 0.023 [-0.007, 0.055] | 12 | 0.19 [-0.08, 0.43] | 0.058 [-0.018, 0.139] | 0.19 [-0.08, 0.43] |
| (iii) verbalized confidence | b_weighted_plurality | 0.653 [0.587, 0.718] | -0.004 [-0.012, 0.003] | 4 | -0.03 [-0.12, 0.03] | -0.009 [-0.030, 0.008] | -0.03 [-0.12, 0.03] |
| (iii) verbalized confidence | c_highest_signal | 0.660 [0.600, 0.719] | 0.003 [-0.022, 0.029] | 41 | 0.03 [-0.25, 0.20] | 0.008 [-0.056, 0.072] | 0.03 [-0.24, 0.20] |

### pool first 10

Plurality 0.678 [0.614, 0.740]; coverage 0.815 [0.760, 0.865]; gap 0.137 [0.094, 0.182]; non-unanimous questions: 93.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.675 [0.610, 0.740] | -0.003 [-0.021, 0.014] | 12 | -0.02 [-0.16, 0.10] | -0.007 [-0.044, 0.029] | -0.02 [-0.16, 0.10] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.672 [0.605, 0.738] | -0.006 [-0.023, 0.010] | 11 | -0.04 [-0.18, 0.08] | -0.012 [-0.049, 0.022] | -0.04 [-0.17, 0.08] |
| (i) before-stated-answer distribution | c_highest_signal | 0.651 [0.590, 0.712] | -0.027 [-0.053, -0.000] | 41 | -0.20 [-0.47, -0.00] | -0.058 [-0.113, -0.002] | -0.20 [-0.47, -0.00] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.690 [0.625, 0.755] | 0.012 [-0.002, 0.028] | 11 | 0.09 [-0.02, 0.21] | 0.026 [-0.005, 0.060] | 0.09 [-0.02, 0.21] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.660 [0.590, 0.725] | -0.018 [-0.065, 0.027] | 30 | -0.13 [-0.58, 0.18] | -0.039 [-0.136, 0.058] | -0.13 [-0.57, 0.17] |
| (iii) verbalized confidence | b_weighted_plurality | 0.682 [0.618, 0.745] | 0.004 [-0.013, 0.021] | 10 | 0.03 [-0.10, 0.16] | 0.009 [-0.028, 0.046] | 0.03 [-0.10, 0.16] |
| (iii) verbalized confidence | c_highest_signal | 0.659 [0.599, 0.716] | -0.020 [-0.044, 0.004] | 50 | -0.14 [-0.40, 0.03] | -0.042 [-0.093, 0.010] | -0.14 [-0.40, 0.03] |

### debate3 post-debate

Plurality 0.668 [0.602, 0.732]; coverage 0.755 [0.695, 0.815]; gap 0.087 [0.050, 0.127]; non-unanimous questions: 56.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.665 [0.600, 0.730] | -0.003 [-0.008, 0.000] | 2 | -0.04 [-0.10, 0.00] | -0.012 [-0.030, 0.000] | -0.04 [-0.10, 0.00] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.668 [0.602, 0.732] | 0.000 [0.000, 0.000] | 0 | 0.00 [0.00, 0.00] | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] |
| (i) before-stated-answer distribution | c_highest_signal | 0.667 [0.604, 0.728] | -0.001 [-0.017, 0.015] | 23 | -0.01 [-0.28, 0.14] | -0.003 [-0.060, 0.054] | -0.01 [-0.27, 0.14] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.670 [0.605, 0.735] | 0.002 [-0.005, 0.010] | 2 | 0.02 [-0.05, 0.12] | 0.006 [-0.018, 0.036] | 0.02 [-0.05, 0.12] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.680 [0.615, 0.745] | 0.012 [-0.020, 0.043] | 12 | 0.13 [-0.29, 0.44] | 0.042 [-0.071, 0.155] | 0.13 [-0.29, 0.45] |
| (iii) verbalized confidence | b_weighted_plurality | 0.671 [0.605, 0.733] | 0.003 [0.000, 0.007] | 1 | 0.03 [0.00, 0.10] | 0.009 [0.000, 0.027] | 0.03 [0.00, 0.10] |
| (iii) verbalized confidence | c_highest_signal | 0.676 [0.612, 0.738] | 0.007 [-0.019, 0.035] | 17 | 0.09 [-0.30, 0.35] | 0.027 [-0.068, 0.125] | 0.09 [-0.29, 0.34] |
| (iv) peer-exposure readout | a_soft_vote | 0.665 [0.600, 0.730] | -0.003 [-0.032, 0.025] | 10 | -0.04 [-0.46, 0.27] | -0.012 [-0.113, 0.089] | -0.04 [-0.46, 0.25] |
| (iv) peer-exposure readout | b_weighted_plurality | 0.670 [0.605, 0.735] | 0.002 [-0.020, 0.025] | 7 | 0.02 [-0.28, 0.26] | 0.006 [-0.071, 0.089] | 0.02 [-0.29, 0.26] |
| (iv) peer-exposure readout | c_highest_signal | 0.670 [0.605, 0.735] | 0.002 [-0.037, 0.040] | 17 | 0.02 [-0.54, 0.40] | 0.006 [-0.131, 0.149] | 0.02 [-0.57, 0.40] |

### structured3 post-debate

Plurality 0.670 [0.603, 0.733]; coverage 0.730 [0.665, 0.790]; gap 0.060 [0.030, 0.093]; non-unanimous questions: 60.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.670 [0.605, 0.735] | 0.000 [-0.007, 0.008] | 3 | 0.00 [-0.12, 0.16] | 0.000 [-0.022, 0.028] | 0.00 [-0.12, 0.16] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.671 [0.604, 0.735] | 0.001 [0.000, 0.003] | 1 | 0.01 [0.00, 0.05] | 0.003 [0.000, 0.008] | 0.01 [0.00, 0.05] |
| (i) before-stated-answer distribution | c_highest_signal | 0.660 [0.598, 0.720] | -0.010 [-0.030, 0.011] | 26 | -0.17 [-0.81, 0.13] | -0.033 [-0.097, 0.039] | -0.17 [-0.82, 0.14] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.665 [0.600, 0.730] | -0.005 [-0.012, 0.000] | 3 | -0.08 [-0.20, 0.00] | -0.017 [-0.039, 0.000] | -0.08 [-0.20, 0.00] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.675 [0.610, 0.740] | 0.005 [-0.015, 0.027] | 7 | 0.08 [-0.31, 0.40] | 0.017 [-0.050, 0.089] | 0.08 [-0.30, 0.40] |
| (iii) verbalized confidence | b_weighted_plurality | 0.672 [0.605, 0.735] | 0.002 [-0.005, 0.010] | 2 | 0.03 [-0.08, 0.18] | 0.006 [-0.017, 0.033] | 0.03 [-0.08, 0.18] |
| (iii) verbalized confidence | c_highest_signal | 0.677 [0.613, 0.739] | 0.007 [-0.012, 0.026] | 10 | 0.11 [-0.26, 0.37] | 0.022 [-0.039, 0.086] | 0.11 [-0.26, 0.38] |
| (iv) peer-exposure readout | a_soft_vote | 0.670 [0.605, 0.735] | 0.000 [-0.043, 0.043] | 21 | 0.00 [-1.06, 0.62] | -0.017 [-0.139, 0.106] | -0.08 [-1.05, 0.43] |
| (iv) peer-exposure readout | b_weighted_plurality | 0.690 [0.625, 0.750] | 0.020 [-0.010, 0.050] | 11 | 0.33 [-0.23, 0.70] | 0.067 [-0.028, 0.167] | 0.33 [-0.20, 0.70] |
| (iv) peer-exposure readout | c_highest_signal | 0.690 [0.625, 0.750] | 0.020 [-0.013, 0.053] | 13 | 0.33 [-0.32, 0.71] | 0.067 [-0.039, 0.178] | 0.33 [-0.30, 0.72] |

### debate5 post-debate

Plurality 0.660 [0.593, 0.723]; coverage 0.755 [0.695, 0.815]; gap 0.095 [0.058, 0.135]; non-unanimous questions: 69.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.655 [0.585, 0.720] | -0.005 [-0.018, 0.007] | 6 | -0.05 [-0.19, 0.07] | -0.014 [-0.051, 0.022] | -0.05 [-0.19, 0.07] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.667 [0.600, 0.730] | 0.007 [0.000, 0.018] | 3 | 0.08 [0.00, 0.19] | 0.022 [0.000, 0.051] | 0.08 [0.00, 0.19] |
| (i) before-stated-answer distribution | c_highest_signal | 0.653 [0.590, 0.713] | -0.007 [-0.025, 0.011] | 30 | -0.07 [-0.35, 0.10] | -0.019 [-0.070, 0.033] | -0.07 [-0.35, 0.10] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.660 [0.590, 0.725] | 0.000 [-0.013, 0.013] | 6 | 0.00 [-0.13, 0.13] | 0.000 [-0.036, 0.036] | 0.00 [-0.13, 0.13] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.635 [0.570, 0.700] | -0.025 [-0.060, 0.010] | 18 | -0.26 [-0.81, 0.10] | -0.072 [-0.174, 0.029] | -0.26 [-0.81, 0.10] |
| (iii) verbalized confidence | b_weighted_plurality | 0.662 [0.595, 0.728] | 0.003 [-0.007, 0.013] | 5 | 0.03 [-0.09, 0.15] | 0.007 [-0.022, 0.036] | 0.03 [-0.09, 0.15] |
| (iii) verbalized confidence | c_highest_signal | 0.664 [0.598, 0.727] | 0.004 [-0.032, 0.039] | 22 | 0.04 [-0.44, 0.36] | 0.011 [-0.087, 0.113] | 0.04 [-0.40, 0.35] |
| (iv) peer-exposure readout | a_soft_vote | 0.675 [0.610, 0.740] | 0.015 [-0.015, 0.048] | 15 | 0.16 [-0.20, 0.44] | 0.043 [-0.043, 0.138] | 0.16 [-0.20, 0.45] |
| (iv) peer-exposure readout | b_weighted_plurality | 0.690 [0.625, 0.750] | 0.030 [0.007, 0.058] | 11 | 0.32 [0.09, 0.56] | 0.087 [0.022, 0.159] | 0.32 [0.09, 0.55] |
| (iv) peer-exposure readout | c_highest_signal | 0.680 [0.615, 0.745] | 0.020 [-0.018, 0.058] | 19 | 0.21 [-0.23, 0.53] | 0.058 [-0.051, 0.167] | 0.21 [-0.22, 0.52] |

### structured5 post-debate

Plurality 0.667 [0.603, 0.730]; coverage 0.730 [0.665, 0.790]; gap 0.062 [0.033, 0.098]; non-unanimous questions: 63.

| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |
|---|---|---|---|---|---|---|---|
| (i) before-stated-answer distribution | a_soft_vote | 0.670 [0.605, 0.735] | 0.003 [-0.005, 0.010] | 3 | 0.04 [-0.10, 0.20] | 0.008 [-0.016, 0.032] | 0.04 [-0.10, 0.20] |
| (i) before-stated-answer distribution | b_weighted_plurality | 0.667 [0.600, 0.733] | 0.000 [-0.007, 0.007] | 2 | 0.00 [-0.12, 0.12] | 0.000 [-0.024, 0.024] | 0.00 [-0.12, 0.12] |
| (i) before-stated-answer distribution | c_highest_signal | 0.643 [0.581, 0.704] | -0.024 [-0.048, -0.000] | 35 | -0.39 [-1.23, -0.00] | -0.077 [-0.152, -0.002] | -0.39 [-1.23, -0.01] |
| (ii) response mean log-prob (as exp) | b_weighted_plurality | 0.670 [0.605, 0.735] | 0.003 [-0.005, 0.010] | 3 | 0.04 [-0.10, 0.20] | 0.008 [-0.016, 0.032] | 0.04 [-0.10, 0.20] |
| (ii) response mean log-prob (as exp) | c_highest_signal | 0.655 [0.585, 0.720] | -0.013 [-0.045, 0.020] | 13 | -0.20 [-1.00, 0.26] | -0.040 [-0.143, 0.063] | -0.20 [-1.00, 0.26] |
| (iii) verbalized confidence | b_weighted_plurality | 0.670 [0.605, 0.735] | 0.003 [-0.005, 0.010] | 3 | 0.04 [-0.10, 0.20] | 0.008 [-0.016, 0.032] | 0.04 [-0.10, 0.20] |
| (iii) verbalized confidence | c_highest_signal | 0.677 [0.612, 0.739] | 0.009 [-0.014, 0.033] | 14 | 0.15 [-0.31, 0.45] | 0.029 [-0.046, 0.106] | 0.15 [-0.32, 0.46] |
| (iv) peer-exposure readout | a_soft_vote | 0.660 [0.595, 0.725] | -0.007 [-0.052, 0.035] | 22 | -0.12 [-1.20, 0.48] | -0.008 [-0.135, 0.119] | -0.04 [-1.00, 0.50] |
| (iv) peer-exposure readout | b_weighted_plurality | 0.665 [0.600, 0.730] | -0.003 [-0.033, 0.028] | 12 | -0.04 [-0.75, 0.38] | -0.008 [-0.111, 0.087] | -0.04 [-0.76, 0.38] |
| (iv) peer-exposure readout | c_highest_signal | 0.665 [0.600, 0.730] | -0.003 [-0.035, 0.030] | 14 | -0.04 [-0.83, 0.41] | -0.008 [-0.119, 0.095] | -0.04 [-0.85, 0.42] |

Rules (b) and (c) choose among the answers given: on unanimous answer sets the gap is zero and they equal plurality, so their share of the gap closed is the same overall and on non-unanimous questions by construction (only the per-question difference grows). A soft vote (a) can choose an option no agent gave, so it can differ even on unanimous sets (e.g. (iv) on structured3).
