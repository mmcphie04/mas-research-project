# Oracle gate ceilings (EXPLORATORY)

Exact expected values per panel. Perfect gate: accepts helpful, rejects harmful, and for wrong→wrong revisions takes the accept/reject combination best for the panel. Random gate: right with probability 0.6 / 0.7 on each helpful/harmful revision independently; wrong→wrong accepted with probability ½. main250: train+validation questions only.

## val12b debate (panel 3, 200 questions)

| policy | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| reject every revision | 0.588 [0.527, 0.648] | 0.588 [0.522, 0.653] | -0.003 [-0.048, 0.042] |
| accept every revision | 0.600 [0.538, 0.660] | 0.592 [0.525, 0.658] | 0.000 [0.000, 0.000] |
| perfect gate | 0.677 [0.615, 0.737] | 0.693 [0.630, 0.755] | 0.102 [0.063, 0.145] |
| random gate, 60% right | 0.611 [0.550, 0.670] | 0.619 [0.558, 0.680] | 0.028 [0.004, 0.053] |
| random gate, 70% right | 0.627 [0.567, 0.686] | 0.641 [0.579, 0.701] | 0.049 [0.023, 0.078] |

## main250 debate3 (panel 3, 102 questions)

| policy | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| reject every revision | 0.657 [0.575, 0.739] | 0.676 [0.585, 0.765] | -0.016 [-0.075, 0.039] |
| accept every revision | 0.667 [0.582, 0.748] | 0.693 [0.601, 0.778] | 0.000 [0.000, 0.000] |
| perfect gate | 0.742 [0.657, 0.824] | 0.739 [0.650, 0.824] | 0.046 [0.010, 0.088] |
| random gate, 60% right | 0.678 [0.597, 0.758] | 0.693 [0.610, 0.775] | 0.001 [-0.028, 0.032] |
| random gate, 70% right | 0.694 [0.612, 0.773] | 0.709 [0.626, 0.791] | 0.017 [-0.013, 0.051] |

## main250 structured3 (panel 3, 102 questions)

| policy | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| reject every revision | 0.657 [0.575, 0.739] | 0.676 [0.585, 0.765] | 0.003 [-0.049, 0.052] |
| accept every revision | 0.650 [0.565, 0.735] | 0.673 [0.582, 0.761] | 0.000 [0.000, 0.000] |
| perfect gate | 0.719 [0.634, 0.801] | 0.722 [0.637, 0.807] | 0.049 [0.016, 0.088] |
| random gate, 60% right | 0.667 [0.585, 0.748] | 0.685 [0.599, 0.770] | 0.012 [-0.013, 0.040] |
| random gate, 70% right | 0.680 [0.598, 0.760] | 0.696 [0.610, 0.781] | 0.023 [-0.003, 0.054] |

## main250 debate5 (panel 5, 102 questions)

| policy | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| reject every revision | 0.649 [0.569, 0.729] | 0.676 [0.588, 0.765] | -0.020 [-0.074, 0.034] |
| accept every revision | 0.663 [0.580, 0.743] | 0.696 [0.608, 0.784] | 0.000 [0.000, 0.000] |
| perfect gate | 0.733 [0.651, 0.812] | 0.755 [0.667, 0.833] | 0.059 [0.020, 0.108] |
| random gate, 60% right | 0.671 [0.592, 0.750] | 0.698 [0.614, 0.779] | 0.002 [-0.024, 0.031] |
| random gate, 70% right | 0.687 [0.607, 0.765] | 0.716 [0.632, 0.797] | 0.020 [-0.006, 0.051] |

## main250 structured5 (panel 5, 102 questions)

| policy | agent accuracy | panel accuracy | panel − accept-all |
|---|---|---|---|
| reject every revision | 0.649 [0.569, 0.729] | 0.676 [0.588, 0.765] | 0.000 [-0.039, 0.034] |
| accept every revision | 0.653 [0.569, 0.735] | 0.676 [0.588, 0.765] | 0.000 [0.000, 0.000] |
| perfect gate | 0.702 [0.618, 0.784] | 0.706 [0.618, 0.794] | 0.029 [0.005, 0.064] |
| random gate, 60% right | 0.661 [0.580, 0.742] | 0.685 [0.598, 0.772] | 0.009 [-0.012, 0.036] |
| random gate, 70% right | 0.671 [0.590, 0.753] | 0.693 [0.606, 0.781] | 0.017 [-0.005, 0.046] |
