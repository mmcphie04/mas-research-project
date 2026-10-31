# Cross-model judge: follow-ups (EXPLORATORY)

Question-grouped 95% intervals (10,000 resamples). Own answers at the 250-word protocol; Gemma's include pool agents 5–7 for the 42 main250/pilot250 questions.

## 1. Mixed 3-answer panels (149 questions)

| panel | accuracy | − GGG | − QQQ |
|---|---|---|---|
| GGG | 0.453 [0.376, 0.530] | — | -0.048 [-0.144, 0.047] |
| QQQ | 0.501 [0.425, 0.577] | 0.048 [-0.047, 0.144] | — |
| LLL | 0.379 [0.308, 0.452] | -0.074 [-0.157, 0.008] | -0.122 [-0.197, -0.047] |
| QQG | 0.496 [0.426, 0.567] | 0.043 [-0.039, 0.126] | -0.005 [-0.036, 0.026] |
| QGG | 0.499 [0.430, 0.568] | 0.045 [0.010, 0.082] | -0.003 [-0.078, 0.074] |

QQQ-thinking is not possible (one thinking sample per question). Single answers on the 44 thinking questions: Qwen3-thinking 0.500 [0.364, 0.659]; Qwen3 0.439 [0.303, 0.576] (mean of its 3 samples); Gemma 0.394 [0.273, 0.515].

**Pair-level error statistics** (one sample from each side; within a model: distinct samples):

| pair | questions | correlation of correctness | same-wrong-answer rate | wrong pairs | uniform expectation |
|---|---|---|---|---|---|
| G-G | 149 | 0.590 [0.486, 0.686] | 0.569 [0.480, 0.657] | 204 | 0.120 |
| Q-Q | 149 | 0.678 [0.579, 0.766] | 0.637 [0.545, 0.728] | 193 | 0.121 |
| L-L | 142 | 0.448 [0.326, 0.562] | 0.448 [0.363, 0.537] | 194 | 0.126 |
| G-Q | 149 | 0.243 [0.127, 0.359] | 0.302 [0.213, 0.400] | 464 | 0.118 |
| G-L | 149 | 0.289 [0.185, 0.393] | 0.313 [0.251, 0.382] | 517 | 0.119 |
| Q-L | 149 | 0.367 [0.265, 0.465] | 0.354 [0.282, 0.429] | 514 | 0.123 |

## 2. Judge accuracy by the judge model's own majority answer

| judge | own = original | own = revision | own = neither (incl. no majority) |
|---|---|---|---|
| gemma | 0.606 [0.505, 0.706] (n=175, helpful 58) | 0.621 [0.474, 0.753] (n=132, helpful 89) | 0.655 [0.561, 0.776] (n=84, helpful 48) |
| qwen3 | 0.664 [0.547, 0.776] (n=137, helpful 19) | 0.742 [0.645, 0.828] (n=151, helpful 128) | 0.592 [0.465, 0.730] (n=103, helpful 48) |
| qwen3_think | 0.846 [0.686, 0.957] (n=39, helpful 11) | 0.838 [0.684, 0.972] (n=37, helpful 28) | 0.917 [0.625, 1.000] (n=12, helpful 4) |
| llama | 0.462 [0.341, 0.596] (n=117, helpful 26) | 0.716 [0.583, 0.821] (n=116, helpful 93) | 0.557 [0.457, 0.664] (n=158, helpful 76) |

## 3. Thinking subsample (44 questions, 110 Gemma panels)

| method | accuracy | seconds per question |
|---|---|---|
| Gemma panels, accept every revision | 0.363 [0.238, 0.496] | 38 |
| Gemma panels, gated by the Qwen3-thinking judge | 0.559 [0.418, 0.701] | 78 |
| Qwen3-thinking, single own answer | 0.500 [0.364, 0.659] | 10 |

Differences: gated − accept-all 0.196 [0.096, 0.312]; own − accept-all 0.137 [-0.035, 0.308]; own − gated -0.059 [-0.192, 0.076]. Seconds: Gemma from the source jobs' measured stage times (round 0 shared per agent plus the panel's condition); the judge adds 22.5 s per changed answer (both orders, measured); the own answer is the judge run's measured thinking own-answer time. Different jobs and protocols (150/250 words), all on an A100.
