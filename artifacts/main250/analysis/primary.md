# main250 primary analysis (200 questions)

Specification: `docs/MAIN250_ANALYSIS_SPEC.md`. Score: exact expected accuracy of the panel's plurality vote under random tie-breaking, per question; differences are paired by question. Intervals: 95% percentile, 10,000 question resamples (seed 0). 6 primary contrasts, no multiplicity correction. With 200 questions, small differences are not expected to be resolvable; an interval covering 0 is not evidence of no effect.

Missing questions by condition: {'pool': 0, 'debate3': 0, 'structured3': 0, 'debate5': 0, 'structured5': 0}

## Primary contrasts

| contrast | arm A | arm B | mean diff (A − B) | 95% interval | questions differing | expected gain | expected loss | tokens in/out per q, A | tokens in/out per q, B |
|---|---|---|---|---|---|---|---|---|---|
| debate3 vs independent 6 | 0.668 | 0.655 | +0.013 | [-0.025, +0.050] | 22 | 9.83 | 7.17 | 4,983 / 1,564 | 1,936 / 1,867 |
| structured3 vs independent 6 | 0.670 | 0.655 | +0.015 | [-0.028, +0.058] | 27 | 12.33 | 9.33 | 5,367 / 1,969 | 1,936 / 1,867 |
| debate5 vs independent 10 | 0.660 | 0.678 | -0.018 | [-0.066, +0.028] | 33 | 10.83 | 14.45 | 11,370 / 2,589 | 3,226 / 3,113 |
| structured5 vs independent 10 | 0.667 | 0.678 | -0.011 | [-0.052, +0.029] | 27 | 8.83 | 10.95 | 12,010 / 3,220 | 3,226 / 3,113 |
| structured3 vs debate3 | 0.670 | 0.668 | +0.002 | [-0.028, +0.033] | 14 | 6.00 | 5.67 | 5,367 / 1,969 | 4,983 / 1,564 |
| structured5 vs debate5 | 0.667 | 0.660 | +0.007 | [-0.037, +0.050] | 25 | 11.50 | 10.00 | 12,010 / 3,220 | 11,370 / 2,589 |

## Independent-voting curve (descriptive)

| answers voted | expected accuracy | 95% interval |
|---|---|---|
| 3 | 0.667 | [0.602, 0.730] |
| 4 | 0.650 | [0.585, 0.714] |
| 5 | 0.657 | [0.591, 0.721] |
| 6 | 0.655 | [0.590, 0.718] |
| 7 | 0.666 | [0.600, 0.728] |
| 8 | 0.674 | [0.608, 0.738] |
| 9 | 0.678 | [0.613, 0.740] |
| 10 | 0.678 | [0.614, 0.740] |

## Revision events

| condition | panel before → after | net change [95%] | revisions | changed | helpful | harmful | correct minority recovered | correct decision preserved | correct decision lost | correct absent, found |
|---|---|---|---|---|---|---|---|---|---|---|
| debate3 | 0.667 → 0.668 | +0.002 [-0.040, +0.042] | 600 | 158 | 51 | 42 | 10 | 123 | 8 | 4 |
| structured3 | 0.667 → 0.670 | +0.003 [-0.037, +0.043] | 600 | 139 | 40 | 37 | 9 | 123 | 8 | 3 |
| debate5 | 0.657 → 0.660 | +0.003 [-0.037, +0.043] | 1000 | 272 | 84 | 79 | 11 | 121 | 10 | 1 |
| structured5 | 0.657 → 0.667 | +0.011 [-0.024, +0.044] | 1000 | 206 | 59 | 59 | 8 | 125 | 6 | 1 |

Panel event counts are questions (fractional panel scores enter as described in `mad.analysis.panels.panel_events`).

## Stated-answer fallback rate

| condition | answers | readout fallback | rate | scope |
|---|---|---|---|---|
| pool | 2000 | 17 | 0.85% | the 10 independent answers |
| debate3 | 600 | 1 | 0.17% | the revised (post-debate) answers |
| structured3 | 600 | 0 | 0.00% | the revised (post-debate) answers |
| debate5 | 1000 | 0 | 0.00% | the revised (post-debate) answers |
| structured5 | 1000 | 0 | 0.00% | the revised (post-debate) answers |

## Measured cost

Round 0 (the pool's 10 independent answers per question): 11,946 s in total (generation + readouts of all 10 independent answers; not separable in the records).

| condition | responses | input tokens | output tokens | generation s | generation s / response | extraction s (readouts) |
|---|---|---|---|---|---|---|
| pool | 2000 | 645,220 | 622,563 | (in round 0) | 5.97 incl. readout | (in round 0) |
| debate3 | 600 | 802,996 | 125,932 | 3,398 | 5.66 | 299 |
| structured3 | 600 | 879,796 | 206,871 | 5,023 | 8.37 | 355 |
| debate5 | 1000 | 1,951,421 | 208,499 | 6,198 | 6.2 | 715 |
| structured5 | 1000 | 2,079,421 | 334,807 | 9,379 | 9.38 | 810 |
