# Fresh-question validation: results (2026-10-14)

200 MMLU-Pro questions never used before (master positions 121–320; the collected IDs equal the
planned list; no overlap with the pilot). One shared round 0 (three answers per question) feeds
natural debate (three revised answers) and an independent branch (three fresh samples of the
original prompt). A self-reconsideration child covers the first 100 questions.

**Frozen analysis.** Scorers fitted on the pilot (SHA-256 `8e5785fe…`, `3857a72`) were applied
once (`probe_validate.py apply`, 10,000 question-bootstrap draws). Plan and reading rules:
docs/preregistration.md §10 and docs/NEXT_STEP_MEMO.md (dated 2026-10-13, before collection).
Exports: `artifacts/val12b/frozen/`; reconsideration: `artifacts/val12b/frozen_self/`.

This is a development tranche, so a later untouched pool is needed for any claim chosen using it.
Intervals are conditional on the frozen scorers, and AUROC is not final-answer accuracy.

## 1. Primary: activation best-of-six vs matched output best-of-six (R1, natural debate, 200 questions)

| rule | questions correct of 200 |
|---|---|
| original vote / revised vote / pooled-six vote | 116 / 119 / 116 (random-tie expectation 117.7 / 118.3 / 117.5) |
| best of six, R1 activation | **122** |
| best of six, matched output (`out_lr[full]`) | **122** |
| best of six, output HGB | 126 |
| best of six, equal-weight activation + output | 123 |
| best first answer, R1 activation / matched output | 122 / 121 |

- **R1 − matched output:** **+0.000** [−0.045, +0.045]; wins 10, losses 10. **Inconclusive** at the
  +2-point target: the interval includes both no gain and the target. Narrowing the interval to
  ±1 point would need about 4,050 fresh questions, roughly 54 raw GPU-hours at the measured
  47.7 s/question. That is not practical within the budget.
- **R1 − pooled-six vote:** +0.030 [−0.015, +0.075]; 14 / 8. That reaches the target in point
  estimate, a *provisional lead*. But the matched output selector gains exactly the same (+0.030,
  11 / 5), and HGB gains more (+0.050 [+0.015, +0.090], 13 / 3). So the gain over voting comes
  from selecting by a score, not from activations.

## 2. Key explanatory comparison: revision ranking (R1, natural debate)

53 helpful and 46 harmful revisions on 48 questions (2,438 comparison pairs), plus 70
wrong-to-wrong revisions.

| | revision AUROC (helpful vs harmful) | revised-panel AUROC (76 pairs / 38 questions) |
|---|---|---|
| R1 activation | 0.559 [0.42, 0.70] | 0.500 [0.35, 0.65] |
| matched output | 0.653 [0.49, 0.81] | 0.618 [0.47, 0.76] |
| output HGB | 0.715 [0.57, 0.84] | 0.592 [0.45, 0.74] |
| R1 − matched output | **−0.095 [−0.27, +0.09]** | −0.118 [−0.31, +0.08] |

The pilot's revision-ranking advantage (+0.06 R1, +0.12 R2, exploratory) **did not reproduce**:
the point estimate now favours the output model. Formally the difference is inconclusive (the
interval still reaches +0.05). The activation score's own revision AUROC has an interval that
includes 0.5.

## 3. Candidate source: independent six vs debate six (same first three answers, 200 questions)

| | debate | independent | difference (debate − independent) |
|---|---|---|---|
| some correct answer among the six | 144 | **156** | −0.060 [−0.100, −0.025] (2 vs 14 questions) |
| pooled-six vote | 116 | 120 | −0.020 [−0.050, +0.005] |
| best of six, R1 activation | 122 | 121 | +0.005 [−0.040, +0.050] |
| best of six, matched output | 122 | 123 | −0.005 [−0.030, +0.020] |
| best of six, HGB | 126 | 125 | +0.005 [−0.030, +0.040] |

- Round 0 alone covers 141 questions. Debate added a correct option missing from round 0 on 3
  questions; three independent samples did so on 15.
- Debate's revised vote (119) beat the original vote (116) here, unlike the pilot (−2).
- With either pool, picking the best of six adds at most 0–3 questions over the same scorer's
  best first answer.
- The independent branch's old/new "transitions" are resampling comparisons, not persuasion.

## 4. Secondary

- **R2 (before-answer recipe).** Complete-case, 182 questions with every debate state: revision
  AUROC 0.619 vs matched output 0.633 (−0.014 [−0.25, +0.22]); revised panel +0.015; six
  candidates −0.040. All-question fallback (200): best of six 123 vs pooled vote 116, and
  essentially equal to the output selectors. No R2 advantage reproduced.
- **Combinations.**
  - The equal-weight sums rank round-0 panels best: R1 sum 0.648 vs matched output 0.568,
    +0.080 [−0.02, +0.19]; R2 sum +0.10 to +0.11, intervals including 0. They rank revised panels
    worse (R1 sum −0.105 [−0.21, −0.01]).
  - The learned combination tracks the output model (differences ≤ 0.02).
- **Gates.** The zero-threshold activation gate accepts 8 of 53 helpful revisions, blocks 39 of 46
  harmful ones and blocks 58 of 70 wrong-to-wrong changes. It scores 117, against 120 for the
  matched output gate and **125** for the HGB gate. HGB gate − pooled-six vote is +0.045
  [+0.010, +0.085] (12 / 3).
- **Output HGB, a fixed comparator, was the best selector and gate on these fresh questions.**
  That is a secondary result: it suggests a later hypothesis ("a nonlinear output-only scorer
  beats voting") and replaces no primary endpoint.
- **Reconsideration (100 questions; secondary).**
  - Coverage: six-answer coverage 80 (round 0 covers 71; on 9 questions only a reconsidered
    answer is correct).
  - Votes: revised 49, original 56, pooled six 57.
  - R1 revision AUROC 0.695 vs matched output 0.576 (+0.119 [−0.06, +0.29]); HGB 0.737.
  - Best of six: R1 60, matched output 56, HGB 61.
  - This is a hint for a later hypothesis, on 100 questions, not a replacement for the debate
    result.

## 5. Diagnostics

- **Shift:** the frozen R1 score ranks round-0 states at 0.714 on the new questions vs 0.652
  out of fold in the pilot (R2 0.753 vs 0.672). The matched output model gets 0.746 vs 0.769.
  Calibration-in-the-large is close: R1 mean predicted 0.63 vs observed 0.60 on debate's revised
  states; matched output 0.60 vs 0.60.
- **Operational:**
  - Capped / fallback answers: 4.8% of round 0, 0.3% of debate revisions, 4.2% of independent
    answers.
  - Absent before-answer states: 29 / 2 / 25.
  - Every state is finite, with 49 layers × 3,840 (`verify_archive`).

## 6. Reading (rules of docs/NEXT_STEP_MEMO.md)

- **Primary:** *inconclusive*, with no observed difference (0 of 200 net). Resolving a 2-point
  difference would need about 20× this tranche.
- **Key explanatory:** the pilot's activation advantage in revision ranking *did not reproduce*.
  The point estimate favours the output model; formally inconclusive at the +0.05 target.
- **Candidate source:** independent sampling produced more correct candidates than debate, with an
  interval excluding zero (+6 points of coverage), and slightly better pooled votes.
- **Simpler controls dominate in this setting:**
  - output-only selectors match or beat the activation selectors;
  - independent sampling beats debate as a source of candidates.

  These conclusions are scoped to these two recipes and this setting.
