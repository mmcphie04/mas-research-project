# Next step: recommendation (2026-10-13)

Based on the corrected pilot comparison (docs/PROBE_AUDIT.md). One path is recommended. The
other two are kept as conditional follow-ons, triggered by the decision rules below.

## What the corrected pilot says

- **Ranking signal survives correction.** A fixed activation recipe ranks answer states by
  correctness above chance on held-out pilot questions. Natural-debate helpful vs harmful
  revisions score 0.72 (R1) and 0.78 (R2), and the signal transfers to a held-out condition.
- **Its edge over a strong, matched output model is narrow.**
  - Inside a revised debate panel: +0.12 to +0.18.
  - Revision direction: +0.06 to +0.12.
  - Six-candidate ranking, round-0 ranking and single-state correctness: nothing.
  - No interval excludes zero in any partition, and a learned combination ignores it.
- **Selection is competitive, not established.** Activation-based selection matches or slightly
  exceeds pooled-six voting (0–4 of 120) and the matched output selector (median +2).
- **Debate barely contributed.** It added a correct option missing from round 0 on 1 of 120
  questions. Best-of-six picks are the same answers whichever debate variant supplied the revised
  candidates.

Per the brief's rule, a useful ranking signal remains and activation-augmented selection is
competitive with strong output baselines. That points to the candidate-assessment path, with one
adjustment: the pilot itself says debate's value as a candidate source is doubtful, so the
independent branch is essential, not optional.

## Recommendation: candidate assessment on fresh questions

**Question.** On new questions, does a compact internal scorer, frozen from the pilot, help choose
among candidate answers better than strong output-only scorers? And is discussion a better source
of candidates than more independent samples?

- **Population and split.**
  - 200 fresh MMLU-Pro questions (master-order positions 121–320, subject-balanced), disjoint from
    the pilot and from the reserved confirmation pool (positions 321 onward), enforced by the
    driver.
  - No re-splitting: every scorer is fitted once on the whole pilot and frozen.
  - This tranche is validation/development; it may change the main design.
- **Collection.**
  - The same first three answers feed natural debate and an `independent` branch (three fresh
    answers), so the debate six and the independent six share their first half exactly.
  - Optionally, self-reconsideration on the first 100 questions, under a usage rule.
  - Specification and caps: docs/VALIDATION_JOB.md.
- **Cost.** About 2.8 h of raw chunk compute and 3.3–4.1 h expected account usage. Hard stop at
  5.0 h (26.5 units) including the smoke test.
- **Budget.** Measured quota: 281.34 units, about 53 A100-h at 5.30/h. Against the 10–20 h working
  budget, about 3.5 h are used. After this job at least 11.5 h of a 20-h budget remains, the
  majority, for fresh confirmation and one robustness check.
- **Frozen before the data exist** (preregistration §10):
  - the two recipes R1 and R2, scorers fitted on the whole pilot (SHA-256 recorded);
  - the comparators: `out_lr[full]`, `out_hgb[full]`, and the equal-weight sum;
  - the endpoints below, and these decision rules.

### Endpoints (natural debate primary; question bootstrap; one frozen model, so no fold issue)

1. **Ranking:** six-candidate AUROC, revised-panel AUROC and revision AUROC (helpful vs harmful)
   for R1 and R2, each paired against `out_lr[full]`.
2. **Selection:** best of the debate six, best of the independent six, and best first answer, by
   activation, output and equal-weight scores. Each is compared against pooled-six voting (debate
   and independent) and the matched output selector. Report wins and losses with question ids.
3. **Candidate source:** coverage (any correct) and pooled votes for 3 first answers, debate six
   and independent six. Report how often a correct option is new in each branch.
4. **Shift check:** calibration-in-the-large of the frozen Platt probabilities, and round-0
   ranking on the new questions vs the pilot.

### Decision rules (stated before the data)

- **D1, ranking reproduces:** for R1 or R2, the six-candidate AUROC and the revision AUROC both
  have 95% intervals above 0.5.
- **D2, incremental value:** for the same recipe, activation or the equal-weight sum minus
  `out_lr[full]` is at least +0.05 on revision AUROC or revised-panel AUROC, with the interval's
  lower bound above 0.
- **D3, practical value:** best-of-six by the D2 scorer is at least as good as both pooled-six
  voting and the matched output selector (point estimates).
- **D4, candidate source:** best-of-six from the independent six vs from the debate six.

| outcome | next step |
|---|---|
| D1 and D2 and D3 hold | Confirmation run on the reserved pool, candidate assessment as the primary hypothesis, analysis copied from §10. Size from the observed effect; at most the remaining budget. |
| D1 holds, D2 fails | Fixed-candidate endorsement diagnostic (hold question and candidate fixed; vary peer endorsement; plausible wrong candidates; matched-format training; output controls) at small scale before any confirmation. |
| D1 fails | Stop the activation line. Report the pilot as exploratory, and spend the remainder on the debate-protocol study (simultaneous swaps, round 2 with symmetric presentation). |
| D4 favours independent sampling, whatever else | The paper reports debate's candidate value as null and frames selection as assessment among samples. More agents only after an independent-sampling coverage curve. |

## Why not the others now

- **Endorsement diagnostic.** It answers what the score means, which matters only if the score
  reproduces on fresh questions. It also needs new prompts and matched-format training. It is the
  designated follow-on if D2 fails. The pilot hint for it is that the activation advantage
  concentrates where a peer endorses a wrong answer.
- **Debate dynamics (round 2).** It needs round-aware storage and new long-context memory checks.
  The pilot shows debate rarely creates correct options, so a second round is a weaker bet than
  first testing whether debate beats independent sampling at all, which D4 does at no extra cost.
