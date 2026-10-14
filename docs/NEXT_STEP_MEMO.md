# Validation plan and how its results will be read (dated 2026-10-13, before collection)

The corrected pilot (docs/PROBE_AUDIT.md, exploratory, 120 questions) supports a possible
revision-ranking signal. It does not support an established practical advantage over strong
output controls:

- **Selection:** R1 picks 71–75 of 120 correct answers, against 71 for pooled-six voting. Its
  zero-threshold gate gets 70.
- **Little from revised candidates:** adding them improves on the same scorer's best first answer
  by only 0–2 questions.
- **New correct options are rare:** natural debate supplied a correct option missing from the
  first answers on 1/120 questions; reconsideration alone did so on 9/120.

Two questions are kept apart:

1. Does discussion produce better candidates than additional independent samples?
2. Do internal representations improve selection among the available candidates beyond output-only
   information?

Little new coverage from debate would not rule out debate improving the *assessment* of existing
options.

**What is not claimed.** The six pilot partitions are not replications. An AUROC is not
final-answer accuracy. The probes score completed candidate answers; they are not gates that act
before generation.

## The validation tranche

200 fresh questions (positions 121–320). Shared round 0 feeds natural debate and an independent
branch: matched six-answer pools with identical first halves. Collection and caps:
docs/VALIDATION_JOB.md. Frozen analysis: preregistration §10.

The frozen scorers are fitted on the pilot only, with no recalibration or tuning on validation
outcomes:

- R1 `readout_last:letter:meandiff:15-39` (layers 15, 18, 21, 24, 27, 30, 33, 36, 39);
- R2 `before_stated_answer:letter:pca16:21-33` (layers 21, 24, 27, 30, 33);
- the full-feature output logistic model, the full-feature HGB, and the equal-weight
  activation + output sums.

Any later validation-trained analysis is separate and labelled developmental.

## Reporting hierarchy

1. **Primary practical comparison:** R1 activation best-of-six vs the matched output-only
   best-of-six, natural debate, all collected validation questions. Also the paired difference
   against pooled-six voting.
2. **Key explanatory comparison:** R1 activation vs matched output on revision AUROC (helpful vs
   harmful), and on ranking within the revised panel.
3. **Candidate source:** independent six vs debate six on identical questions (shared round 0):
   coverage, pooled voting, and each fixed selector, including best-of-six minus best first answer.
4. **Secondary:** R2 (complete-case on a common eligible population, and the all-question fallback),
   equal-weight combinations, HGB, zero-threshold gates, missingness sensitivity, reconsideration
   (the optional child).

## Reading the results (practical targets, not established effects or power promises)

Targets for this development decision: **+2 percentage points** of selection accuracy, and
**+0.05** revision AUROC.

- **Promising.** The fixed comparison reaches its target in point estimate, with coherent
  diagnostics and competitive operational performance. An interval that includes zero makes it a
  *provisional lead*, not a demonstrated improvement. An interval entirely above zero is stronger
  evidence.
- **Inconclusive.** The interval includes both no gain and a worthwhile gain. Report the paired
  disagreements (wins and losses) and project the fresh sample size and GPU cost needed to resolve
  it. Lack of significance is not proof of failure. More collection is not launched automatically.
- **Uncompetitive at the chosen targets.** A reasonably narrow interval's upper end excludes the
  worthwhile gain, or the frozen policy clearly loses to simpler controls. The conclusion is scoped
  to these recipes in this setting.

How the three kinds of result combine:

- **Mixed results stay mixed.** Good revision ranking with weak selection is a reason to
  investigate what the score measures. Failure on revision AUROC alone does not settle candidate
  selection, and vice versa.
- **No silent substitution.** Secondary successes suggest a later hypothesis; they do not replace a
  failed primary endpoint.
- **No size changes.** The sample size is not changed after seeing validation outcomes.

## Possible next steps (one is recommended after the analysis; none is launched under this plan)

- **Fresh confirmation of candidate selection**, if its practical advantage looks worth pursuing.
- **A small fixed-candidate / peer-endorsement diagnostic**, if a revision-ranking signal
  reproduces but its meaning or extra value remains unclear.
- **Candidate-generation or debate-protocol research**, if simpler controls dominate: independent
  sampling, reconsideration, or later a carefully implemented round-2 / presentation experiment.

More agents or more rounds are not automatic next steps. Each needs a specific hypothesis, a cost
estimate, and the storage and prompt changes it requires. Most of the remaining budget stays in
reserve.

---

## Results (frozen analysis run once, 2026-10-14; details in docs/VALIDATION_RESULTS.md)

| level | comparison | result | reading |
|---|---|---|---|
| 1 primary | R1 best-of-six − matched output best-of-six (debate, 200 q) | +0.000 [−0.045, +0.045], wins/losses 10/10 | **inconclusive**: no observed difference; about 4,050 fresh questions (≈ 54 raw GPU-h) to resolve ±1 point |
| 1 primary | R1 best-of-six − pooled-six vote | +0.030 [−0.015, +0.075], 14/8 | provisional lead, but the matched output selector gains the same (+0.030) and HGB more (+0.050 [+0.015, +0.090]): not activation-specific |
| 2 key | R1 − matched output, revision AUROC / revised panel | −0.095 [−0.27, +0.09] / −0.118 [−0.31, +0.08] | the pilot's activation advantage **did not reproduce**; formally inconclusive at +0.05 |
| 3 source | independent six − debate six: coverage / pooled vote | +0.060 [+0.025, +0.100] (14 vs 2 q) / +0.020 [−0.005, +0.050] | independent sampling yields more correct candidates; selection results equal across pools |
| 4 secondary | R2; combinations; gates; reconsideration | no R2 advantage; the HGB comparator is the best selector and gate (+4.5 to +5 points over pooled voting); reconsideration shows an activation revision-AUROC hint (+0.12 [−0.06, +0.29], 100 q) | hypotheses for later, not substitutes for the primary |

Mixed results kept separate:

- **Activations:** they still rank correctness on fresh states (round-0 AUROC 0.71, no shift
  degradation), but add nothing measurable to selection beyond output-only scorers, and lost
  their revision-ranking edge in debate.
- **Debate:** it improved the plain vote here (119 vs 116), yet as a source of candidates it was
  dominated by three more independent samples.

## Recommendation: candidate-generation research (simpler controls dominate)

Of the three paths, the evidence supports the third:

- **Not confirmation of activation-based selection:** no difference from the matched output
  selector, and resolving a 2-point gap would cost far more than the remaining budget.
- **Not the endorsement diagnostic:** the revision-ranking signal it would explain did not
  reproduce on fresh debate data.
- **Candidate generation:** independent sampling and output-only selection beat the debate and
  activation machinery that the project set out to test.

**Proposed follow-up (not launched).** An output-only candidate-generation study on fresh
questions from the main pool (positions 321 onward), with:

- an independent-sampling coverage curve: 3, 6, 9 and 12 answers per question;
- selection by vote, matched output logistic and HGB (frozen from pilot + validation);
- debate and reconsideration as fixed comparison arms at matched sample counts.

It needs no activation storage, which cuts disk and readout time.

- **Hypotheses**, stated before data:
  1. coverage and HGB-selected accuracy rise with independent samples;
  2. at matched answer counts, debate does not beat independent sampling.
- **Budget:** used so far about 7.7 A100-h (pilot 3.5 + this phase 4.2), against the
  author's 10–20 h goal. A 300-question, 12-sample study without activation readout is
  estimated at 3–4 raw GPU-h (to be benchmarked). That leaves the remainder in reserve.

More agents or more rounds are not proposed: the data give no hypothesis that needs them. A
paper framing consistent with these results: internal-activation probes reproduce correctness
signal but not incremental value for selection over strong output-only scorers; and in one-round
debate, peers' answers mostly redistribute existing options, which independent sampling supplies
more cheaply.

