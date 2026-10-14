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

*(Results and the recommendation are appended below after the frozen analysis.)*
