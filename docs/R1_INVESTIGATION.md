# Why the frozen R1 selector never changes its pick when debate answers are added (2026-10-28)

CPU only. Exploratory: the question, the alternative scorers and the corrections were chosen
after seeing the 200-question validation tranche. Code: `scripts/investigate_r1.py`. Results:
`artifacts/val12b/investigation/`. Uncertainty: bootstrap over questions (10,000 draws for
shifts, 2,000 for rankings). An agent's first and revised states, and the three agents of a
question, are never treated as independent.

## Demonstrated

1. **The finding is real, and not an implementation error** (`reproduce.json`).
   - **Score reproduction.** The exported R1 scores were recomputed two ways: from the analysis
     cache, and from the raw archive vectors with a hand-written formula (letter centering,
     standardization, per-layer direction, layer and ensemble normalizers, Platt). Both match to
     5×10⁻⁷, the export's rounding.
   - **Indexing and inputs.** Every revised state has its agent's first state. The two condition
     views hold identical round-0 vectors. The answer letter used for letter centering equals the
     letter written in the response in 1,744 of 1,744 stated answers. Platt probabilities follow
     from the stored coefficients.
   - **The counts.** On debate, the option chosen from the six pooled answers equals the option
     chosen from the three first answers on **200 of 200** questions. Choosing from the revised
     answers alone changes it on 23. With the independent branch, the six-way pick changes on 31.
2. **After seeing peers, an agent's score shifts with the panel's initial agreement.** This is
   true even when the agent keeps its answer (`shifts.json`). Mean revised-minus-first R1 score
   on debate:
   - unanimous questions: +0.480 [+0.42, +0.53] (390 agents);
   - two-against-one: −0.717 [−0.89, −0.54] (153);
   - three-way splits: −0.797 [−1.19, −0.36] (57);
   - disputed minus unanimous: −1.22 [−1.40, −1.03].

   The pattern holds for agents who did *not* change, whether their answer was right or wrong:

   | agents who kept their answer | shift |
   |---|---|
   | unanimous, right | +0.50 |
   | unanimous, wrong | +0.51 |
   | two-against-one, right | −0.44 |
   | two-against-one, wrong | −0.55 |

   So the shift tracks the context, not the answer's correctness. It appears at every one of the
   9 layers (unanimous +0.24 to +0.67, disputed −0.56 to −1.05).
3. **The shift needs peer exposure.**
   - **Independent answers:** fresh samples of the same prompt, no peers. The shift is +0.03
     overall and disputed minus unanimous is +0.01 [−0.08, +0.09].
   - **Self-reconsideration:** the agent sees only its own answer. The shift is a constant
     −0.93, the same for all agreement levels (difference −0.00 [−0.33, +0.32]).
   - **The output model shows a smaller version in debate:** disputed minus unanimous −0.42
     [−0.55, −0.30].
4. **It was already in the training data.** The pilot's out-of-fold R1 scores show the same
   debate pattern (unanimous +0.41, two-against-one −0.60, three-way −0.88) and the same in the
   answer-only variant.
5. **This produces the unchanged picks** (`counterfactual.json`).
   - **Disputed questions:** the 70 debate questions that are not unanimous are the only ones
     where a different option could win. The six-way maximum is a revised state on just 12 of
     them, and that state never holds a better-scored new option.
   - **Unanimous questions:** the revised states win 125 of 130 times, but every answer there is
     the same option.
   - **Correction:** subtracting each question's own mean shift from its revised scores
     (label-free) makes the six-way pick differ on 11 questions. Six-candidate AUROC rises from
     0.488 to 0.577, and best-of-six goes from 122 to 124 correct.
   - **Still fewer changes than independent (31):** most debate revisions adopt options already
     among the first answers, so there is little new to choose.

## Tested and *not* supported

**R1's across-question training is not the main cause.** A within-question scorer learns its
direction only from correct-versus-incorrect contrasts inside each question
(`within_question_direction`). It does no better (`compare.json`):

- **Option changes:** on debate validation its six-way pick still changes on 0–3 questions,
  against 24–32 with independent answers.
- **Six-candidate AUROC:** 0.47–0.55, against 0.49–0.58 for R1.
- **Training set:** the same holds whether it is trained on first answers only, on first answers
  plus debate, or on all pilot conditions.

Matched output-only scorers, with the same two estimators, also change picks rarely on debate
(4–14) and more often on independent answers (10–25).

All variants select 118–129 of 200 correct, within noise of one another and of the frozen
result (122). No scorer–training-set combination stands out once the number of comparisons is
considered.

## Hypotheses (not demonstrated)

- **What the activation direction reads.** The readout-position activation partly encodes how
  much the context endorses the agent's answer, which is high in a unanimous panel and contested
  in a split one. A direction that separates correct from incorrect states across questions
  absorbs that endorsement signal, because consensus correlates with correctness.
- **Why within-question training doesn't help.** The shift is in the activations themselves, a
  change of input context, so any linear readout of "confidence" inherits it. Reweighting the
  training contrasts cannot remove it.
- **Self-reconsideration** may induce a generic "doubt" state, giving a uniform offset.

## Recommended next experiment

A **fixed-candidate context readout** (GPU, readout only, no generation; under 0.5 GPU-hour for
the 200 validation questions):

- **Design:** teacher-force each *existing* first and revised response into three contexts:
  1. no peers;
  2. peers who agree with it;
  3. peers who disagree with it.

  Read R1's position in each. Nothing is generated.
- **What it decides:**
  - whether the score shift is caused by endorsement, with the candidate text held fixed;
  - whether scoring every candidate in one *neutral* context restores within-question ranking
    and lets the selector use debate answers.

This is the most informative next step for the probe question. For the project's practical
question, the candidate-generation study in docs/NEXT_STEP_MEMO.md remains the recommendation,
since debate supplied few new correct options here (3 vs 15 for independent sampling).
