# Does the internal state before the revision predict the outcome?

## Procedure

Data: `runs/night3-debate`. gemma-3-1b-it, ARC-Challenge, 160 questions x 3 agents = 480 events,
of which 139 revised and 243 started correct. 192 generation tokens.

Three states recorded per agent, each the residual stream at the token after `Final answer: (`:

| state | when |
|---|---|
| pre | after the agent's own original reasoning |
| exposure | after reading the two peers and the reconsider instruction, before writing anything |
| post | after writing its revision |

Target: is the agent's final answer correct.

8 random splits, 70% of questions train / 30% test, every row of a question on the same side.
Per split, a logistic regression on the 1,152-number state; layer (from 6 depths) and
regularisation (4 values) chosen by 3-fold grouped cross-validation **on training questions only**.

Compared against, on identical splits:
- **output-only**: logistic regression on the forced-answer confidence and letter at pre and at
  exposure, and whether the letter changed between them. No internal state.
- **constant**: always predicts the training-set rate of correct answers.

Approximate test-set sizes per split: ~144 events, ~42 revisions, ~73 initially correct.

Not run here: the combined output-plus-activation model. So these results can show whether
activations beat outputs on their own, but not whether they add anything on top of outputs.

## Results (ranking score, 0.5 = coin flip; mean ± sd over 8 splits)

The eight splits reuse the same 350 questions in overlapping test sets, so they are not
independent replications, and the ± figures are spreads across splits, not confidence intervals.

| | all events | revisions only | initially correct |
|---|---|---|---|
| constant | 0.500 | 0.500 | 0.500 |
| **output-only** | **0.644 ± 0.052** | 0.500 ± 0.073 | **0.887 ± 0.030** |
| pre | 0.571 ± 0.062 | 0.501 ± 0.069 | 0.627 ± 0.086 |
| **exposure** | 0.568 ± 0.079 | 0.439 ± 0.105 | 0.634 ± 0.110 |
| post | 0.605 ± 0.043 | 0.511 ± 0.087 | 0.705 ± 0.102 |
| pre + exposure | 0.567 ± 0.074 | 0.431 ± 0.118 | 0.652 ± 0.085 |
| pre + post | 0.592 ± 0.060 | 0.498 ± 0.069 | 0.683 ± 0.086 |
| all three | 0.581 ± 0.055 | 0.494 ± 0.067 | 0.683 ± 0.081 |

## What the numbers show

1. **No internal-state representation beats the output-only model** on all events
   (best 0.605 vs 0.644) or on initially-correct agents (best 0.705 vs 0.887).
2. **The exposure state is not better than the pre state** (0.568 vs 0.571 on all events). The
   pre-revision moment does not carry extra signal that a linear probe can extract.
3. **On revision events, nothing works.** Every predictor — activation at any moment, output-only,
   constant — sits between 0.43 and 0.51. This is consistent with the earlier null, in a separate
   run with longer generations and a new state.
4. **The strong result in the table is output-only on initially-correct agents, 0.887.** It is
   driven mostly by whether the forced instant answer has already moved at the exposure moment.

## What it might mean, and what it does not establish

The exposure moment does contain information about whether a correct agent will end up wrong —
but that information reaches us through the model's forced letter at that moment, not through
anything the linear probe recovers on top of it.

Competing explanations for the 0.887, not separated by this data:
- the peers' arguments shifted the model toward their answer before it wrote anything; or
- with no reasoning step, the model echoes an answer letter that appears just above the readout
  point in the peers' text. Earlier check: when an agent's instant answer moved and both peers
  agreed, it moved to the peers' answer 67% of the time against ~33% for a random switch.

Within the initially-correct subset, any change of answer is a change to a wrong answer, so a
"did it move" feature is expected to be strong there. That is the review's design and is
legitimate, but it is why output features look so much better on this subset than on others.

The test sets are small and the spread across splits for activation probes is 0.04-0.12, so the
activation representations cannot be ranked against each other. This document does not claim to
know why the probes underperform.

## Bearing on the project

The pre-revision state was the last untested route for the activation direction. It does not
change the picture: at this model size, internal states do not predict the accept-or-reject
decision better than the model's own outputs, at any of the three moments.
