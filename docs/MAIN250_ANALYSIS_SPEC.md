# Main collection (main250): headline analysis specification

**Status.** Committed on 2026-10-29, **before any fresh-question (positions 321–520) data
exists**. It fixes the primary analysis so that it cannot be tuned on the results. Anything not
listed here is exploratory and will be labelled so. Design and protocol: docs/NEXT_PHASE_PLAN.md
(M1–M4, third and fourth reviews). Question set: `artifacts/main250/plan/main250-split.json`.

## Data and units

- **Questions:** the 200 fresh questions in the split manifest (dataset revision `b189ec76…`).
- **Protocol:** 250-word guidance and a 1,280-token cap (`configs/main_250.yaml`).
- **Responses:** a pool of 10 independent answers per question. Ordinary and structured debate
  with the nested 3-agent (agents 0–2) and 5-agent (agents 0–4) panels, one round each.
- **Unit of analysis: the question.** Every comparison is paired by question.
- **Coverage:** a question enters an analysis only if every response that analysis needs exists.
  Missing questions are reported by condition, never imputed.

## Score

Each panel's decision is the plurality of its agents' **latest** answers. It is scored as
**exact expected accuracy under uniformly random tie-breaking**: 1/k if the answer key is among
the k tied leaders, else 0 (`mad.analysis.actions.expected_vote`). Answers are the canonical
answers (stated, with a readout fallback where none is stated). The fallback rate is reported per
condition.

## Primary comparisons (all 200 questions; the split is NOT used here)

1. **More discussion vs more independent answers, response-count matched:**
   - debate with 3 agents (6 responses) vs independent voting over 6;
   - debate with 5 agents (10 responses) vs independent voting over 10.

   Each for ordinary and for structured debate.
2. **Structured vs ordinary debate** at each panel size (same initial answers, peer texts and
   peer order).
3. **The independent-voting curve** at panel sizes 3–10 (nested prefixes of the pool), as a
   descriptive reference.

**For each comparison, report:**

- the mean per-question difference;
- a 95% percentile interval from 10,000 bootstrap resamples of questions;
- the number of questions differing;
- the expected gain and expected loss (sums of positive and negative per-question differences).

**Revision events** (per agent and per panel, from `mad.analysis.panels`):

- helpful and harmful changes;
- net expected accuracy change;
- correct minority recovered;
- correct decision preserved or lost;
- correct answer absent from the start but found.

**Multiplicity and interpretation.** The four matched contrasts (comparison 1) are the headline.
Comparison 2 is co-primary for the prompt question. No correction is applied, but the number of
primary contrasts (6) is stated with the results. With 200 questions, small accuracy differences
are not expected to be resolvable; intervals are reported as such, not as tests of "no effect".

**Cost** is reported alongside, measured and not estimated:

- input and output tokens;
- generation time per condition, separately from readout and activation extraction.

## Not primary (exploratory, labelled)

- Subgroups: initial agreement pattern, subject, calculation vs non-calculation.
- Prediction curves for the new responses (extension E3).
- The second-round extension (E1) and the no-peer control (E2), on the 50 questions in
  `artifacts/main250/plan/extension_subset.json` (2 per chunk, seed 20261030, outcome-blind;
  committed before any main250 accuracy). Both run regardless of the primary results.
- **The action predictor / controller.** Trained on the split's *train* questions (74), tuned on
  *validation* (28), evaluated once on *test* (98). Treated as exploratory: 98 test questions
  cannot establish a reliable controller.

## Analysis fixes required before any fresh data is analysed (from the fourth review)

1. **Round-2 rows** must carry round-1 panel metadata (peer answers and agreement relative to the
   answer being revised), not round-0.
2. **Round-2 events** in `panels.compare` and `panel_events` must use the round-1 answers as the
   "before" baseline. Round-1 events keep the original answers.

Both are CPU changes with tests. Neither affects the base collection's primary comparisons,
which use one round only.

**Status: both implemented and tested** (after the G2 pilot launched; the pilot runs the earlier
commit, so its round-2 rows carry round-0 panel metadata and are re-labelled at analysis):

- `multi.run_condition` writes round-1 panel answers for a further round;
- `panels.debate_table` and `panels.compare` take the previous round's answers as the
  `before` baseline;
- `tests/test_panels.py` covers both.
