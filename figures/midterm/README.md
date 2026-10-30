# Midterm figures

Regenerate every matplotlib figure (fig2–fig9) and `plotted_values.md` with one command from the repo root:

```
uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/make_midterm_figures.py
```

matplotlib is not a project dependency, so the command runs in a throwaway environment and leaves `.venv` alone.
The script reads only the small committed reports listed under each figure. It reads no activations,
nothing under `runs/`, and nothing from main250's test split beyond what those reports already state.
It does no GPU work. Each figure is written as a vector PDF plus a 200-dpi PNG preview, cropped to exactly
6.5 in or 3.2 in wide so the fonts print at 9 pt when the figure is included at that width.

**Number checks.** The script looks up every plotted number, as the report prints it, in the matching
`.md` report: 108 strings, all found except one known discrepancy (fig2, below). It exits non-zero if a
new mismatch appears. `plotted_values.md` (generated) has the full plotted tables and the check log.

**Colors** are the same concept in every figure: debate `#2a78d6` (blue), reconsideration `#eb6834`
(orange), independent voting `#199e70` (green), structured debate `#b8429a` (magenta), gates `#3b3b98`
(indigo), and baselines/fixed rules in neutral grey. The five concept colors were checked with a Python
port of the dataviz skill's palette validator (`node` is not installed here). All pairs pass, not only
the pairs that share a figure: protan/deutan OKLab ΔE ≥ 8.4, normal-vision ΔE ≥ 17.6, and contrast
≥ 3.2:1 on white. One caveat: the gate indigo is slightly darker (OKLCH L 0.41) than the validator's
lightness band (≥ 0.43).

Exploratory analyses are marked in the captions. "Primary" means pre-specified in
`docs/MAIN250_ANALYSIS_SPEC.md`.

---

## fig1_pipeline.tex (TikZ)

Needs `\usepackage{tikz}` and `\usetikzlibrary{positioning,arrows.meta,calc}`; use it with
`\input{figures/midterm/fig1_pipeline.tex}`. `fig1_pipeline_preview.tex` is a standalone wrapper for
compiling it on its own. The current version was supplied by the author, who compile-checked it
(fits 6.5 in); it has not been compiled in this environment, which has no TeX installation.

Draft caption: *Protocol. Round 0: three copies of Gemma-3-12B answer independently. In debate, each
agent then reads its two peers' responses and revises once, and the panel answer is the plurality of the
revised answers. We compare three actions after round 0: stop (vote over the round-0 answers), debate,
and adding more independent answers. Snapshots 1–3 mark where internal states and readouts are taken:
after the original answer, after reading peers but before writing, and after the revision.*

## fig2_commitment_curves

Sources: `artifacts/next_phase/commit_curves/core_bf16_bs1/curves.csv` (pool `pooled`, metric
`is_eventual`), `.../analysis.json` (paired gap), checked against `docs/NEXT_PHASE_CPU_RESULTS.md` §12.

| responses | n | 0% | 10% | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|---|---|
| debate, changed | 251 | 0.769 [0.709, 0.825] | 0.761 | 0.809 | 0.904 | 0.952 | 1.000 |
| reconsideration, changed | 308 | 0.146 [0.102, 0.201] | 0.369 | 0.531 | 0.709 | 0.844 | 0.998 |
| debate, unchanged | 662 | 0.967 | 0.977 | 0.977 | 0.986 | 0.989 | 1.000 |
| reconsideration, unchanged | 318 | 0.947 | 0.876 | 0.892 | 0.876 | 0.931 | 0.997 |

Paired gap at 0% (changed answers): +0.629 [0.535, 0.718], 71 questions (60 favour debate, 2 favour
reconsideration). Intervals for every cell are in `plotted_values.md`.

Draft caption: *Exploratory (pilot and validation questions). At each fraction of an agent's own revision
text, the next-token answer readout after the prefix `Answer: (` is compared with the answer the agent
finally states. For answers that changed, debate's readout already matches the eventual answer in 77% of
cases before the agent has written anything; reconsideration's matches in 15% (paired gap +0.63
[0.54, 0.72], 71 questions). After reading peers, the debate answer is mostly settled before the agent
writes its reasoning. Bands: question-cluster bootstrap 95% intervals.*

**Discrepancy.** In §12 of `docs/NEXT_PHASE_CPU_RESULTS.md`, the 10% cell for "reconsideration,
unchanged" reads 0.87. The committed `curves.csv` gives 0.876, which rounds to 0.88. The table predates the
tie-aware rescoring in §15.1: at commit `7da85eb` the same cell was 0.874. §15.1 says the values were
"unchanged at two decimals", which is not true for this cell. The figure uses the committed CSV. No
other cell differs. The readouts use the bf16-logits head; §12 discusses this caveat.

## fig3_voting_curve

Sources: `artifacts/main250/analysis/exploratory.json` §1 (coverage) and §3 (observed and predicted
voting curves, ICC, n_eff), checked against `exploratory.md`. The observed curve also equals
`primary.json` `independent_curve` (asserted in the script and checked against `primary.md`).

| k | ≥ 1 correct | plurality, observed | predicted if independent |
|---|---|---|---|
| 3 | 0.750 [0.690, 0.805] | 0.667 [0.602, 0.730] | 0.788 [0.720, 0.847] |
| 5 | 0.775 [0.715, 0.830] | 0.657 [0.591, 0.721] | 0.909 [0.858, 0.946] |
| 10 | 0.815 [0.760, 0.865] | 0.678 [0.614, 0.740] | 0.989 [0.973, 0.996] |

All k = 3–10 are in `plotted_values.md`. ICC(1) = 0.704 [0.642, 0.761]; n_eff at k = 10 = 1.363
[1.274, 1.475].

Draft caption: *Voting over independent answers (200 main250 questions). Observed plurality accuracy
stays between 0.65 and 0.68 from 3 to 10 answers. If errors were independent at each agent's own
accuracy, it would approach 0.99. At least one correct answer is present for 0.82 of questions at k = 10,
so the right answer is often there but outvoted. Correctness is strongly correlated across samples (ICC
0.70), so 10 answers are worth about 1.4 independent ones. Coverage, the independence prediction, ICC
and n_eff are exploratory; the observed curve is descriptive. Bands: 95% question bootstrap.*

## fig4_accuracy_vs_compute

Sources: `artifacts/main250/analysis/primary.json` (`cost`, `independent_curve`, `revision_events`),
`exploratory.json` §1 (`plurality_after` intervals for the debate arms), checked against `primary.md`
and `exploratory.md`.

Seconds: round-0 cost per response = 11,946.3 s / 2,000 = 5.973 s. Independent k = k × 5.973. A
debate arm adds k × (generation s + extraction s) / revisions:

| arm | seconds per question | panel accuracy [95%] |
|---|---|---|
| debate3 | 3 × 5.973 + 3 × (3,398.2 + 298.9) / 600 = **36.4** | 0.668 [0.602, 0.732] |
| structured3 | 3 × 5.973 + 3 × (5,023.3 + 354.7) / 600 = **44.8** | 0.670 [0.603, 0.733] |
| debate5 | 5 × 5.973 + 5 × (6,197.7 + 715.4) / 1,000 = **64.4** | 0.660 [0.593, 0.723] |
| structured5 | 5 × 5.973 + 5 × (9,379.1 + 810.0) / 1,000 = **80.8** | 0.667 [0.603, 0.730] |
| independent k = 3 / 6 / 10 | 17.9 / 35.8 / 59.7 | 0.667 / 0.655 / 0.678 |

**Single-arm CIs.** `primary.json` stores intervals only for the independent curve and the paired
contrasts. The single-arm intervals for the debate arms are stored in `exploratory.json`
(`1_coverage_vs_selection.debate.<arm>.plurality_after`). They use the same per-question scores and the
same bootstrap (10,000 resamples, seed 0), and their means equal `primary.json`'s `panel_acc_after`
(asserted). Nothing was recomputed.

Draft caption: *Accuracy against measured compute on 200 main250 questions. The green line votes over
3–10 independent answers. Each marker is a debate arm: 3 or 5 agents, ordinary or structured prompt.
Cost is measured wall-clock seconds per question, including answer readouts. Round-0 cost is not
separable from readouts, so it is apportioned per response. At similar cost, no debate arm is
distinguishable from independent voting. None of the pre-specified paired contrasts (debate vs independent voting over 2k answers)
is distinguishable from zero, e.g. debate3 vs 6 independent answers +0.013 [−0.025, +0.050]. Error
bars: 95% question bootstrap.*

## fig5_what_debate_does

Sources: (a) `primary.json` `revision_events.<arm>.per_agent`; (b) `exploratory.json` §4. Checked
against `primary.md` and `exploratory.md`.

| arm | revisions | changed | helpful | harmful | wrong unanimity before → after | paired change |
|---|---|---|---|---|---|---|
| debate3 | 600 | 158 | 51 | 42 | 0.120 → 0.145 | +0.025 [−0.005, +0.055] |
| structured3 | 600 | 139 | 40 | 37 | 0.120 → 0.130 | +0.010 [−0.025, +0.045] |
| debate5 | 1,000 | 272 | 84 | 79 | 0.090 → 0.125 | +0.035 [+0.010, +0.060] |
| structured5 | 1,000 | 206 | 59 | 59 | 0.090 → 0.170 | +0.080 [+0.045, +0.120] |

Before/after intervals are in `plotted_values.md`. In panel (b), Δ (after − before, paired) is printed in
percentage points above each pair, only where its interval excludes 0: debate5 +3.5 pts [+1.0, +6.0],
structured5 +8.0 pts [+4.5, +12.0].

Draft caption: *What one round of debate does (200 main250 questions). (a) Revised answers that went from
wrong to right (up) and from right to wrong (down), per arm; counts are agents, out of 600 (3 agents) or
1,000 (5 agents) revisions. Helpful and harmful revisions roughly cancel. (b) Exploratory: share of
questions where every agent gives the same wrong answer, before and after revision. Debate never lowers
it, and with 5 agents it rises (paired Δ shown where its 95% interval excludes 0). Debate pushes panels
toward consensus whether the consensus is right or wrong.*

## fig6_signal_auroc

Source: `artifacts/main250/analysis/selection_signals.json` `within_question`, checked against
`selection_signals.md` (within-question table).

| answers | mean log-prob: ordinary → within | stated confidence: ordinary → within | peer-exposure readout: ordinary → within |
|---|---|---|---|
| 10 independent | 0.648 → 0.555 | 0.674 → 0.560 | — |
| debate3 | 0.632 → 0.542 | 0.788 → 0.585 | 0.771 → 0.542 |
| structured3 | 0.600 → 0.641 | 0.788 → 0.641 | 0.807 → 0.688 |
| debate5 | 0.643 → 0.461 | 0.809 → 0.610 | 0.787 → 0.685 |
| structured5 | 0.584 → 0.579 | 0.789 → 0.688 | 0.818 → 0.699 |

Intervals and pair counts are in `plotted_values.md`. Within-question pairs are few after debate (64–232
pairs in 32–45 questions), so those intervals are wide.

Draft caption: *Exploratory. AUROC of three signals for whether an answer is correct. Open: ordinary
AUROC over all answers. Filled: within-question AUROC, comparing only correct and incorrect answers to the
same question. After debate, confidence and the peer-exposure readout reach about 0.8 ordinarily but fall to 0.54–0.70 within
questions. (Ordinary AUROC for confidence is 0.67 on the independent answers.) Most of their apparent skill comes from ranking easy questions above hard ones, not from
picking the right answer within a question. Lines: 95% question bootstrap; dashed: chance.*

## fig7_revision_gate

Sources: `artifacts/revision_gate/feasibility.json` (`simulation["all panels"]`,
`models.*.auroc`), checked against `feasibility.md`. (`oracle.json/.md` holds the perfect-gate ceilings per
dataset; the pooled perfect-gate value plotted here is the one in `feasibility.json`.)

| policy | gain over accept-all (panel accuracy) |
|---|---|
| reject every revision | +0.006 [−0.019, +0.031] |
| (a) accept iff new answer = peer majority | +0.007 [−0.016, +0.029] |
| (b) conviction: reject if confidence before ≥ 90 | +0.006 [−0.015, +0.027] |
| (c) MAD-Conformist: keep if old = previous majority | +0.006 [−0.019, +0.030] |
| logistic gate (AUROC 0.707 [0.618, 0.794]) | +0.018 [−0.000, +0.036] |
| boosted-tree gate (AUROC 0.727 [0.659, 0.795]) | +0.017 [+0.002, +0.032] |
| perfect gate (oracle) | +0.078 [+0.059, +0.097] |

Accept-all panel accuracy: 0.616 [0.568, 0.665]; 1,016 panels from 422 questions.

Draft caption: *Exploratory. Simulated panel accuracy when a gate decides whether to accept each changed
answer, pooled over 1,016 panels from all Gemma-3-12B debate datasets (main250 test split excluded). Shown
as the gain over accepting every revision. Gates use output features only (confidence, answer readouts,
panel agreement, length, log-prob). They are trained with question-grouped cross-validation and evaluated
out of fold. They separate helpful from harmful revisions moderately well (AUROC 0.71–0.73) but gain only
about 2 points, a quarter of the perfect gate's 7.8. Fixed rules do no better than rejecting every
revision. Lines: 95% question-grouped bootstrap.*

Note for the text: the pooled set includes the pilot's `misleading_one` condition, where rejecting is
easy. Excluding it from evaluation, the gains are +0.011 [−0.008, +0.030] (logistic) and +0.010
[−0.005, +0.025] (boosted tree) (`feasibility_followup.md` §iii). Within-question gate AUROC is 0.600 /
0.646 (§i). Neither is plotted.

## fig8_probe_agreement (optional; data committed)

Source: `artifacts/val12b/investigation/shifts.json` (`activation_validation`), checked against
`docs/R1_INVESTIGATION.md` item 2–3. The doc's kept-answer table gives point estimates only. The
intervals come from the JSON (10,000 question resamples).

| (a) debate, kept answer | n agents | mean shift [95%] |
|---|---|---|
| unanimous, kept right | 288 | +0.504 [+0.453, +0.557] |
| unanimous, kept wrong | 96 | +0.513 [+0.404, +0.625] |
| two against one, kept right | 19 | −0.442 [−0.775, −0.140] |
| two against one, kept wrong | 23 | −0.551 [−0.922, −0.172] |

(b) Disputed minus unanimous, all agents: debate −1.218 [−1.400, −1.034]; reconsideration −0.003
[−0.326, +0.316]; fresh independent sample +0.007 [−0.081, +0.094]. Not plotted: three-way splits with a
kept answer (5 agents, all initially wrong).

Draft caption: *Exploratory (validation tranche, 200 questions). (a) Change in the frozen R1 activation
probe's score between an agent's first and revised answer, for debate agents who kept their answer. The
score rises when the panel was unanimous and falls when it was split, by the same amount whether the kept
answer was right or wrong. (b) The effect needs peer exposure. It is absent when agents reconsider alone
or answer afresh. The probe tracks how much the context endorses the answer, not whether the answer is
correct. Bars: 95% question bootstrap.*

## fig9_peer_order (optional; data committed)

Source: `artifacts/next_phase/exposure_commit/summary.json` (`echo_check`), checked against
`docs/NEXT_PHASE_CPU_RESULTS.md` §3.

| pool | revisions with two different peer answers | adopted last-shown | adopted first-shown | neither |
|---|---|---|---|---|
| validation | 130 | 91 | 34 | 5 |
| pilot | 87 | 52 | 29 | 6 |

The exposure readout shows the same pattern (validation 95 vs 33; pilot 63 vs 22); it is not plotted.

Draft caption: *Exploratory. Among debate revisions where the two peers shown gave different answers
(peer order randomized per agent), the agent's final answer matches the peer shown last 2.7 times as
often as the peer shown first in the validation pool (91 vs 34), and 1.8 times as often in the pilot
(52 vs 29). The revision is sensitive to presentation order, not only to content.*
