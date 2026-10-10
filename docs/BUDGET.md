# GPU budget and disk: estimates before any 12B run

**Status:** first-principles estimates. Nothing has run on a GPU. The smoke test measures the
real seconds per question and memory; the pilot report (`scripts/pilot_report.py`) replaces
every number here with a measured one. Set the main-run size only after that.

Target: **10–20 A100-hours in total**, well under the ~56 available.

## How the per-question cost is estimated

Generation dominates. Gemma-3-12B decoding on one A100-40GB is limited by reading ~24 GB of
weights per step: roughly 35–45 ms per step in Hugging Face `generate`, almost independent of
batch size at these sizes. A batch runs until its longest response finishes; with the ~150-word
brevity instruction, assume ~450 steps for the longest of 8 (MMLU-Pro math runs long).

| item | batch 8 | batch 4 |
|---|---|---|
| one response (450 steps × 40 ms ÷ batch) | ~2.3 s | ~4.5 s |
| one readout (prefill ~1,500 tokens + hidden states) | ~0.4 s | ~0.4 s |
| round 0 (3 responses + 3 readouts) | ~8 s | ~15 s |
| one condition (3 responses + 3 exposure + 3 post readouts) | ~9 s | ~16 s |
| **per question, 2 conditions** | **~26 s** | **~47 s** |
| **per question, 3 conditions** | **~35 s** | **~63 s** |
| **per question, 4 conditions (pilot)** | **~45 s** | **~79 s** |

Uncertainty is about ±50%. Batch 8 is plausible but unverified: Gemma 3 uses 1,024-token sliding
windows in 5 of 6 layers, so the generation cache at the worst-case length is ~0.6 GB per
sequence, not ~1.5 GB. The smoke test's stress pass tries batches 4, 6 and 8 at the worst-case
lengths of the actual question pool and reports the largest that fits.

## Plans

| step | questions | conditions | hours (batch 8 / batch 4) |
|---|---|---|---|
| smoke + stress (incl. 24 GB download, load) | 2 | 4 | 0.4 / 0.5 |
| pilot (pilot pool) | 120 | self, debate, answer-only, misleading | 1.5 / 2.6 |
| **A. lean main** | 600 | debate, self | 4.3 / 7.8 |
| **B. recommended main** | 900 | debate, self (full) | 6.5 / 11.8 |
| B. + answer-only peers on a matched subset (child job) | 450 | answer-only | 1.1 / 2.0 |
| **C. fuller main** | 900 | debate, self, answer-only (full) | 8.8 / 15.8 |
| candidate evaluation (readout only, 6 per question) | 300 | — | 0.2 / 0.2 |
| three fresh independent answers (child job) | 300 | independent | 0.8 / 1.3 |

Totals including smoke, pilot, candidates, independent answers, and 15% contingency for
failures, reruns and idle runtime:

| plan | batch 8 | batch 4 |
|---|---|---|
| A (600 × 2) | **8.3 h** | **14.3 h** |
| B (900 × 2 + answer-only on 450) | **12.1 h** | **21.2 h** |
| C (900 × 3) | **13.5 h** | **23.5 h** |

**Recommendation:** run the smoke test and pilot (~2–3 h), then choose. If batch 8 fits and the
pilot measures ≤ 30 s/question for two conditions, plan B (~12 h) or C (~13.5 h). If only batch 4
fits, plan A (~14 h), or B with fewer questions. Every plan stays under 25 h even in the slow
case, leaving more than half the allocation for reruns or a second model size.

The budget ledger (`<name>-ledger.jsonl`) records model loading, stress passes, failed chunks
and every session's wall time. It cannot see Colab idle time outside the driver: check the unit
balance (`colab usage`, or the Colab UI) after each session.

## What these sample sizes can and cannot show

From `scripts/pilot_report.py` (two-sided 5%, 80% power), the test questions needed to detect a
difference in **panel accuracy** between two gates:

| share of questions the two gates decide differently | 3 points | 4 points |
|---|---|---|
| 5% | 386 | 190 |
| 10% | 849 | 467 |
| 15% | 1,293 | 720 |

With 900 questions and half for test (450), only a **~4–5 point panel difference is detectable**
unless the gates rarely disagree. Agent-level comparisons are better powered: ~1,350 held-out
agent answers per condition, ~300–450 of them revisions, for ranking (AUROC) differences between
activation and output-only predictors with question-clustered intervals. The preregistration must
say which is primary; see `docs/preregistration.md` §5.

## Disk

Per stored state: 49 layers × 3,840 × 2 bytes = 0.376 MB per position. Round-0 states are stored
once per chunk and referenced by each condition (not copied).

| per question | states | MB |
|---|---|---|
| round 0: 3 agents × 4 positions | 12 | 4.5 |
| one condition: exposure 3 × 2 positions + post 3 × 4 | 18 | 6.8 |
| 2 conditions | 48 | 18.1 |
| 3 conditions | 66 | 24.8 |
| 4 conditions (pilot) | 84 | 31.6 |

| run | GB |
|---|---|
| pilot (120 × 4) | 3.8 |
| plan A (600 × 2) | 10.9 |
| plan B (900 × 2) + answer-only child (450) | 19.4 |
| plan C (900 × 3) | 22.3 |
| candidates (300) + independent answers (300) | ~3.5 |

**Before the pilot, confirm the Google Drive quota** for the account running Colab: plan C with
the pilot and extras needs ~30 GB. Archive chunks and round-0 caches, not only views (views
reference them), and run `scripts/verify_archive.py --copy-to` on anything moved off Drive.
