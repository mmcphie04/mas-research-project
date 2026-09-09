# Execution Plan

**Project:** Using Internal Representations to Evaluate Answer Revisions in Multi-Agent Debate
**Author:** M. McPhie · **Advisor:** Prof. Mengyu Wang · Harvard CS Independent Study, Fall 2026
**Written:** 2026-09-09 · **Companion:** [`RISKS.md`](RISKS.md)

---

## 1. Context

Multi-agent debate (MAD) can improve reasoning, but agents also exhibit peer conformity —
abandoning correct answers after exposure to misleading peer reasoning. Existing mitigations rely on
pre-debate confidence or verbalized uncertainty, both poorly calibrated. This project tests whether
a language model's **internal activations** carry a signal about the quality of a proposed answer
revision that is not available from its verbalized or token-level confidence, and whether that
signal can be used to gate revisions and improve final panel accuracy.

**Setup.** Three homogeneous Gemma agents answer an MMLU-Pro question independently (round 0), then
each sees the other two agents' rationales and answers and produces a revised answer (round 1).
Exactly one round. Residual-stream activations are captured before and after peer exposure. Where an
agent changed its answer, the change is labeled **beneficial** (wrong→right), **harmful**
(right→wrong), or **unhelpful** (wrong→wrong).

**Deliverables.** Progress report ~2026-10-26. ~20-page analytical paper at the start of Reading
Period, early December 2026.

**Environment.** WSL2 / Ubuntu, 13 GB RAM, no local CUDA, 947 GB free disk, `uv` available.
GPU access unresolved as of week 1.

**Confirmed parameters.** N = 3,000 MMLU-Pro questions (pilot-adjustable) · primary model
Gemma-3-12B-it · three-tier model backend so code correctness is decoupled from GPU availability.

---

## 2. The structural property this plan is built around

**With exactly one round, the intervention stage requires zero new inference.** The gate only ever
chooses, per agent, between the round-0 answer and the round-1 answer — both already generated.
Every gating policy, threshold, baseline, and oracle is evaluated offline from a single collected
dataset, on a laptop.

Three consequences:

- GPU access is on the critical path **only for data collection** (weeks 3–4). Everything else is CPU.
- The real risk shifts from *can I run the experiment* to *will I overfit the test set by
  re-analyzing it*. The test split is sealed until week 8 and every read after that is a documented
  amendment.
- **Do not extend to two rounds.** With ≥2 rounds the gate changes what peers see downstream, which
  makes the probe off-policy and forces re-running inference for every candidate policy. One round is
  not a limitation here; it is what makes the project tractable in a semester. The paper should say so.

---

## 3. Design decisions that depart from the original proposal

Five changes. Everything else follows the proposal as written.

### 3.1 Panel-level metrics, not just net revision gain

Majority voting is nonlinear. Agent-level *B − H* does not translate into panel accuracy: a harmful
revision only matters if it flips the majority, and with three agents over ten options many flips are
absorbed or land in three-way ties. A design can post +150 net revisions and +0.4 points of panel
accuracy.

All oracles and metrics are computed at **both** the agent and panel level, and the go/no-go gate
uses the panel-level number.

**Pre-registered tie-break for 1-1-1 outcomes:** fall back to the round-0 plurality; if that also
ties, take the lowest agent id. The tie-break is deliberately **neutral**: with ten options and a
weak model, three-way splits run around 40% of questions in tier-0 smoke data, so breaking ties by
confidence would quietly turn the panel metric into a confidence baseline and make gated-vs-ungated
partly a comparison of tie-break quality. `round0_then_confidence` is reported as a variant.
Tie frequency is reported alongside every panel number.

### 3.2 There is no non-degenerate "revision quality" label

Among revision events with a unique correct option, **every** binary partition of {B, H, U} reduces
to a single-state correctness label:

| Partition | Equivalent to |
|---|---|
| B vs. (H ∪ U) | correct(post) |
| H vs. (B ∪ U) | correct(pre) |
| U vs. (B ∪ H) | neither correct |

B and H are mutually exclusive by construction. The probe is therefore **not** learning a separate
"revision quality" construct at the label level and must not be described as doing so.

**The study is a representation × target grid.** Correctness is the label; the *representation* is
the independent variable.

| | target `y_pre` = correct(pre) | target `y_post` = correct(post) |
|---|---|---|
| **φ_pre** | (a) standard pre-state correctness probe | (c) pre-state forecasts revision outcome |
| **φ_post** | (b) post-state recall of prior correctness | (d) standard post-state correctness probe |
| **φ_paired** = [pre; post; post−pre] | (e) | (f) |

> **Central question: does the transition add predictive value beyond the corresponding single-state
> probe?** That is (f) vs. (d), and (e) vs. (a), by paired DeLong on the same held-out revision
> events.

Cell (c) is independently interesting and genuinely non-degenerate: forecasting a not-yet-observed
revision's outcome from the pre-state alone.

**Intervention.** Train and calibrate `ĝ_pre` (estimating P(correct(pre))) and `ĝ_post` (estimating
P(correct(post))) on whichever representation wins its column, then

```
accept the revision  iff  ĝ_post − ĝ_pre > τ
```

with τ tuned on validation to maximize simulated panel accuracy. τ ≠ 0 encodes risk aversion about
defecting from a possibly-correct answer. Report the simpler single-probe rule (accept iff
`ĝ_post > τ`) alongside it. **Calibration is a core result, not an add-on**, because the decision rule
compares two probabilities.

**Two consequences.**

1. The correctness target is defined for *every* agent-state, not just revision events, so probes
   train on all ~18,000 pre+post states rather than ~700 revision events — a ~25× increase that
   removes probe training as a sample-size constraint. Ablate against a probe trained only on
   revision events to confirm the distribution shift is benign.
2. Because the label is now plain correctness, the **difficulty-detector shortcut is more acute, not
   less**. The within-question conditional AUROC test (RISKS.md §1) is the primary guard and is
   reported as a headline number, not a robustness footnote.

### 3.3 Forced-format readout pass

"Pre- and post-discussion activation vectors" is underspecified, and the naive choices are wrong: an
activation at the last prompt token cannot know *which* answer is being proposed, and pre/post states
after free-form generation sit at different context lengths, so any difference vector partly encodes
"how much text is in my context."

**Fix.** After free-form CoT generation, run one extra forward pass appending a fixed template
(`\nFinal answer: (`) and read the residual stream at that token, plus the logits over the ten
option-letter tokens. This yields:

- a canonical, identical, answer-bearing extraction position for pre and post, with no length confound;
- a guaranteed parse — argmax over letter tokens never fails, so parse failures become a *measured
  disagreement rate* between free-text and readout rather than dropped rows;
- a free token-probability confidence baseline (max softmax over the ten letters), which is the
  strongest cheap competitor the probe has to beat.

Both the free-text and readout answers are recorded, and their disagreement rate is reported. **The
readout answer is canonical.**

### 3.4 Pre-registered metric hierarchy

The end-to-end endpoint is underpowered at achievable N; the probe endpoint is not.

| | Endpoint | Unit | Power at N = 3,000 |
|---|---|---|---|
| **P1** | Probe AUROC vs. confidence and metadata baselines (paired DeLong) | ~300–480 test revision events | Strong |
| **P2** | Agent-level accuracy / net revision gain under gating | ~350 discordant agent-instances | Strong |
| **P3** | Panel accuracy under gating vs. ungated debate (McNemar) | 900 test questions | Marginal — report with CI |

**Commitment rule.** At the week-3 pilot gate, using pilot data only and before the test set is ever
unsealed: declare P1 and P3 co-primary only if the pilot's correctness-probe AUROC reaches **≥ 0.80**
*and* the panel-optimal oracle headroom clears G5; otherwise P1 primary with P3 secondary. Written
and dated into `docs/preregistration.md`.

> **Revised 2026-09-09 after simulation.** The original trigger (oracle panel-flip rate *f* ≥ 0.15)
> is the wrong one, and the analytic power estimate below it was too optimistic. Simulating the
> actual protocol — τ tuned on validation, evaluated once on test — gives, at N = 3,000:
>
> | probe AUROC | mean τ | Δ panel (pts) | discordant | power |
> |---|---|---|---|---|
> | 0.65 | 0.10 | −0.03 | 28 | 0.00 |
> | 0.70 | 0.10 | +0.22 | 25 | 0.15 |
> | 0.75 | 0.22 | +0.31 | 46 | 0.10 |
> | 0.80 | 0.26 | +0.66 | 55 | 0.35 |
> | 0.85 | 0.25 | +1.11 | 52 | 0.70 |
> | 0.90 | 0.32 | +1.59 | 64 | 1.00 |
>
> The reason is structural, not a simulation artefact: beneficial revisions outnumber harmful ones
> (B ≈ 658 vs H ≈ 529 per 3,000 questions), so **accepting every revision is already net-positive**
> and the gate has to clear a bar that ungated debate has set for it. A validated threshold on a
> moderate-AUROC probe is therefore nearly degenerate — it accepts almost everything and barely
> deviates from standard MAD. A +3-point panel gain would need AUROC ≈ 0.95, which is not a
> plausible target. Reproduce with `mad power --by-auroc 0.65 0.75 0.85 --expect-n 3000`.

This lets the framing adapt to evidence without letting it adapt to the test set. Choosing the
primary endpoint after seeing results is the standard mechanism for producing an unreproducible
finding, and it is the first thing a careful reader checks.

### 3.5 Three-tier model backend

So the entire codebase is developed and tested with no GPU. See §4. Corollary: the analysis layer is
built and unit-tested in **week 2**, not week 5.

---

## 4. Backend ladder

One `model:` config block with `backend: stub | hf | vllm`. Swapping tiers is a config change.

| Tier | Backend | What it is | Runtime | What it validates |
|---|---|---|---|---|
| **0** | `stub` | No model. Scripted LM returning canned/templated responses plus seeded fake hidden states of correct shape and dtype. **Configurable to plant a known B/H/U distribution and a known probe-signal strength.** | ms | Schemas, determinism, split integrity, storage round-trip, and the full analysis layer against known ground truth |
| **1** | `hf` + Gemma-3-270M-it | Real tokenizer, real generation, real hidden states, tiny. Fits 13 GB RAM easily. | seconds–minutes for 5–20 q | Prompt templates render, readout token indexing, letter-logit extraction position, activation shape/dtype, end-to-end wiring |
| **2** | `vllm` / `hf` + Gemma-3-12B-it | The real run. | hours on 1×A100 | Science |

**Tier 0 is the important one and is easy to under-build.** A 270M model is at chance on MMLU-Pro and
produces no realistic revision distribution, so tier 1 validates plumbing but not logic. Tier 0 with
planted ground truth is what proves the labeler, panel vote, tie-break, four oracles, gating simulator
and McNemar code are correct — in week 2, not week 8.

Tier 0 also enables a **power simulation** with zero compute: plant B/H/U at assumed rates, simulate a
probe at a given AUROC and calibration, read off panel-accuracy gain and its CI as a function of N and
*f*. This sharpens the pilot thresholds before the pilot runs, and is a figure in the paper.

**Fallback ladder** if Gemma-3-12B-it misses the G1 accuracy band: Gemma-2-9B-it (lower accuracy →
more revisions, fits a 24 GB card) or Gemma-3-27B-it (80 GB card; week-10 stretch comparison only).

---

## 5. Architecture

```
mas_research_project/
  pyproject.toml            # uv, Python 3.12
  configs/                  # YAML; hashed into manifests. backend + model + prompts + seeds
  src/mad/
    backends/               # stub.py | hf.py | vllm.py behind one interface
    data.py                 # MMLU-Pro load, subject stratification, frozen grouped split
    permute.py              # option permutation + canonical-label mapping, seeded
    prompts/v1/             # round-0, round-1 debate, round-1 self-reconsider, readout
    generate.py             # batched generation -> JSONL transcripts
    readout.py              # prefill pass -> hidden states + letter logits
    store.py                # fp16 shards + Parquet index; per-layer streaming reads
    provenance.py           # manifest.json before run, completion.json after
  src/mad/analysis/         # CPU only, laptop, built in week 2
    labels.py               # B/H/U labeling, panel voting, tie-break
    probes.py               # representation x target grid; L2-logistic, mass-mean, PCA; GroupKFold
    calibrate.py            # Platt/isotonic, ECE, reliability diagrams
    baselines.py            # confidence, metadata, random-gate, permutation test
    gating.py               # offline policy simulation, threshold selection
    stats.py                # bootstrap-by-question, DeLong, McNemar
    power_sim.py            # synthetic power/N analysis (week 2)
  tests/                    # tier-0 ground-truth assertions
  runs/<run_id>/            # transcripts, activations, manifest, completion
  docs/                     # PLAN.md, RISKS.md, preregistration.md, pilot_report.md
```

**Activation index row:** `(run_id, question_id, agent_id, phase, condition, layer, position)`.

**Split grouping key** is the *underlying question id* — it spans agents, permutations, and replicate
runs. Enforced by an assertion test, not by convention.

**Storage rule: fixed token positions only.** Per state = (n_layers + 1) × d_model × 2 bytes
(≈ 376 KB for a 12B model). 18,000 states × 1 position ≈ 6.8 GB; a single layer at *all* token
positions would be ≈ 1.8 TB. Layer sweeps are affordable; token sweeps are not.

**RAM:** read one layer at a time — 18,000 × 3840 × fp16 ≈ 138 MB, trivial in 13 GB. Never load a full
shard.

**Permutation:** one shared option permutation per question per run, so peer messages' letter
references stay coherent without translation. Randomized across questions; seed and mapping recorded.

---

## 6. Stage 1 pilot (week 3) and go/no-go

**Scale.** 300 questions, subject-stratified (≈ 21 × 14 subjects), plus a 150-question
self-reconsideration control. ~1.5–2 GPU-hours. Prompt development happens on the MMLU-Pro
*validation* split (~70 q), never on test. Because the analysis layer already exists, the pilot report
generates in hours.

The oracle ceiling alone is **necessary but not sufficient** — it shows headroom exists but says
nothing about whether a linear probe can reach it, whether enough labels will accumulate, or whether
the effect survives to the panel level. Measure all thirteen:

**Feasibility — does the phenomenon exist?**
1. Round-0 single-agent accuracy, overall and per subject, with CIs.
2. Initial agreement distribution: P(3-0), P(2-1), P(1-1-1).
3. Round-0 plurality vs. self-consistency@3 vs. round-1 standard MAD accuracy. *Does debate help at all?*
4. Per-agent revision rate, split by pre-correctness (`r_C`, `r_I`).
5. B / H / U counts and rates.
6. Conformity signature: P(switch | peers disagree) vs. P(switch | peers agree); P(switch lands on peer majority | switched).
7. Self-reconsideration control revision rate and B/H/U mix — the peer-attribution baseline.

**Headroom — is there room for a gate?**

8. Five policies at agent *and* panel level: standard MAD, round-0 plurality, reject-harmful-only, `oracle_full` (per-agent optimal), and **`oracle_panel_optimal`** — plus the panel-flip rate *f*.

   > `oracle_full` is **not** a panel upper bound. Rejecting an unhelpful revision restores an
   > agent's original wrong answer, which can consolidate a wrong plurality the revision had
   > split, so the per-agent-optimal policy can lose a question ungated debate won. The true
   > ceiling searches accept subsets per question (≤ 8 combinations with three agents).
   > **G5 gates on `oracle_panel_optimal`.** See `tests/test_oracles.py`.

**Learnability — can a linear probe get there?**

9. Pre-activation correctness probe, grouped 5-fold CV on the pilot's 900 agent-instances, with a layer sweep.
10. Readout-confidence AUROC for pre- and post-correctness. This is the number the probe must beat.

**Engineering**

11. Free-text vs. readout answer disagreement rate; free-text parse-failure rate.
12. Tokens and wall-clock per question; GB of activations per 100 questions → extrapolate.
13. Permutation sensitivity: rerun 50 questions with a second seed, measure the answer-flip rate.

### Go/no-go gates

| # | Criterion | Threshold | If it fails |
|---|---|---|---|
| G1 | Round-0 accuracy | 0.25–0.70 (target 0.35–0.60) | Change model size per the fallback ladder |
| G2 | Non-unanimous round-0 | ≥ 35% of questions | Raise temperature, add persona diversity, or pick a harder subset |
| G3 | Per-agent revision rate | ≥ 10% (target 15–35%) | Scale N; strengthen the peer-influence prompt |
| G4 | Class balance | H ≥ 8% and B ≥ 15% of revision events | Reconsider model/prompt — a probe needs both |
| G5 | **Panel-level oracle headroom over standard MAD** | **≥ 4 points** | Drop to MINIMUM scope |
| G6 | Projected **test-split** revision events at N = 3,000 | ≥ 250 total, with B ≥ 60 and H ≥ 60 | Increase N |
| G7 | Pre-correctness probe AUROC, grouped CV | ≥ 0.60 | Sweep positions/layers first; if still ~0.5, the paper becomes a negative-result probing study |
| G8 | Extrapolated budget | ≤ 15 GPU-h and ≤ 40 GB | Reduce N or the stored layer set |

**G5 and G7 are the two that actually decide the project.**

- G5 fail + G7 pass → pivot to the probe-quality paper (MINIMUM scope).
- G7 fail + G5 pass → the phenomenon is real but not linearly decodable. Reportable, weaker paper.

Also at this gate: write and date `docs/preregistration.md` with the P1/P3 commitment.

---

## 7. Sample size and compute

Let *p* = round-0 accuracy, *r_C* / *r_I* = revision rate given initially correct / incorrect,
*s* = P(a switch from wrong lands on correct). Per *N* questions and 3 agents:

```
H = 3N · p · r_C
B = 3N · (1−p) · r_I · s
U = 3N · (1−p) · r_I · (1−s)
```

| Scenario | p | r_C | r_I | s | B | H | U | Revisions (per 1,000 q) |
|---|---|---|---|---|---|---|---|---|
| Pessimistic | .35 | .10 | .20 | .35 | 136 | 105 | 254 | 495 |
| Central | .42 | .14 | .28 | .45 | 219 | 176 | 268 | 663 |
| Optimistic | .50 | .18 | .35 | .55 | 289 | 270 | 236 | 795 |

- **Probe training: no longer binding.** Correctness is defined on every agent-state, so training uses
  ~18,000 pre+post states (main run) + ~9,000 (control), not ~700 revision events.
- **Probe evaluation (P1):** need ≥ 250 held-out revision events for AUROC SE ≈ 0.03. At N = 3,000
  with a 30% test split: 900 q × 0.50–0.80 revisions/q = 300–480 events. Comfortable even under the
  pessimistic scenario.
- **Power (P3): the binding constraint, and worse than it first looks.** McNemar needs ≈ 85
  discordant test questions for a 65/35 effect and ≈ 194 for 60/40. The tempting shortcut
  `n_d ≈ 0.5 · f · N_test` uses the *oracle* flip rate and so overstates the achievable count
  badly: a validated gate is far more conservative than the oracle. Simulation of the real
  protocol at N = 3,000 gives `n_d ≈ 25` at AUROC 0.70 and `n_d ≈ 52` at AUROC 0.85. **P3 is
  effectively unpowered below AUROC ≈ 0.80 at any N this project can collect.** Plan for it to be
  a bounded null and let the pilot decide otherwise.

**Split 50 / 20 / 30 — train 1,500 q, val 600 q, test 900 q, grouped by underlying question id.**
The larger test split is affordable *precisely because* probe training no longer depends on
revision-event counts (train still holds ~9,000 states). This is what buys marginal P3 power at
N = 3,000.

**N = 3,000** (~25% of MMLU-Pro's 12,032 test questions), finalized at the pilot gate.

| Stage | Work | Estimate |
|---|---|---|
| Generation, main | 18,000 generations, ~11M output + ~25M prefill tokens | 3–4 h |
| Generation, control | Self-reconsideration, same N | 1.5–2 h |
| Activation extraction | 27,000 prefill-only readout passes | 2–3 h |
| Pilot + reruns + slack | | 6–8 h |
| **Total** | | **~15–20 GPU-hours ≈ $40–60 rented** |

Storage ≈ 25 GB. The cost figure should shape risk posture: **do not let unresolved cluster access
block week 3.** Budget ~$100 cloud as a guaranteed fallback and treat cluster access as an
optimization.

---

## 8. Week-by-week

| Wk | Dates | Work | Gate |
|---|---|---|---|
| 1 | Sep 9–15 | **Start cluster applications (Kempner / FASRC / HMS) — lead time is the risk, not cost.** Repo, uv env py3.12, config system with backend switch, provenance module, MMLU-Pro loader, frozen grouped split, permutation machinery, **tier-0 stub + planted-ground-truth generator**. Literature triage. | — |
| 2 | Sep 16–22 | **Analysis layer built and unit-tested against tier-0 ground truth:** labels, panel vote + tie-break, four oracles, probe trainer with GroupKFold, baselines, gating simulator, bootstrap/DeLong/McNemar. **`power_sim.py`** → refine pilot thresholds. Tier-1 (Gemma-3-270M) validates prompts, readout indexing, activation shapes, storage round-trip. | **Hard GPU decision point: if no cluster access, provision ~$100 cloud now** |
| 3 | Sep 23–29 | **Stage 1 pilot** (300 + 150 control) on GPU. All 13 statistics → `docs/pilot_report.md`. Write and date `docs/preregistration.md`. | **G1–G8 · set N · commit metric hierarchy** |
| 4 | Sep 30–Oct 6 | Main generation run at N; self-reconsideration control run; activation extraction. Freeze `runs/main-01/`. | Data collection complete |
| 5 | Oct 7–13 | Probe v1 on real data — the code already exists, so this week is analysis, not building. Layer sweep, grouped CV, L2-logistic + mass-mean + PCA. Baselines: readout confidence, verbalized confidence, metadata-only. | Val AUROC ≥ 0.60 |
| 6 | Oct 14–20 | Offline gating on **validation only**: threshold by simulated panel accuracy, calibration (Platt / ECE / reliability), permutation test, random-gate-at-matched-rate control. | — |
| 7 | Oct 21–27 | **Progress report (~Oct 26).** Freeze interim results. | — |
| 8 | Oct 28–Nov 3 | **Unseal the test set, once.** P1 / P2 / P3 with bootstrap CIs by question. Ablations: pre / post / paired, layer, position. | — |
| 9 | Nov 4–10 | Robustness: LOSO, per-subject, within-question conditional AUROC, metadata-residualized incremental AUROC, 40-item harmful-revision audit. | — |
| 10 | Nov 11–17 | Stretch items **only if on schedule**. Otherwise pure buffer. | — |
| 11 | Nov 18–24 | **All experiments frozen.** Final figures and tables. Paper draft 1. | — |
| 12 | Nov 25–Dec 1 | Thanksgiving; light week. Draft 2, advisor feedback, intro / related work / limitations. | — |
| 13 | Dec 2–8 | Final paper, reproducibility appendix, repo cleanup. Submit at start of Reading Period. | — |

**Critical path: GPU access by week 3, and nothing else.** Weeks 1–2 and everything from week 5
onward is laptop-only.

### Cut order if behind

1. Second model / model-size comparison
2. Adversarial-peer and peer-content-ablation conditions
3. Replicate run with a second permutation seed
4. Isotonic calibration study → keep Platt + ECE only
5. Leave-one-subject-out → keep the random-grouped split only
6. N 3,000 → 1,800 — cheaper than it looks now, since probe training is unaffected and only P3 power is lost
7. Harmful-revision hand audit

**Never cut:** the self-reconsideration control · the metadata-only baseline · self-consistency@3 ·
the permutation test · the representation × target grid · within-question conditional AUROC ·
calibration. These are what make the result defensible; without them the paper is a correlation with
no attribution.

---

## 9. Baselines and controls

**Essential — omitting any of these leaves an obvious hole:**

1. Round-0 plurality (= always-reject endpoint)
2. Standard ungated MAD (= always-accept endpoint)
3. **Self-consistency@3 at matched compute** — three independent samples, majority vote. MAD papers routinely fail to beat this; find out in week 3, not week 11.
4. Token-probability confidence gating (letter-logit softmax) — free from the readout pass
5. Verbalized confidence gating (agent states 0–100)
6. **Metadata-only logistic gating** — peer agreement counts, post = peer majority, unanimity, subject, option count, rationale length, readout confidence. If this matches the probe, the interpretability claim is dead.
7. Random gating at matched acceptance rate
8. Label-permutation test on the probe
9. Panel-level and agent-level oracles
10. **Self-reconsideration control at full N**

**Optional:** self-consistency-derived confidence, full-distribution entropy, single-agent no-debate,
adversarial-peer condition, larger panel sizes.

### Why the self-reconsideration control is not optional

Without it you cannot distinguish *"agents revise because peers influenced them"* from *"agents revise
because a second pass at temperature > 0 produces a different answer."* Those two stories have
completely different implications and a reviewer will ask. The control is identical in every respect
except that the peer block is replaced by a neutral reconsideration instruction with no peer content.
It costs ~2 GPU-hours, and it doubles as extra correctness-probe training data. Run it at full N.

---

## 10. Scope tiers

**MINIMUM — ships regardless, and is a real paper.**

> A characterization of answer revision under one-round MAD in Gemma on MMLU-Pro: revision rates, the
> B/H/U decomposition, the peer-attributable component isolated by the self-reconsideration control,
> the panel-level oracle ceiling, and an evaluation of whether linear probes on internal activations
> predict answer correctness better than verbalized confidence, token-probability confidence, and
> debate metadata. Downstream gating reported with a confidence interval, positive or null, alongside
> a headroom analysis explaining the gap between achieved AUROC and the oracle.

A well-characterized null on gating *with* a demonstrated AUROC advantage for the probe is a
defensible 20-page paper. A null on both is still defensible if the oracle ceiling is high: it says
the information exists but is not linearly decodable, which is a genuine finding about representations.

**CORE (target).** Everything above, plus the full representation × target grid, layer sweep,
calibration, permutation and random-gate controls, LOSO, within-question conditional AUROC, and the
metadata-residualized incremental analysis.

**STRETCH (week 10 only, in this order).**
1. Second Gemma size — does the probe transfer across scale?
2. Adversarial-peer condition (one agent prompted to argue a wrong answer confidently) — the cleanest test of whether the probe detects *bad* influence specifically.
3. Peer-content ablation (answer letters only, no rationales) — separates label conformity from argument persuasion.
4. Replicate run under a second permutation seed.

---

## 11. Verification

### Tier 0 — week 2, all must pass before any GPU time is spent

- Planted-ground-truth run: the labeler recovers the exact planted B/H/U counts.
- Panel vote + tie-break reproduce hand-computed outcomes on a fixture covering all 3-0 / 2-1 / 1-1-1 cases.
- All four oracles match closed-form values on the planted distribution.
- Oracle-gated panel accuracy ≥ ungated MAD, asserted by construction.
- A probe with planted signal strength recovers the expected AUROC within Monte Carlo error; the same probe on shuffled labels gives AUROC ≈ 0.5 with a CI covering 0.5.
- McNemar / DeLong / bootstrap reproduce known values on synthetic input.
- Assertion: no underlying question id appears in more than one split.
- Storage round-trip: a reloaded fp16 shard matches the in-memory tensor bitwise.
- Backend conformance suite passes for `stub` and `hf`-tiny (and later `vllm`).

### Tier 1 — week 2

20-question smoke run on Gemma-3-270M: transcripts parse, both phases extracted, index row count
equals `n_q × 3 agents × 2 phases`, the readout token index lands on the intended token, letter logits
present, manifest and completion both written.

### Tier 2 — week 3

The pilot report regenerates end-to-end from `runs/pilot-01/` with a single command.

---

## 12. Open items

**Defaults unless corrected:** all 14 MMLU-Pro subjects · public GitHub for analysis artifacts but not
raw activations · temperature 0.7 with distinct seeds and light persona diversity · Python 3.12 via
`uv` · readout answer canonical over free-text.

**To ask the advisor this week:** does Prof. Wang expect a positive end-to-end result, or is a
well-characterized null acceptable? That answer determines whether budget goes toward N (chasing P3
power) or toward controls and ablations (strengthening P1), and it feeds the week-3 preregistration.
