# Related Work and Positioning

**Compiled 2026-09-10.** Status of every claim is marked. Only papers under "Verified" were
fetched and confirmed directly; the rest come from an automated scan and **must be checked before
citation**. Six of eight scan agents died to a session rate limit, so coverage is uneven — the gaps
are listed at the end.

---

## Verified directly (abstract fetched and confirmed)

### 1. Hao et al. (2026), *Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate* — arXiv:2606.00820, 30 May 2026

Xiqi Hao, Zengqing Wu, Yu-Xuan Qiu, Chuan Xiao, Ruiqi Xu, Shuyuan Zheng, Jianbin Qin.

Decomposes stance change into **spontaneous instability, stance-induced conformity, and
reasoning-induced persuasion**. Primary setting **MMLU-Pro**. Confirmed verbatim:

- "37% of agent-question observations change under self-reflection alone"
- "Harmful conformity can be predicted from Round 0 features (AUC = 0.79)"
- "risk-targeted intervention reduces it by 13.6 percentage points (p < 0.001)"

**Impact on this project: severe, and it lands on the fallback.** The MINIMUM scope in `PLAN.md`
— characterise revisions on MMLU-Pro, isolate the peer-attributable component with a
self-reconsideration control, predict harmful revisions — is essentially this paper. It is no
longer a novel contribution; it is a replication target.

**What it leaves open:** their predictor uses **round-0 behavioural features**, not internal
activations. So the internal-representation thesis survives intact, and now has a concrete,
quotable bar rather than a vague one:

> **beat AUC 0.79 for predicting harmful revision, and beat 13.6pp of harmful-conformity reduction.**

Note how well this matches the simulation in `PLAN.md` §3.4: revision gating needs roughly AUROC
≥ 0.80 before it moves panel accuracy. Their behavioural baseline sits exactly at that threshold,
which makes the comparison unusually clean — a probe that cannot beat 0.79 genuinely has no case.

### 2. Hu & Qu (2026), *Most LLM Conformity Needs No Speaker: Measuring the Speaker-Free Floor in Peer-Pressure Benchmarks* — arXiv:2607.05545, 6 Jul 2026

Six open-weight LLMs, seven QA/reasoning datasets. Confirmed verbatim:

> "this condition alone causes harmful revision in 66.5% of initially correct cases, compared with
> 10.3% under a plain re-ask."

The "condition" is a **no-source** presentation: the contradicting answer with all speaker
attribution stripped. Roughly 6.5× the plain re-ask floor with no social cue at all.

**Impact: the control design in `PLAN.md` §9 is insufficient.** A single self-reconsideration
control cannot separate peer influence from the effect of simply being shown a contradicting
answer again. Required control ladder:

| condition | round-1 input | isolates |
|---|---|---|
| plain re-ask | nothing new | spontaneous instability floor |
| length-matched filler | neutral text of matched length | context-length effects |
| no-source assertion | peer answers and rationales, attribution stripped | repeated-assertion effect |
| full debate | peer answers and rationales, attributed | social/peer effect |

Cost: two extra conditions, roughly +4–6 GPU-hours at N = 3,000. Cheap, and without them the
peer-attribution claim is contestable.

### 3. Zhu, Zhang, Chi, Stafford, Collier & Vlachos (2026), *Demystifying Multi-Agent Debate: The Role of Confidence and Diversity* — arXiv:2601.19921, Jan 2026 (rev. Jun 2026)

Confirmed verbatim from the abstract:

- "recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost"
- "under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes"
- proposes "a confidence-modulated debate protocol in which agents **express** calibrated confidence and condition their updates on others' confidence"
- "across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote"

**Impact: two-sided.**

*Supporting.* Their theoretical result is the analytic counterpart of the regime analysis in
`PLAN.md` §3.4 — homogeneous agents with uniform updates cannot reliably gain. This is now a
citable justification for the design rather than a simulation artefact of ours, and it predicts
the favourable regime for a gate.

*Competing.* Confidence-modulated debate is the verbalized-confidence version of the
weighted-aggregation idea. **The internal-signal version remains open**, and their protocol is now
the baseline to beat — which is a good thing, because verbalized confidence is the known-weak
signal (Xiong et al. 2023).

---

## Unverified (automated scan; check before citing)

Reported by the scan but **not** independently confirmed. Treat as leads.

**Debate underperforms for small open models.** Smit et al., *Should we be going MAD?* (ICML 2024,
arXiv:2311.17371) — MAD does not reliably beat self-consistency/ensembling. Bertalanič & Fortuna
(arXiv:2605.00914) — debate costs 2.1–3.4× the tokens for equal or lower accuracy on Qwen2.5-7B /
Llama-3.1-8B / Ministral-3-8B, with a plurality-voting **oracle gap up to 32.3pp**. Wynn, Satija &
Hadfield (arXiv:2509.05396, ICML MAS workshop 2025) — accuracy *decreases* over debate rounds.

**Conformity magnitude, Gemma specifically.** BenchForm (Weng et al., ICLR 2025 Oral,
arXiv:2501.13381) — Gemma2-27B conformity 39.7% under wrong guidance, 66.1% under a "Doubt"
protocol. Hu & Qu (2607.05545) — Gemma-2-9B harmful revision 42.0% no-source / 63.0% expert
framing, the most resistant of six small open models. Hussain & Nielbo (arXiv:2608.11247) —
gemma-3-27b-it 8.7% MMLU / 42.0% GPQA reversal, among the most resistant of 23 models; conformity
is **uncorrelated with parameter count** but strongly negatively correlated with baseline task
accuracy (r ≈ −0.64 on MMLU).

**Asymmetry.** Qu, Fu & Hu, *Easier to Mislead Than to Correct* (arXiv:2606.01637) — peer agreement
misleads correct models more readily than it corrects wrong ones; CoT and reflection do not
reliably reduce harmful revision while preserving beneficial revision.

**Internal signals, diagnosed but not deployed.** Kumarappan & Mujoo, *Not Just RLHF*
(arXiv:2605.12991) — activation patching localises multi-agent sycophancy to a narrow mid-layer
window; patching above it restores 96% of the clean-to-pressured P(correct) gap. Diagnostic only:
they never convert the direction into a deployed intervention inside a live debate.

**Reported as unoccupied by the scan** (absence of evidence only — the scan was incomplete):
internal-signal-gated revision; per-item agent-level conviction gates; steering deployed inside
debate with accuracy and flip-rate results; prescriptive bounded-confidence thresholds; a unified
treatment of the degeneration-of-thought ↔ sycophancy trade-off.

---

## Revised positioning

The behavioural characterisation is taken. The internal-representation claim is not. Positioning
follows from that:

> **Do internal model states carry information about answer-revision quality that behavioural
> features and verbalized confidence do not — and is it enough to change how a multi-agent panel
> should aggregate?**

A 3 × 2 design over the same collected data, with published baselines for two of three signals:

| control signal | vote weighting | revision gating |
|---|---|---|
| verbalized confidence | Zhu et al. 2601.19921 | Lin & Hooi 2509.14034 |
| behavioural / round-0 features + token logits | — | Hao et al. 2606.00820 (AUC 0.79) |
| **internal activations (this work)** | **open** | **open** |

Both interventions run offline on one collection. Per `scripts/design_explorations.py` they are
powered in **complementary regimes** — weighted voting when debate helps, revision gating when it
hurts — so the design has two shots regardless of which regime Gemma turns out to be in.

### Consequences for the plan

1. **MINIMUM scope must change.** "Characterise revisions and predict them" is now a replication.
   The new floor is the *signal comparison*: does an internal probe beat behavioural features and
   verbalized confidence at predicting revision quality? That is still defensible as a null.
2. **Add the speaker-free control ladder** (§2 above). Non-optional.
3. **Model choice is now evidence-based, and cuts against Gemma-3-12B.** Gemma-3-27B is reported
   among the *most* conformity-resistant models tested (8.7% MMLU reversal), which threatens gates
   G3/G4 on revision-event yield. Gemma-2-9B shows far higher harmful-revision rates (42%) and
   would supply more labelled events. Decide at the week-3 pilot on measured revision rate, not on
   the a-priori preference for Gemma 3.
4. **Concrete numeric targets** replace vague ones: beat AUC 0.79, beat 13.6pp.

---

## Gaps — scan did not complete

Six of eight agents were killed by a session rate limit. Not covered:

- **Achievable correctness-probe AUROC** on hard multiple-choice reasoning, and the incremental
  value over token-probability confidence. **This is the single most important open number** — it
  determines whether beating AUC 0.79 is plausible at all.
- Published MMLU-Pro accuracy for Gemma-2-9B / Gemma-3-12B (needed for gate G1).
- Efficiency / accuracy-per-compute framings and selective invocation.
- Heterogeneous vs homogeneous agents.
- Calibration and selective-prediction endpoints.
- MMLU-Pro label-noise rate.
