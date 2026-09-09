# Risk Register

**Project:** Using Internal Representations to Evaluate Answer Revisions in Multi-Agent Debate
**Written:** 2026-09-09 · **Companion:** [`PLAN.md`](PLAN.md)

Each risk records **how it is detected early**, **how serious it is**, and **the smallest design change
or fallback that addresses it**. Risks are ordered by how likely they are to produce a result that
looks good and does not replicate.

---

## Part I — Probe shortcuts

These are the ways the probe could score well while learning nothing about revision quality. They are
the highest-priority risks because each one produces a *positive-looking* result.

### 1. Difficulty detector — HIGH

**What happens.** The probe learns "this question is easy," not anything about this agent's revision.
Question difficulty correlates strongly with correctness and is plainly visible in activations. Since
the label is plain correctness (see PLAN.md §3.2), this is the natural degenerate solution, and
reframing the label made it *more* acute, not less.

**Detection.**
- **Within-question conditional AUROC.** Restrict to questions where the three agents' revisions carry
  mixed labels, and recompute. Collapse toward 0.5 is diagnostic.
- Within-question variance of probe scores. Near-zero variance means the probe is only reading the
  question.

**Smallest fix.** Report conditional AUROC as a **headline number alongside marginal AUROC**, not as a
robustness footnote. If conditional AUROC is at chance, the honest claim is "activations encode item
difficulty," which is a much weaker but still reportable finding.

---

### 2. Peer-majority readout — HIGH

**What happens.** The probe learns "my new answer matches the peer majority." This is the single
strongest signal in the data and it is fully visible in the post-state context. It is a legitimate
predictor but not an *internal-representation* claim — a five-feature logistic regression on debate
metadata would get it for free.

**Detection.** Fit a metadata-only logistic baseline on: peer agreement counts, whether post = peer
majority, peer unanimity, subject, option count, rationale length, readout confidence pre and post.
If it matches the probe's AUROC, the interpretability claim is gone.

**Smallest fix.** Make the metadata baseline mandatory, and report **incremental AUROC over metadata**
— paired DeLong on the same held-out events, plus a likelihood-ratio test in a combined
metadata + probe model. Incremental value over metadata and verbalized confidence is the paper's core
claim; state it that way.

---

### 3. Threshold and hyperparameter overfitting — HIGH

**What happens.** Roughly 40 configurations (layers × positions × regularization strength × feature
sets) get selected on validation. The validation number is then optimistically biased, and reporting
it as the result is a silent error.

**Detection.** Report the validation→test AUROC gap. A large gap means selection overfitting.

**Smallest fix.** Select once on validation, unseal the test set once in week 8, report test only.
Use nested CV for any validation number that appears in the paper.

---

### 4. Test-set re-analysis — HIGH

**What happens.** Because the gating stage requires zero new inference (PLAN.md §2), re-running the
whole analysis on the test set costs seconds. That is exactly the condition under which a garden of
forking paths grows.

**Detection.** Log every test-split read to an append-only file with a timestamp and a reason.

**Smallest fix.** Test split sealed until week 8. Every read after the first is a **documented
amendment in the paper**, listed in the appendix. This is cheap to do and expensive to retrofit.

---

### 5. Question leakage across splits — HIGH if missed, easy to prevent

**What happens.** The same underlying question appears three times per run (one per agent), again
under any permutation replicate, and again in the self-reconsideration control. If any of those land
on opposite sides of a split, the probe has seen the test questions.

**Detection.** A unit test asserting no underlying question id appears in more than one split.

**Smallest fix.** `GroupKFold` on the *underlying question id* — not the row id, not the
(question, permutation) pair. Enforce by assertion, not convention.

---

### 6. Subject leakage — MEDIUM

**What happens.** The probe learns "this is a math question, math is hard" and generalizes within but
not across domains.

**Detection.** Leave-one-subject-out evaluation.

**Smallest fix.** Report LOSO alongside the random-grouped split. A large gap is itself a reportable
finding about probe generality.

---

### 7. Answer-identity and position bias — MEDIUM

**What happens.** MMLU-Pro option position correlates with correctness if permutation coverage is
imperfect, and the probe reads off "the answer is option C."

**Detection.** Train a probe to predict the option letter from activations (it will succeed — that is
expected). Then cross-tabulate letter × correctness in the collected sample. A dependence there is the
actual problem.

**Smallest fix.** Verify permutation coverage is balanced; report per-position accuracy. If unbalanced,
reweight or add a replicate run under a second permutation seed.

---

### 8. Context-length confound — MEDIUM

**What happens.** Post-state contexts are far longer than pre-state contexts (they contain three peer
rationales). Any pre/post difference vector partly encodes "how much text is in my context."

**Detection.** Correlate probe score with context token count.

**Smallest fix.** The forced-format readout pass (PLAN.md §3.3) puts both states at an identical
template position, which largely removes it. Report the correlation anyway.

---

## Part II — Design and measurement risks

### 9. Debate fails to beat self-consistency@3 — HIGH for framing, not fatal

**What happens.** Three agents debating for one round does no better than three independent samples
majority-voted at the same compute. Multi-agent debate papers routinely have this problem.

**Detection.** Pilot statistic #3, week 3.

**Smallest fix.** Reframe rather than abandon. The paper is about revision quality, and "one-round MAD
underperforms matched-compute self-consistency in Gemma on MMLU-Pro" is itself a contribution worth
reporting cleanly. Do not discover this in week 11.

---

### 10. Panel-level oracle headroom below 4 points — HIGH

**What happens.** Agent-level net revision gain looks healthy but perfect gating buys almost nothing at
the panel level, because majority voting absorbs individual errors. No learned gate can then show
anything at achievable N.

**Detection.** Pilot gate G5.

**Smallest fix.** Make agent-level results (P2) primary and report panel accuracy as a documented
negative with a headroom analysis. This is the MINIMUM scope in PLAN.md §10 and it still yields a paper.

---

### 11. Revision rate too low to yield labels — HIGH

**What happens.** Agents rarely change their answers, so there are too few revision events to evaluate
on, whatever N is.

**Detection.** Pilot gate G3 (< 10% per-agent revision rate).

**Smallest fix.** In order of preference: strengthen the peer-influence prompt; raise temperature or
add persona diversity to increase initial disagreement; drop to Gemma-2-9B-it (lower accuracy → more
revisions); scale N.

---

### 12. MMLU-Pro label noise inflates the harmful class — MEDIUM

**What happens.** A few percent of MMLU-Pro gold labels are wrong, concentrated in some subjects. A
revision away from a mislabeled "correct" answer is scored harmful when it was actually right. This
caps achievable AUROC and biases the B/H/U decomposition.

**Detection.** Hand-audit ~40 sampled harmful revisions.

**Smallest fix.** Report the audited noise rate as an explicit **ceiling on achievable AUROC**. Cheap,
and it makes the limitations section substantially stronger.

---

### 13. Three-way ties dominate panel outcomes — MEDIUM

**What happens.** Three agents over ten options produce 1-1-1 splits often enough that the tie-break
rule, rather than the gate, drives panel accuracy.

**Detection.** Pilot statistic #2.

**Smallest fix.** Pre-registered tie-break (round-0 plurality, then highest readout confidence) and
report tie frequency. If ties exceed ~20% of questions, report panel accuracy under two tie-break rules
to show the result is not an artifact of the choice.

---

### 14. Homogeneous agents measure self-conformity, not peer conformity — LOW severity, must be named

**What happens.** All three agents share weights and differ only in sampling seed and persona. "Peer
influence" is therefore influence from samples of the same model, which is not the same construct as
influence between different models.

**Detection.** Inherent to the design; not detectable from the data.

**Smallest fix.** The self-reconsideration control quantifies exactly how much of the revision behavior
is peer-attributable. Name the limitation explicitly and frame a heterogeneous-agent study as the
natural follow-up.

---

### 15. Benchmark contamination — LOW

**What happens.** MMLU-Pro items may appear in Gemma's pretraining data, inflating round-0 accuracy and
altering revision behavior.

**Detection.** Not directly measurable within this project's scope.

**Smallest fix.** One paragraph in limitations.

---

### 15b. A fixed acceptance threshold makes the intervention worse than no intervention — HIGH

**What happens.** Beneficial revisions outnumber harmful ones, so *accepting everything* is already
net-positive and standard MAD is a strong baseline. A gate thresholded at a fixed 0.5 rejects
beneficial revisions faster than it catches harmful ones: simulation at AUROC 0.68 with a 0.5 cut
loses ≈ 0.8 panel points relative to ungated debate.

**Detection.** Simulation, before any real data — `mad power --by-auroc ...`. Also visible in the
validation threshold sweep: if the selected τ sits at the edge of the grid, the gate is degenerate.

**Severity.** High, and easy to mistake for "the probe doesn't work" when the real fault is the
decision rule.

**Smallest fix.** τ is always selected on validation against simulated panel accuracy, never fixed
a priori and never tuned on an F1 or Youden criterion whose cost matrix does not match this problem
(accepting an unhelpful revision is free; accepting a harmful one costs a correct answer). Report
the full τ sweep as a figure so a degenerate gate is visible rather than hidden.

---

## Part III — Engineering and reproducibility risks

### 16. Activation storage explosion — HIGH if unmanaged, fully controlled by one rule

**Quantified.** Per state = (n_layers + 1) × d_model × 2 bytes ≈ 376 KB for a 12B model.

| What you store | Size at N = 3,000 (18,000 states) |
|---|---|
| All layers, 1 fixed position | ≈ 6.8 GB |
| All layers, 2 fixed positions | ≈ 13.5 GB |
| One layer, all ~800 token positions | ≈ 1.8 TB |

**Smallest fix.** **Fixed token positions only.** Layer sweeps are affordable; token sweeps are not.
Store all layers at 1–3 canonical positions in fp16, sharded, with a Parquet index. Read one layer at
a time (≈ 138 MB) so peak RAM stays trivial on a 13 GB machine.

---

### 17. Tier-0 stub diverges from the real backends — MEDIUM

**What happens.** The stub passes tests that the real generation path would fail, giving false
confidence before GPU time is spent.

**Detection.** A backend conformance test suite that all three tiers must pass identically.

**Smallest fix.** One shared backend interface; the conformance suite is part of the week-2 exit
criteria, not an afterthought.

---

### 18. vLLM sampling is not bitwise reproducible across batch sizes — MEDIUM

**What happens.** Re-running a config does not reproduce the transcripts, so "reproducible" claims based
on re-execution are false.

**Detection.** Obvious on any rerun.

**Smallest fix.** **Store all generated text.** Provenance is artifacts, not re-execution. The
reproducibility claim is "these transcripts produced these results," verified by content hash.

---

### 19. Answer parsing failures conflated with wrong answers — MEDIUM

**What happens.** A free-text response that does not match the expected answer format gets scored
incorrect, silently mixing a pipeline failure into the accuracy estimate.

**Detection.** Track parse outcome as a separate field on every row, never as a wrong answer.

**Smallest fix.** The forced-format readout pass (PLAN.md §3.3) makes a parse failure impossible —
argmax over the ten letter tokens always yields an answer. Free-text parse failure becomes a *measured
disagreement rate* between the free-text and readout answers, reported as a pipeline health statistic.
The readout answer is canonical.

---

### 20. Unrecorded permutation — MEDIUM

**What happens.** Answer-choice order is randomized but not recorded, so answers cannot be mapped back
to canonical labels and the run is unusable.

**Smallest fix.** One shared permutation per question per run (so peer messages' letter references stay
coherent without translation), with the seed and the full index mapping written into the run manifest
and carried on every row.

---

## Part IV — Provenance specification

Provenance must exist from the first real experiment, not be retrofitted. Write `manifest.json`
**before** the run starts and `completion.json` **after** it finishes, so a crashed run is
distinguishable from a completed one.

`manifest.json` contains:

```
run_id, timestamp_start
git_sha, git_dirty_flag
config (fully resolved), config_hash
model_repo_id, model_revision (commit hash), tokenizer_hash, dtype
dataset_name, dataset_revision, split, question_id_list_hash
prompt_template_hashes (round-0, round-1 debate, round-1 self-reconsider, readout)
permutation_seed, per_agent_sampling_seeds
temperature, top_p, max_tokens
uv_lock_hash, pip_freeze
torch/transformers/vllm versions
gpu_type, gpu_count
```

`completion.json` adds: `timestamp_end`, per-shard `sha256`, row counts, and any non-fatal errors.

Two rules that make this useful rather than decorative:

- **Every output row carries `run_id`.**
- **Every figure and table in the paper cites the `run_id` it came from.**

---

## Part V — Constraints identified before scoping, and their status

| Constraint | Status | Where handled |
|---|---|---|
| Activation storage can explode | **Validated and quantified.** 1.8 TB if token positions are swept; 6.8 GB under the fixed-position rule | Risk 16 |
| Splits must be grouped by underlying question | **Validated.** Must also group across agents, permutation replicates, and the control condition — not just across rows | Risk 5 |
| Parse failures tracked separately from wrong answers | **Validated, and largely designed away** by the forced-format readout pass | Risk 19 |
| Randomized answer permutation must be recorded | **Validated.** Also: use one shared permutation per question per run, or peer messages need letter translation | Risk 20 |
| Run provenance from the first real experiment | **Validated and specified.** Note the before/after split so crashed runs are identifiable | Part IV |

### Added during scoping

- Panel-level vs. agent-level metric mismatch (Risk 10) — the largest missed issue.
- No non-degenerate "revision quality" label exists (PLAN.md §3.2) — reframes the study.
- P3 is underpowered at achievable N (PLAN.md §3.4) — reframes the paper's primary claim.
- Self-consistency@3 as an essential matched-compute baseline (Risk 9).
- Metadata-only baseline as the claim-killer if omitted (Risk 2).
- Test-set re-analysis is unusually tempting here because gating is free (Risk 4).
- MMLU-Pro label noise as an AUROC ceiling (Risk 12).
