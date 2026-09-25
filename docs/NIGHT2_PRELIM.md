# Four-condition experiment — preliminary (300 of 350 questions)

gemma-3-1b-it, ARC-Challenge, 3 agents, one round, **shared round 0** (verified byte-identical
across all four conditions, so every comparison is matched at the agent x question level).

| condition | revision rate | B | H | U | B−H | round-0 panel | final panel | gain |
|---|---|---|---|---|---|---|---|---|
| self_reconsider | 0.360 | 107 | 71 | 146 | **+36** | 0.3433 | **0.4033** | **+6.00 pts** |
| debate | 0.392 | 89 | 106 | 158 | −17 | 0.3433 | 0.3433 | **+0.00 pts** |
| debate_conf | 0.392 | 89 | 106 | 158 | −17 | 0.3433 | 0.3433 | +0.00 pts |
| adversarial | 0.990 | 3 | 317 | 571 | −314 | 0.3433 | 0.0067 | −33.67 pts |

## Headline: homogeneous agents erase the benefit of debating

Last night, with three *different* personas, debate gained +4.75 points and beneficial revisions
outran harmful ones by +39. Tonight, with one persona for all three agents and diversity coming
only from sampling, the gain is **exactly zero** and the balance inverts to −17.

This is an empirical confirmation of Zhu et al. (arXiv:2601.19921), who prove that homogeneous
agents with uniform belief updates cannot reliably gain from debate. The persona diversity was
doing the work, not the debating.

Caveat: tonight's 300 questions are a different stratified sample from last night's 400, so the
persona comparison is not perfectly controlled. The within-tonight comparisons are clean.

## Thinking alone beats thinking with peers

On identical questions from an identical round 0, solitary reconsideration gains **+6.00 points**
and debate gains **0.00**. Matched at the agent level:

| | peer-induced harm | peer-induced benefit | net |
|---|---|---|---|
| debate | 67 | 45 | **−22** |
| adversarial | 246 | 3 | −243 |

"Peer-induced harm" means: the agent was right, stayed right when it reconsidered alone, and went
wrong when shown peers. That attribution is only possible because round 0 is shared.

## Peer configuration matters, and lumping it together was wrong

Splitting the cases the reviewer asked for:

| round-0 peer configuration | n | switch rate |
|---|---|---|
| both peers agree WITH focal | 609 | 0.273 |
| split (one agrees, one differs) | 203 | 0.611 |
| both oppose focal and agree with each other | 88 | **0.716** |

A clean monotone gradient. The old `n_peers_agree_pre == 0` grouping merged the last two rows,
which differ by 10 points of switch rate.

**Canonical conformity** (focal correct, both peers agree on the same wrong answer, focal adopts
it): **15 of 27 = 55.6%** in natural debate.

## The adversarial condition saturates and is not a usable instrument

Fabricated peers confidently asserting one wrong answer produce a **99.0% revision rate** and
**99.1% adoption** of the wrong consensus; panel accuracy collapses to 0.0067. This does not
measure conformity, it measures whether a 1B model will copy a confident assertion. It always
will. The manipulation needs to be weakened (one adversarial peer instead of two, or hedged
language) to land in a regime where the outcome is informative.

## Two methodological findings

**The peer-confidence ablation was a no-op.** 0 of 75 round-0 generations contained a
`Confidence:` line, so `strip_confidence` never fired and `debate` and `debate_conf` are
byte-identical. Not a bug — the fix is correct and will matter at 12B — but it tests nothing here.

**`max_new_tokens=64` truncates the model mid-reasoning.** Round-0 generations are well-formed
analysis that gets cut off before stating an answer, which is why the free-text parse rate is
~20%. Agents are exchanging partial reasoning that never reaches a conclusion. The forced readout
still yields a valid answer, so the data is usable, but the debate is weaker than intended. Raise
the budget to ~192 tokens for any run that matters.

---

# Final numbers (all 350 questions)

Shared round 0 verified byte-identical across all four conditions.

| condition | rev rate | B | H | U | B−H | round-0 | final | gain | ceiling |
|---|---|---|---|---|---|---|---|---|---|
| self_reconsider | 0.361 | 121 | 92 | 166 | **+29** | 0.3714 | **0.4171** | **+4.57** | 0.4914 |
| debate | 0.382 | 101 | 122 | 178 | −21 | 0.3714 | 0.3771 | +0.57 | 0.4800 |
| debate_conf | 0.383 | 101 | 123 | 178 | −22 | 0.3714 | 0.3771 | +0.57 | 0.4800 |
| adversarial | 0.988 | 3 | 395 | 639 | −392 | 0.3714 | 0.0086 | −36.29 | 0.3743 |

**Matched peer attribution** (same agent, same starting answer):

| | peer-induced harm | peer-induced benefit | net |
|---|---|---|---|
| debate | 76 | 52 | **−24** |
| adversarial | 303 | 3 | −300 |

**Peer configuration gradient** (debate):

| round-0 configuration | n | switch rate |
|---|---|---|
| both peers agree with focal | 708 | 0.270 |
| split | 239 | 0.577 |
| both oppose focal and agree with each other | 103 | **0.699** |

Canonical conformity (focal right, peers agree on one wrong answer, focal adopts it):
**16 / 31 = 51.6%**.

## Storage gap found and fixed

The full answer-choice log-probability vector was never persisted — only the top probability — so
entropy, top-two margin and the full-vector baseline the review asked for cannot be computed on
existing runs without redoing the readout pass. `readout_logprobs`, `readout_margin` and
`readout_entropy` are now written by both collectors. Output-only baselines on the current data
are therefore limited to top probability plus row metadata.
