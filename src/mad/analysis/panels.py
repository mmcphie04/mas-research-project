"""Main-collection analysis: more independent answers vs more discussion (docs/NEXT_PHASE_PLAN.md,
2026-10-29 evening). Every panel score is the exact expected accuracy of a plurality vote over
the panel's LATEST answers with uniformly random tie-breaking (``actions.expected_vote``).

Inputs per question: the pool of independent first answers in agent order (agents 0..9; the
3-agent panel is agents 0-2, the 5-agent panel agents 0-4, nested and prespecified), the gold
answer, and, per debate condition, each panel agent's revised answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mad.analysis.actions import expected_vote

POOL_SIZES = tuple(range(3, 11))
MATCHED = {3: 6, 5: 10}  # debate with k agents generates 2k responses; independent baseline of 2k


def independent_curve(pool: dict[str, list[int]], gold: dict[str, int], sizes=POOL_SIZES) -> pd.DataFrame:
    """Per question and pool size k: expected accuracy of voting the first k independent answers."""
    rows = []
    for q, ans in pool.items():
        for k in sizes:
            if len(ans) >= k:
                rows.append({"question_id": q, "k": k, "acc": expected_vote(ans[:k], gold[q])})
    return pd.DataFrame(rows)


def panel_events(initial: list[int], revised: list[int], gold: int) -> dict:
    """What a debate round did to one panel: per-agent changes, and the panel decision before and
    after (fractional), with the categories the plan names."""
    before, after = expected_vote(initial, gold), expected_vote(revised, gold)
    present = gold in initial
    agents = [(a, b) for a, b in zip(initial, revised, strict=True)]
    return {
        "acc_before": before, "acc_after": after, "net": after - before,
        "helpful": sum(1 for a, b in agents if a != b and b == gold),
        "harmful": sum(1 for a, b in agents if a != b and a == gold),
        "changed": sum(1 for a, b in agents if a != b),
        # a correct answer held by a minority (present, but not the decided plurality) that the
        # round turned into a (more) winning answer
        "minority_correct_recovered": float(present and before < 1 and after > before),
        "correct_decision_preserved": float(before == 1 and after == 1),
        "correct_decision_lost": float(before == 1 and after < 1),
        "absent_correct_found": float((not present) and gold in revised),
        "gold_present_initially": float(present),
    }


def debate_table(initial: dict[str, list[int]], revised: dict[str, list[int]], gold: dict[str, int],
                 condition: str, k: int) -> pd.DataFrame:
    rows = []
    for q, rev in revised.items():
        rows.append({"question_id": q, "condition": condition, "k": k,
                     **panel_events(initial[q][:k], rev, gold[q])})
    return pd.DataFrame(rows)


def paired(a: pd.Series, b: pd.Series, n_boot: int = 10000, seed: int = 0) -> dict:
    """Mean per-question difference a - b over shared questions, with a question bootstrap."""
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).to_numpy(float)
    if len(d) == 0:
        return {"questions": 0}
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"questions": int(len(d)), "mean_diff": float(d.mean()), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "questions_differing": int((d != 0).sum()),
            "expected_gain": float(d[d > 0].sum()), "expected_loss": float(-d[d < 0].sum())}


def compare(pool: dict, gold: dict, debates: dict[str, tuple[int, dict]], n_boot: int = 10000) -> dict:
    """The headline comparisons. ``debates``: condition -> (panel size k, {qid: revised answers})."""
    ind = independent_curve(pool, gold).pivot(index="question_id", columns="k", values="acc")
    out = {"independent_accuracy": {int(k): float(ind[k].mean()) for k in ind.columns}, "debate": {}}
    for cond, (k, rev) in debates.items():
        t = debate_table(pool, rev, gold, cond, k).set_index("question_id")
        out["debate"][cond] = {
            "k": k, "acc_after": float(t.acc_after.mean()), "acc_before": float(t.acc_before.mean()),
            "vs_initial_panel": paired(t.acc_after, t.acc_before, n_boot),
            f"vs_independent_{MATCHED[k]}": paired(t.acc_after, ind[MATCHED[k]], n_boot),
            f"vs_independent_{k}": paired(t.acc_after, ind[k], n_boot),
            "helpful": int(t.helpful.sum()), "harmful": int(t.harmful.sum()), "changed": int(t.changed.sum()),
            **{c: float(t[c].sum()) for c in ("minority_correct_recovered", "correct_decision_preserved",
                                              "correct_decision_lost", "absent_correct_found")},
        }
    return out


def cost_table(completions: list[dict]) -> pd.DataFrame:
    """Measured cost per condition from chunk completion records: generation (inference) kept
    apart from the readout passes that extract activations and answer probabilities."""
    rows = []
    for c in completions:
        t = c.get("timings", {})
        rows.append({"condition": c.get("condition"), "questions": c.get("row_counts", {}).get("questions"),
                     "generation_s": t.get("generation_s"), "prompt_tokens": t.get("prompt_tokens"),
                     "generated_tokens": t.get("generated_tokens"), "capped_responses": t.get("capped_responses"),
                     "extraction_s": (t.get("readout_s") or 0) + (t.get("exposure_readout_s") or 0),
                     "write_s": t.get("write_s")})
    return pd.DataFrame(rows).groupby("condition", dropna=False).sum(numeric_only=True)
