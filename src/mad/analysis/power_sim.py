"""Synthetic power analysis -- zero compute, run before the pilot.

Plants a B/H/U distribution at assumed rates, simulates a probe of a given AUROC, and reads off
the panel-accuracy gain and its McNemar power as a function of N. This is what turns the
sample-size arithmetic in docs/PLAN.md section 7 from an estimate into a curve, and it produces
the events frame in exactly the schema the real pipeline emits, so the analysis code is exercised
end-to-end on data whose answer is known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from mad.analysis.labels import BENEFICIAL, HARMFUL, NONE, UNHELPFUL
from mad.analysis.oracles import panel_flip_rate, policy_accept
from mad.analysis.stats import mcnemar


def simulate_events(
    n_questions: int,
    *,
    p: float = 0.42,
    r_C: float = 0.14,
    r_I: float = 0.28,
    s: float = 0.45,
    n_agents: int = 3,
    n_options: int = 10,
    peer_influence: float = 0.0,
    seed: int = 0,
    condition: str = "debate",
) -> pd.DataFrame:
    """Generate an events frame with the planted revision distribution.

    ``peer_influence`` in [0, 1] blends between a peer-independent revision rate (0) and one
    proportional to the fraction of peers who disagree with you (1), normalised so the marginal
    revision rate stays at r_C / r_I either way. It matters for any question about *selective*
    debate: if revisions only happen when peers disagree, then skipping unanimous panels is free.
    """
    rng = np.random.default_rng(seed)
    gold = rng.integers(n_options, size=n_questions)

    # round 0
    pre = np.empty((n_questions, n_agents), dtype=int)
    pre_correct = rng.random((n_questions, n_agents)) < p
    for q in range(n_questions):
        for a in range(n_agents):
            pre[q, a] = (
                gold[q]
                if pre_correct[q, a]
                else rng.choice([i for i in range(n_options) if i != gold[q]])
            )

    # fraction of peers disagreeing with each agent, and its mean (for rate normalisation)
    disagree = np.empty((n_questions, n_agents))
    for q in range(n_questions):
        for a in range(n_agents):
            peers = [pre[q, b] for b in range(n_agents) if b != a]
            disagree[q, a] = np.mean([x != pre[q, a] for x in peers]) if peers else 0.0
    mean_disagree = disagree.mean() if disagree.mean() > 0 else 1.0

    rows = []
    for q in range(n_questions):
        for a in range(n_agents):
            base = r_C if pre_correct[q, a] else r_I
            mult = (1 - peer_influence) + peer_influence * disagree[q, a] / mean_disagree
            rate = float(np.clip(base * mult, 0.0, 1.0))
            post = int(pre[q, a])
            if rng.random() < rate:
                if pre_correct[q, a]:
                    post = int(rng.choice([i for i in range(n_options) if i != post]))
                elif rng.random() < s:
                    post = int(gold[q])
                else:
                    others = [i for i in range(n_options) if i not in (post, gold[q])]
                    post = int(rng.choice(others)) if others else post
            post_correct = post == gold[q]
            revised = int(post != pre[q, a])
            label = (
                NONE
                if not revised
                else (BENEFICIAL if post_correct else (HARMFUL if pre_correct[q, a] else UNHELPFUL))
            )
            rows.append(
                {
                    "run_id": "sim",
                    "condition": condition,
                    "question_id": f"q{q:06d}",
                    "agent_id": a,
                    "subject": "sim",
                    "n_options": n_options,
                    "gold_display": int(gold[q]),
                    "pre_answer": int(pre[q, a]),
                    "post_answer": post,
                    "pre_correct": int(pre_correct[q, a]),
                    "post_correct": int(post_correct),
                    "pre_confidence": float(rng.uniform(0.3, 0.95)),
                    "post_confidence": float(rng.uniform(0.3, 0.95)),
                    "peer_disagree_frac": float(disagree[q, a]),
                    "revised": revised,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def separation_for_auroc(target_auroc: float) -> float:
    """Mean separation of two unit-variance Gaussians giving the requested AUROC."""
    return float(np.sqrt(2) * stats.norm.ppf(np.clip(target_auroc, 0.5001, 0.9999)))


def attach_probe_scores(
    events: pd.DataFrame, target_auroc: float, seed: int = 0, column: str = "post_correct"
) -> np.ndarray:
    """Simulate a probe achieving ``target_auroc`` against ``column``."""
    rng = np.random.default_rng(seed)
    y = events[column].to_numpy()
    d = separation_for_auroc(target_auroc)
    latent = rng.normal(size=len(y)) + d * y
    return stats.norm.cdf(latent - d / 2)


def simulate_once(
    n_questions: int, probe_auroc: float, seed: int = 0, val_frac: float = 0.4, **kw
) -> dict:
    """One replicate of the REAL protocol: tune tau on validation, evaluate once on test.

    Thresholding at a fixed 0.5 is not a neutral default and must not be simulated as one. Only
    about a third of revisions are beneficial, so a fixed cut accepts far too many and a gate at
    AUROC 0.68 can come out BEHIND ungated debate. Selecting tau on held-out validation data is
    part of the method, not a refinement of it.
    """
    from mad.analysis.gating import apply_gate, select_threshold
    from mad.analysis.labels import apply_policy

    ev = simulate_events(n_questions, seed=seed, **kw)
    ev["post_confidence"] = attach_probe_scores(ev, probe_auroc, seed=seed + 1)
    ev["pre_confidence"] = 1.0 - ev["post_confidence"]

    qids = ev["question_id"].unique()
    n_val = int(len(qids) * val_frac)
    val_ids = set(qids[:n_val])
    val = ev[ev["question_id"].isin(val_ids)].reset_index(drop=True)
    test = ev[~ev["question_id"].isin(val_ids)].reset_index(drop=True)

    scores_val = val["post_confidence"].to_numpy()
    tau, _ = select_threshold(val, None, scores_val, rule="single")
    accept = apply_gate(test, None, test["post_confidence"].to_numpy(), tau, rule="single")

    gated = apply_policy(test, accept).sort_values("question_id").reset_index(drop=True)
    mad = (
        apply_policy(test, policy_accept(test, "mad_standard"))
        .sort_values("question_id")
        .reset_index(drop=True)
    )
    result = mcnemar(gated["panel_correct"], mad["panel_correct"])
    return {
        "n_questions": n_questions,
        "n_test_questions": int(test["question_id"].nunique()),
        "probe_auroc": probe_auroc,
        "tau": tau,
        "panel_gated": float(gated["panel_correct"].mean()),
        "panel_mad": float(mad["panel_correct"].mean()),
        "delta": result.delta,
        "n_discordant": result.n_discordant,
        "p_value": result.p_value,
        "oracle_flip_rate_f": panel_flip_rate(test),
    }


def power_curve(
    n_questions_grid,
    probe_auroc: float = 0.65,
    n_reps: int = 50,
    alpha: float = 0.05,
    seed: int = 0,
    **kw,
) -> pd.DataFrame:
    """Empirical power of the end-to-end McNemar test across test-set sizes."""
    rows = []
    for n in n_questions_grid:
        results = [simulate_once(n, probe_auroc, seed=seed + 1000 * i, **kw) for i in range(n_reps)]
        df = pd.DataFrame(results)
        rows.append(
            {
                "n_questions": n,
                "n_test": float(df["n_test_questions"].mean()),
                "probe_auroc": probe_auroc,
                "mean_tau": float(df["tau"].mean()),
                "mean_delta": float(df["delta"].mean()),
                "mean_discordant": float(df["n_discordant"].mean()),
                "power": float(((df["p_value"] < alpha) & (df["delta"] > 0)).mean()),
            }
        )
    return pd.DataFrame(rows)


def auroc_curve(
    auroc_grid, n_questions: int = 3000, n_reps: int = 20, alpha: float = 0.05, seed: int = 0, **kw
) -> pd.DataFrame:
    """Power of the end-to-end test as a function of how good the probe is.

    This is the curve that decides whether panel accuracy can be a primary endpoint at all. The
    headline result from the default parameters: a validated threshold on a moderate-AUROC probe
    is nearly degenerate, because accepting every revision is already net-positive when beneficial
    revisions outnumber harmful ones. The gate only starts to pay at AUROC ~0.80.
    """
    rows = []
    for a in auroc_grid:
        df = pd.DataFrame(
            [simulate_once(n_questions, a, seed=seed + 1000 * i, **kw) for i in range(n_reps)]
        )
        rows.append(
            {
                "probe_auroc": a,
                "n_questions": n_questions,
                "mean_tau": float(df["tau"].mean()),
                "mean_delta_pts": 100 * float(df["delta"].mean()),
                "mean_discordant": float(df["n_discordant"].mean()),
                "power": float(((df["p_value"] < alpha) & (df["delta"] > 0)).mean()),
            }
        )
    return pd.DataFrame(rows)


def expected_labels(
    n_questions: int, p: float, r_C: float, r_I: float, s: float, n_agents: int = 3
) -> dict:
    """Closed-form B/H/U counts -- the arithmetic in docs/PLAN.md section 7."""
    inst = n_agents * n_questions
    H = inst * p * r_C
    changes_from_wrong = inst * (1 - p) * r_I
    B = changes_from_wrong * s
    U = changes_from_wrong * (1 - s)
    return {"instances": inst, "B": B, "H": H, "U": U, "revisions": B + H + U}
