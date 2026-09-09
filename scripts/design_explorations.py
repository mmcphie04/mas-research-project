"""Design questions answered by simulation, before spending GPU time.

Run: uv run python scripts/design_explorations.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mad.analysis.labels import apply_policy
from mad.analysis.oracles import evaluate_all, policy_accept
from mad.analysis.power_sim import auroc_curve, simulate_events
from mad.analysis.stats import mcnemar

REGIMES = {
    "helps": dict(p=0.42, r_C=0.14, r_I=0.28, s=0.45),
    "neutral": dict(p=0.42, r_C=0.22, r_I=0.24, s=0.42),
    "hurts": dict(p=0.42, r_C=0.30, r_I=0.20, s=0.35),
}
N = 3000


def panel(ev, policy):
    acc = policy if isinstance(policy, np.ndarray) else policy_accept(ev, policy)
    return apply_policy(ev, acc).sort_values("question_id").reset_index(drop=True)


def q1_agent_count():
    """Does a bigger panel help, and does it change how much room the gate has?"""
    rows = []
    for regime, kw in REGIMES.items():
        for k in (3, 5, 7):
            ev = simulate_events(N, n_agents=k, peer_influence=0.6, seed=0, **kw)
            t = evaluate_all(ev).set_index("policy")
            r0 = t.loc["round0_plurality", "panel_accuracy"]
            mad = t.loc["mad_standard", "panel_accuracy"]
            opt = t.loc["oracle_panel_optimal", "panel_accuracy"]
            ties = t.loc["mad_standard", "tie_rate"]
            rows.append(
                {
                    "regime": regime,
                    "n_agents": k,
                    "round0": round(r0, 4),
                    "mad": round(mad, 4),
                    "debate_gain": round(mad - r0, 4),
                    "headroom": round(opt - mad, 4),
                    "tie_rate": round(ties, 3),
                }
            )
    return pd.DataFrame(rows)


def q2_selective_debate():
    """Accuracy vs compute: debate only the questions where round-0 agents disagree.

    Compute unit = 1 generation per agent per round. Round 0 always costs 1.0; debating a
    question costs another 1.0. Selecting on round-0 disagreement is FREE -- it needs no probe.
    """
    rows = []
    for regime, kw in REGIMES.items():
        ev = simulate_events(N, n_agents=3, peer_influence=0.6, seed=0, **kw)
        mad = panel(ev, "mad_standard")
        r0 = panel(ev, "round0_plurality")
        # per-question round-0 unanimity
        uni = ev.groupby("question_id")["pre_answer"].nunique().eq(1)
        uni = uni.reindex(mad["question_id"]).to_numpy()
        for name, debate_mask in [
            ("debate all", np.ones(len(mad), dtype=bool)),
            ("skip unanimous", ~uni),
            ("debate none", np.zeros(len(mad), dtype=bool)),
        ]:
            correct = np.where(debate_mask, mad["panel_correct"], r0["panel_correct"])
            compute = 1.0 + debate_mask.mean()
            rows.append(
                {
                    "regime": regime,
                    "policy": name,
                    "accuracy": round(float(correct.mean()), 4),
                    "compute": round(float(compute), 3),
                    "frac_debated": round(float(debate_mask.mean()), 3),
                    "acc_per_compute": round(float(correct.mean()) / compute, 4),
                }
            )
    return pd.DataFrame(rows)


def q3_confidence_weighted_vote():
    """Weight each agent's vote by its readout confidence instead of one-agent-one-vote."""
    rows = []
    for regime, kw in REGIMES.items():
        ev = simulate_events(N, n_agents=3, peer_influence=0.6, seed=0, **kw)
        # give confidence real signal: separation d against correctness
        rng = np.random.default_rng(1)
        for phase in ("pre", "post"):
            y = ev[f"{phase}_correct"].to_numpy()
            ev[f"{phase}_confidence"] = 1 / (1 + np.exp(-(rng.normal(size=len(y)) + 1.0 * y)))
        plain = panel(ev, "mad_standard")
        weighted = []
        for _, g in ev.groupby("question_id", sort=False):
            score: dict[int, float] = {}
            for a, c in zip(g["post_answer"], g["post_confidence"], strict=True):
                score[a] = score.get(a, 0.0) + c
            best = max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            weighted.append(int(best == g["gold_display"].iloc[0]))
        weighted = np.array(weighted)
        test = mcnemar(weighted, plain["panel_correct"].to_numpy())
        rows.append(
            {
                "regime": regime,
                "plain_majority": round(float(plain["panel_correct"].mean()), 4),
                "confidence_weighted": round(float(weighted.mean()), 4),
                "delta_pts": round(100 * test.delta, 2),
                "n_discordant": test.n_discordant,
                "p": round(test.p_value, 4),
            }
        )
    return pd.DataFrame(rows)


def q4_gate_power_by_regime():
    rows = []
    for regime, kw in REGIMES.items():
        c = auroc_curve([0.70, 0.80], n_questions=N, n_reps=15, peer_influence=0.6, **kw)
        for _, r in c.iterrows():
            rows.append(
                {
                    "regime": regime,
                    "probe_auroc": r.probe_auroc,
                    "delta_pts": round(r.mean_delta_pts, 2),
                    "power": r.power,
                }
            )
    return pd.DataFrame(rows)


def q5_probe_weighted_voting(rho: float = 0.5):
    """Weight each agent's vote by a probe of given CONDITIONAL (within-question) AUROC.

    Conditional AUROC, not marginal, is the decision-relevant quantity for weighted voting:
    probe noise that is shared across agents on a question cancels when you compare those agents
    to each other, so what matters is discrimination *within* a question. This is the same
    statistic that serves as the difficulty-detector guard (docs/RISKS.md risk 1) -- a probe that
    only reads item difficulty has conditional AUROC 0.5 and is worthless for weighting, however
    good its marginal AUROC looks.

    ``rho`` is the share of probe noise common to a question.
    """
    from scipy import stats as sps

    from mad.analysis.stats import auroc, conditional_auroc

    rows = []
    for regime, kw in REGIMES.items():
        ev = simulate_events(N, n_agents=3, peer_influence=0.6, seed=0, **kw)
        plain = panel(ev, "mad_standard")
        uniq = {q: i for i, q in enumerate(plain["question_id"])}
        qidx = np.array([uniq[q] for q in ev["question_id"]])
        y = ev["post_correct"].to_numpy()
        ans = ev["post_answer"].to_numpy()
        gold = plain["gold_display"].to_numpy()

        for cond_target in (0.60, 0.65, 0.70, 0.75):
            d = np.sqrt(2) * np.sqrt(1 - rho) * sps.norm.ppf(cond_target)
            rng = np.random.default_rng(7)
            latent = (
                d * y
                + np.sqrt(rho) * rng.normal(size=len(plain))[qidx]
                + np.sqrt(1 - rho) * rng.normal(size=len(y))
            )
            w = sps.norm.cdf(latent - d / 2)
            correct = np.zeros(len(plain), dtype=int)
            for i in range(len(plain)):
                m = qidx == i
                score: dict[int, float] = {}
                for a, c in zip(ans[m], w[m], strict=True):
                    score[a] = score.get(a, 0.0) + c
                correct[i] = int(max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0] == gold[i])
            t = mcnemar(correct, plain["panel_correct"].to_numpy())
            rows.append(
                {
                    "regime": regime,
                    "cond_auroc": round(conditional_auroc(y, w, ev["question_id"])[0], 3),
                    "marg_auroc": round(auroc(y, w), 3),
                    "plain": round(float(plain["panel_correct"].mean()), 4),
                    "weighted": round(float(correct.mean()), 4),
                    "delta_pts": round(100 * t.delta, 2),
                    "n_disc": t.n_discordant,
                    "p": float(f"{t.p_value:.2g}"),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for title, fn in [
        ("Q1  panel size", q1_agent_count),
        ("Q2  selective debate (accuracy vs compute)", q2_selective_debate),
        ("Q3  confidence-weighted voting", q3_confidence_weighted_vote),
        ("Q4  gate power by regime", q4_gate_power_by_regime),
        ("Q5  probe-weighted voting", q5_probe_weighted_voting),
    ]:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        print(fn().to_string(index=False))
