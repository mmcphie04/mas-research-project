"""Reference policies and the headroom analysis that gates the whole project.

Every policy is a label -> accept decision. The two endpoints are the systems the paper compares
against; the oracles bound what any learned gate could achieve.

The number that matters for go/no-go G5 is the PANEL-level headroom, not the agent-level one:
majority voting is nonlinear, so a large net revision gain can vanish at the panel
(docs/PLAN.md section 3.1). Note also that rejecting UNHELPFUL revisions changes no agent's
correctness yet can still move a panel vote -- which is why it gets its own row.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from mad.analysis.labels import BENEFICIAL, HARMFUL, NONE, UNHELPFUL, apply_policy, panel_vote

POLICIES: dict[str, dict[str, bool]] = {
    # standard one-round debate: every proposed revision stands
    "mad_standard": {BENEFICIAL: True, HARMFUL: True, UNHELPFUL: True},
    # never revise: equivalent to the round-0 plurality vote
    "round0_plurality": {BENEFICIAL: False, HARMFUL: False, UNHELPFUL: False},
    # upper bound on any gate
    "oracle_full": {BENEFICIAL: True, HARMFUL: False, UNHELPFUL: False},
    # the realistic target of a conservative gate
    "oracle_reject_harmful": {BENEFICIAL: True, HARMFUL: False, UNHELPFUL: True},
    # changes no agent's correctness, but can still flip a panel vote
    "oracle_reject_unhelpful": {BENEFICIAL: True, HARMFUL: True, UNHELPFUL: False},
}


def policy_accept(events: pd.DataFrame, policy: dict[str, bool] | str) -> np.ndarray:
    if isinstance(policy, str):
        policy = POLICIES[policy]
    table = {**policy, NONE: True}
    return events["label"].map(table).to_numpy(dtype=bool)


def panel_optimal_accept(events: pd.DataFrame, tie_break: str = "round0_then_agent0") -> np.ndarray:
    """The TRUE panel ceiling: per question, the accept subset that maximises panel correctness.

    ``oracle_full`` (accept exactly the beneficial revisions) is optimal for each agent taken
    alone, but it is NOT a panel upper bound. Rejecting an unhelpful revision restores an agent's
    original wrong answer, which can consolidate a wrong plurality that the revision had split --
    so the per-agent-optimal policy can lose a question the ungated panel won. With three agents
    there are at most eight accept subsets per question, so the exact optimum is cheap.

    Ties are broken toward fewer acceptances, which keeps the result deterministic and reports
    the most conservative gate achieving the ceiling.
    """
    accept = np.ones(len(events), dtype=bool)
    ev = events.reset_index(drop=True)
    for _, grp in ev.groupby(["run_id", "condition", "question_id"], sort=False):
        grp = grp.sort_values("agent_id")
        revised_pos = grp.index[grp["revised"].to_numpy() == 1].tolist()
        if not revised_pos:
            continue
        gold = grp["gold_display"].iloc[0]
        pre = grp["pre_answer"].tolist()
        post = grp["post_answer"].tolist()
        conf_pre = grp["pre_confidence"].tolist()
        conf_post = grp["post_confidence"].tolist()
        order = {pos: i for i, pos in enumerate(grp.index)}
        best = None
        for combo in product([False, True], repeat=len(revised_pos)):
            take = {revised_pos[i]: combo[i] for i in range(len(revised_pos))}
            answers, confs = [], []
            for pos in grp.index:
                i = order[pos]
                use_post = take.get(pos, True)
                answers.append(post[i] if use_post else pre[i])
                confs.append(conf_post[i] if use_post else conf_pre[i])
            correct = int(panel_vote(answers, confs, pre, tie_break) == gold)
            key = (correct, -sum(combo))
            if best is None or key > best[0]:
                best = (key, take)
        for pos, use_post in best[1].items():
            accept[pos] = use_post
    return accept


def random_accept(events: pd.DataFrame, rate: float, seed: int = 0) -> np.ndarray:
    """Accept a random ``rate`` fraction of REVISED events; the matched-rate control."""
    rng = np.random.default_rng(seed)
    accept = np.ones(len(events), dtype=bool)
    revised = events["revised"].to_numpy() == 1
    accept[revised] = rng.random(int(revised.sum())) < rate
    return accept


def evaluate(
    events: pd.DataFrame,
    accept: np.ndarray,
    name: str = "policy",
    tie_break: str = "round0_then_agent0",
) -> dict:
    panel = apply_policy(events, accept, tie_break)
    revised = events["revised"].to_numpy() == 1
    n_rev = int(revised.sum())
    return {
        "policy": name,
        "panel_accuracy": float(panel["panel_correct"].mean()),
        "agent_accuracy": float(panel["agent_accuracy"].mean()),
        "n_questions": int(len(panel)),
        "n_revisions": n_rev,
        "n_accepted": int(np.asarray(accept)[revised].sum()),
        "acceptance_rate": float(np.asarray(accept)[revised].mean()) if n_rev else float("nan"),
        "tie_rate": float(panel["tied"].mean()),
    }


def evaluate_all(
    events: pd.DataFrame,
    extra: dict[str, np.ndarray] | None = None,
    tie_break: str = "round0_then_agent0",
) -> pd.DataFrame:
    rows = [evaluate(events, policy_accept(events, p), p, tie_break) for p in POLICIES]
    rows.append(
        evaluate(events, panel_optimal_accept(events, tie_break), "oracle_panel_optimal", tie_break)
    )
    for name, accept in (extra or {}).items():
        rows.append(evaluate(events, accept, name, tie_break))
    return pd.DataFrame(rows)


def panel_frames(
    events: pd.DataFrame, policy_a: str, policy_b: str, tie_break: str = "round0_then_agent0"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = (
        apply_policy(events, policy_accept(events, policy_a), tie_break)
        .sort_values("question_id")
        .reset_index(drop=True)
    )
    b = (
        apply_policy(events, policy_accept(events, policy_b), tie_break)
        .sort_values("question_id")
        .reset_index(drop=True)
    )
    return a, b


def panel_flip_rate(
    events: pd.DataFrame, policy_a: str = "oracle_full", policy_b: str = "mad_standard"
) -> float:
    """Fraction of questions on which the two policies produce different panel answers.

    This is ``f`` in the sample-size arithmetic: the test split needs roughly 170/f questions for
    the end-to-end comparison to be powered (docs/PLAN.md section 7).
    """
    a, b = panel_frames(events, policy_a, policy_b)
    return float((a["panel_answer"] != b["panel_answer"]).mean())


def headroom(events: pd.DataFrame) -> dict:
    """The Stage-1 gate numbers, agent and panel level, in one dict."""
    table = evaluate_all(events).set_index("policy")
    mad = table.loc["mad_standard"]
    full = table.loc["oracle_full"]
    rej = table.loc["oracle_reject_harmful"]
    r0 = table.loc["round0_plurality"]
    opt = table.loc["oracle_panel_optimal"]
    mad_panel = apply_policy(events, policy_accept(events, "mad_standard")).sort_values(
        "question_id"
    )
    opt_panel = apply_policy(events, panel_optimal_accept(events)).sort_values("question_id")
    f = float((mad_panel["panel_answer"].to_numpy() != opt_panel["panel_answer"].to_numpy()).mean())
    discordant = int(
        (
            (opt_panel["panel_correct"].to_numpy() == 1)
            ^ (mad_panel["panel_correct"].to_numpy() == 1)
        ).sum()
    )
    return {
        "panel_mad": float(mad["panel_accuracy"]),
        "panel_round0": float(r0["panel_accuracy"]),
        "panel_oracle_full": float(full["panel_accuracy"]),
        "panel_oracle_panel_optimal": float(opt["panel_accuracy"]),
        "panel_oracle_reject_harmful": float(rej["panel_accuracy"]),
        "panel_headroom_full": float(full["panel_accuracy"] - mad["panel_accuracy"]),
        "panel_headroom_optimal": float(opt["panel_accuracy"] - mad["panel_accuracy"]),
        "panel_headroom_reject_harmful": float(rej["panel_accuracy"] - mad["panel_accuracy"]),
        "agent_mad": float(mad["agent_accuracy"]),
        "agent_oracle_full": float(full["agent_accuracy"]),
        "agent_headroom_full": float(full["agent_accuracy"] - mad["agent_accuracy"]),
        "debate_gain_over_round0": float(mad["panel_accuracy"] - r0["panel_accuracy"]),
        "panel_flip_rate_f": f,
        "oracle_discordant_questions": discordant,
    }


def self_consistency_at_k(events: pd.DataFrame) -> float:
    """Round-0 majority vote -- three independent samples at matched compute.

    Identical to ``round0_plurality`` by construction; exposed separately because it is the
    baseline the debate system actually has to beat (docs/RISKS.md risk 9).
    """
    panel = apply_policy(events, policy_accept(events, "round0_plurality"))
    return float(panel["panel_correct"].mean())
