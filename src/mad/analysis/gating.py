"""The intervention: decide, per agent, whether to keep the round-0 answer or take the revision.

Two rules, both reported:

    two-probe   accept iff  g_post - g_pre > tau      (the decision-theoretic one)
    single      accept iff  g_post > tau              (the simpler one)

tau is selected on VALIDATION against simulated panel accuracy -- not F1, not Youden. The cost
matrix is asymmetric: accepting an unhelpful revision is free, accepting a harmful one costs a
correct answer, and rejecting a beneficial one costs a chance at gaining one.

Because a single debate round means the gate only ever picks between two already-generated
answers, all of this runs offline with no model in the loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mad.analysis.oracles import evaluate


def two_probe_gate(g_pre: np.ndarray, g_post: np.ndarray, tau: float = 0.0) -> np.ndarray:
    return (np.asarray(g_post) - np.asarray(g_pre)) > tau


def single_probe_gate(g_post: np.ndarray, tau: float = 0.5) -> np.ndarray:
    return np.asarray(g_post) > tau


def _force_unrevised(events: pd.DataFrame, accept: np.ndarray) -> np.ndarray:
    """Non-revised events are no-ops; force accept so the gate only acts where it can."""
    accept = np.asarray(accept).astype(bool).copy()
    accept[events["revised"].to_numpy() != 1] = True
    return accept


def sweep_thresholds(
    events: pd.DataFrame,
    g_pre: np.ndarray | None,
    g_post: np.ndarray,
    taus: np.ndarray | None = None,
    rule: str = "two_probe",
) -> pd.DataFrame:
    if taus is None:
        # 41 points is plenty: the panel-accuracy objective is a step function in tau with at most
        # one step per revision event, and a finer grid only re-evaluates identical policies.
        taus = np.linspace(-0.5, 0.5, 41) if rule == "two_probe" else np.linspace(0.0, 1.0, 41)
    rows = []
    for tau in taus:
        accept = (
            two_probe_gate(g_pre, g_post, tau)
            if rule == "two_probe"
            else single_probe_gate(g_post, tau)
        )
        accept = _force_unrevised(events, accept)
        rows.append(
            {"tau": float(tau), "rule": rule, **evaluate(events, accept, f"{rule}@{tau:.3f}")}
        )
    return pd.DataFrame(rows)


def select_threshold(
    events: pd.DataFrame,
    g_pre: np.ndarray | None,
    g_post: np.ndarray,
    rule: str = "two_probe",
    taus: np.ndarray | None = None,
    objective: str = "panel_accuracy",
) -> tuple[float, pd.DataFrame]:
    """Pick tau on validation data. Returns (tau, full sweep) so the curve can be plotted."""
    sweep = sweep_thresholds(events, g_pre, g_post, taus, rule)
    best = sweep.loc[sweep[objective].idxmax()]
    return float(best["tau"]), sweep


def apply_gate(
    events: pd.DataFrame,
    g_pre: np.ndarray | None,
    g_post: np.ndarray,
    tau: float,
    rule: str = "two_probe",
) -> np.ndarray:
    accept = (
        two_probe_gate(g_pre, g_post, tau)
        if rule == "two_probe"
        else single_probe_gate(g_post, tau)
    )
    return _force_unrevised(events, accept)
