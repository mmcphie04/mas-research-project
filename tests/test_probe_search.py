"""Evaluation code of scripts/probe_search.py on synthetic states (no activations needed)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("probe_search", ROOT / "scripts" / "probe_search.py")
PS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PS)


def fake(n_q=40, seed=0):
    """Round-0 panels plus one revised panel per question, with labels and revision pairs."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        pre = rng.integers(0, 2, 3)
        for a in range(3):
            rows.append({"condition": "round0", "question_id": f"q{q}", "agent_id": a, "phase": "pre",
                         "answer": int(rng.integers(0, 4)), "correct": int(pre[a])})
        for a in range(3):
            flip = rng.random() < 0.4
            rows.append({"condition": "debate", "question_id": f"q{q}", "agent_id": a, "phase": "post",
                         "answer": int(rng.integers(0, 4)), "correct": int(1 - pre[a] if flip else pre[a])})
    st = pd.DataFrame(rows)
    D = SimpleNamespace(states=st, y=st.correct.to_numpy(), groups=pd.factorize(st.question_id)[0],
                        panels=PS.panel_ids(st))
    pre_ix = {(r.question_id, r.agent_id): i for i, r in st[st.phase == "pre"].iterrows()}
    pairs = [{"pre": pre_ix[(r.question_id, r.agent_id)], "post": i, "condition": r.condition,
              "beneficial": int(r.correct)}
             for i, r in st[st.phase == "post"].iterrows()
             if st.loc[pre_ix[(r.question_id, r.agent_id)], "correct"] + r.correct == 1]
    return D, pd.DataFrame(pairs)


def test_folds_never_split_a_question():
    g = np.repeat(np.arange(30), 6)
    for seed in range(3):
        f = PS.make_folds(g, 5, seed)
        assert all(len(set(f[g == q])) == 1 for q in range(30))
        assert len(set(f)) == 5


def test_rev_auc_ignores_a_constant_offset_between_phases():
    """The pooled 'pair accuracy' was biased by a phase offset; rev_auc must not be."""
    D, pairs = fake()
    s = D.y + np.random.default_rng(1).normal(0, 0.5, len(D.y))
    folds = PS.make_folds(D.groups, 5, 0)
    present = np.ones(len(D.y), bool)
    base = PS.metrics(D, s, folds, present, pairs)
    shifted = s - 10.0 * (D.states.phase == "post").to_numpy()
    m = PS.metrics(D, shifted, folds, present, pairs)
    assert abs(m["rev_auc"] - base["rev_auc"]) < 1e-12
    assert abs(m["within_panel"] - base["within_panel"]) < 1e-12  # panels never mix phases


def test_within_panel_is_chance_for_scores_constant_within_panels():
    D, pairs = fake()
    s = D.panels.astype(float)  # identical within every panel: no within-panel information
    m = PS.metrics(D, s, PS.make_folds(D.groups, 5, 0), np.ones(len(D.y), bool), pairs)
    assert m["within_panel"] == 0.5


def test_weighted_bootstrap_matches_point_metrics_at_unit_weights():
    D, pairs = fake()
    s = D.y + np.random.default_rng(2).normal(0, 1.0, len(D.y))
    folds = PS.make_folds(D.groups, 5, 0)
    present = np.ones(len(D.y), bool)
    m = PS.metrics(D, s, folds, present, pairs)
    v = PS._boot_values(D, s, folds, present, pairs, np.ones((1, D.groups.max() + 1)))
    for k in ("auroc_all", "within_panel", "rev_auc"):
        assert abs(v[k][0] - m[k]) < 1e-9


def test_letter_centering_uses_training_rows_only():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    letters = np.array([0, 1] * 10)
    tr = np.arange(20) < 10
    before = PS.transform(X, "letter", tr, letters, np.arange(20))
    X2 = X.copy()
    X2[~tr] += 100.0  # changing test rows must not change how training rows are centered
    after = PS.transform(X2, "letter", tr, letters, np.arange(20))
    assert np.allclose(before[tr], after[tr])


def test_residualizing_removes_linear_covariate_information():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(200, 3))
    X = (Z @ rng.normal(size=(3, 6)) + 0.01 * rng.normal(size=(200, 6))).astype(np.float32)
    tr = np.ones(200, bool)
    R = PS.transform(X, "raw+rout", tr, np.zeros(200, int), np.arange(200), {"rout": Z})
    assert np.abs(R).max() < 0.1  # X was (almost) a linear function of Z
