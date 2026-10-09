"""The reanalysis must make every choice -- hyperparameters, threshold, calibration -- from
training questions only. Regression for the earlier version, which chose the gate threshold by
maximising accuracy on the held-out questions' own labels and reported that accuracy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("reanalysis", ROOT / "scripts" / "reanalysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _data(n_q=60, seed=0):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_q), 3).astype(str)
    X = rng.normal(size=(len(g), 6))
    y_old = (X[:, 0] + rng.normal(size=len(g)) > 0).astype(int)
    y_new = (X[:, 1] + rng.normal(size=len(g)) > 0).astype(int)
    rev = rng.random(len(g)) < 0.4
    return X, y_old, y_new, g, rev


def test_held_out_labels_cannot_influence_any_choice():
    R = _mod()
    X, y_old, y_new, g, rev = _data()
    te = np.isin(g, [str(i) for i in range(12)])
    tr = ~te
    fams = ("output_linear", "output_hgb")
    a = R.fit_fold(fams, {f: X for f in fams}, y_old, y_new, g, rev, tr)
    y_old2, y_new2 = y_old.copy(), y_new.copy()
    y_old2[te], y_new2[te] = 1 - y_old2[te], 1 - y_new2[te]  # flip every held-out label
    b = R.fit_fold(fams, {f: X for f in fams}, y_old2, y_new2, g, rev, tr)
    pa = R.predict_fold(a, {f: X for f in fams}, te)
    pb = R.predict_fold(b, {f: X for f in fams}, te)
    for f in fams:
        assert a[f]["tau"] == b[f]["tau"], f"{f}: threshold moved with the held-out labels"
        assert a[f]["params"] == b[f]["params"]
        for t in ("old", "new"):
            assert np.allclose(pa[f][t], pb[f][t]), f"{f}: held-out predictions moved"


def test_threshold_is_chosen_from_training_rows():
    R = _mod()
    p_new = np.array([0.9, 0.2, 0.8, 0.1])
    p_old = np.array([0.1, 0.8, 0.2, 0.9])
    y_new = np.array([1, 0, 1, 0])
    y_old = np.array([0, 1, 0, 1])
    tau = R.choose_tau(p_new, p_old, y_new, y_old, np.ones(4, bool))
    take = p_new - p_old > tau
    assert (np.where(take, y_new, y_old) == 1).all()
