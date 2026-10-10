"""Revision gates: fit on training questions, choose the threshold on the SAME calibrated score
used at evaluation, score held-out questions once.

A gate accepts an agent's revision iff  P(new answer correct) - P(old answer correct) > tau.
Non-revised agents are unaffected. tau = -inf is accept-every-revision, +inf is reject-every-
revision; both are always candidates, so a learned gate can never be forced to do worse on its
selection data than either endpoint.

Score construction, identical at selection and evaluation:
  raw model probability -> Platt calibrator -> calibrated probability -> difference -> tau.
The previous script chose tau on RAW out-of-fold probabilities and then applied it to
CALIBRATED test probabilities; with a non-identity calibrator that compares a threshold on one
scale with scores on another.

Two routes:
  * cross-fitted (exploratory CV): inside the training questions, out-of-fold raw predictions;
    a calibrator fitted on the OTHER inner folds is applied to each fold, so tau is chosen on
    calibrated scores none of whose calibration saw that row. The final calibrator is fitted
    on all out-of-fold scores.
  * validation (preregistered split): models fit on train; calibrator and tau chosen on
    validation; test scored once.

Threshold objective: panel accuracy (the primary endpoint) under the declared tie-break, or
agent accuracy (reported as a secondary rule). Ties between thresholds go to the one nearest 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mad.analysis.calibrate import PlattScaler
from mad.analysis.labels import apply_policy
from mad.analysis.stats import auroc

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
HGB_GRID = ({"max_iter": 100, "max_depth": 3}, {"max_iter": 300, "max_depth": 3})
TAU_GRID = np.concatenate([[-np.inf], np.round(np.linspace(-0.5, 0.5, 41), 4), [np.inf]])
OBJECTIVES = ("panel", "agent")


def linear(C):
    return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=4000))])


def hgb(params):
    return HistGradientBoostingClassifier(learning_rate=0.05, random_state=0, **params)


def cv_auroc(make, X, y, g, k=3) -> float:
    s = []
    for a, b in GroupKFold(n_splits=k).split(X, y, g):
        if len(np.unique(y[a])) > 1 and len(np.unique(y[b])) > 1:
            s.append(auroc(y[b], make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]))
    return float(np.mean(s)) if s else -np.inf


def oof(make, X, y, g, k=5) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities and the inner fold of each row."""
    p, fold = np.full(len(y), np.nan), np.full(len(y), -1)
    for f, (a, b) in enumerate(GroupKFold(n_splits=k).split(X, y, g)):
        p[b] = make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        fold[b] = f
    return p, fold


def crossfit_calibrate(raw, y, fold) -> np.ndarray:
    """Calibrated out-of-fold scores: each inner fold calibrated by a Platt fit on the others."""
    out = np.full(len(raw), np.nan)
    for f in np.unique(fold):
        m = fold == f
        if len(np.unique(y[~m])) < 2:
            out[m] = raw[m]
            continue
        out[m] = PlattScaler().fit(raw[~m], y[~m]).transform(raw[m])
    return out


def gate_accept(p_new, p_old, revised, tau) -> np.ndarray:
    return ((p_new - p_old) > tau) | ~revised


def objective_value(ev, accept, objective, tie_break) -> float:
    if objective == "panel":
        return float(apply_policy(ev, accept, tie_break).panel_correct.mean())
    y_old, y_new = ev.pre_correct.to_numpy(), ev.post_correct.to_numpy()
    return float(np.mean(np.where(accept, y_new, y_old)))


def choose_tau(ev, p_new, p_old, objective, tie_break) -> tuple[float, float]:
    """(tau, objective value) over TAU_GRID, on the calibrated scores given."""
    rev = (ev.revised == 1).to_numpy()
    vals = [objective_value(ev, gate_accept(p_new, p_old, rev, t), objective, tie_break) for t in TAU_GRID]
    best = max(vals)
    cands = [t for t, v in zip(TAU_GRID, vals, strict=True) if v == best]
    tau = min(cands, key=lambda t: abs(t))
    return float(tau), float(best)


# ----------------------------------------------------------------------------- feature sources


class Features:
    """A predictor family's inputs. ``layers`` is empty for output-only families; for activation
    families ``matrix(layer)`` returns the feature matrix at that layer for ALL events."""

    def __init__(self, kind, output=None, act=None, layers=()):
        self.kind, self.output, self.act, self.layers = kind, output, act, list(layers)

    def matrix(self, layer=None) -> np.ndarray:
        if self.kind in ("output_linear", "output_hgb"):
            return self.output
        A = self.act(layer)
        return A if self.kind == "activation" else np.hstack([A, self.output])


def select(F: Features, X_rows, y, g):
    """Hyperparameters by grouped CV AUROC on the given (training) rows only.
    Returns (factory, params)."""
    if F.kind == "output_hgb":
        X = F.matrix()[X_rows]
        bp = max(HGB_GRID, key=lambda p: cv_auroc(lambda p=p: hgb(p), X, y, g))
        return (lambda p=bp: hgb(p)), dict(bp)
    if F.kind == "output_linear":
        X = F.matrix()[X_rows]
        bc = max(C_GRID, key=lambda C: cv_auroc(lambda C=C: linear(C), X, y, g))
        return (lambda C=bc: linear(C)), {"C": bc}
    best_l = max(F.layers, key=lambda L: cv_auroc(lambda: linear(1e-2), F.matrix(L)[X_rows], y, g))
    X = F.matrix(best_l)[X_rows]
    bc = max(C_GRID, key=lambda C: cv_auroc(lambda C=C: linear(C), X, y, g))
    return (lambda C=bc: linear(C)), {"layer": int(best_l), "C": bc}


def _X(F, params, rows):
    return F.matrix(params.get("layer"))[rows]


def fit_crossfit(F: Features, ev: pd.DataFrame, tr: np.ndarray, tie_break: str) -> dict:
    """Everything decided from training rows ``tr`` (boolean mask over ``ev``). No other rows'
    labels are read."""
    g = ev.question_id.to_numpy()[tr]
    targets = {"old": ev.pre_correct.to_numpy()[tr].astype(int),
               "new": ev.post_correct.to_numpy()[tr].astype(int)}
    out = {"models": {}, "params": {}, "calibrators": {}, "oof_raw": {}, "oof_cal": {}}
    for t, y in targets.items():
        make, prm = select(F, tr, y, g)
        X = _X(F, prm, tr)
        raw, fold = oof(make, X, y, g)
        out["oof_raw"][t], out["oof_cal"][t] = raw, crossfit_calibrate(raw, y, fold)
        out["oof_fold"] = fold
        out["calibrators"][t] = PlattScaler().fit(raw, y)
        out["models"][t] = make().fit(X, y)
        out["params"][t] = prm
    sub = ev[tr].reset_index(drop=True)
    out["tau"] = {o: choose_tau(sub, out["oof_cal"]["new"], out["oof_cal"]["old"], o, tie_break)[0]
                  for o in OBJECTIVES}
    return out


def fit_validation(F: Features, ev: pd.DataFrame, tr, va, tie_break: str) -> dict:
    """Models on train; calibrators and tau on validation."""
    g = ev.question_id.to_numpy()[tr]
    out = {"models": {}, "params": {}, "calibrators": {}, "val_raw": {}, "val_cal": {}}
    for t, col in (("old", "pre_correct"), ("new", "post_correct")):
        y = ev[col].to_numpy()[tr].astype(int)
        make, prm = select(F, tr, y, g)
        model = make().fit(_X(F, prm, tr), y)
        raw = model.predict_proba(_X(F, prm, va))[:, 1]
        cal = PlattScaler().fit(raw, ev[col].to_numpy()[va].astype(int))
        out["models"][t], out["params"][t], out["calibrators"][t] = model, prm, cal
        out["val_raw"][t], out["val_cal"][t] = raw, cal.transform(raw)
    sub = ev[va].reset_index(drop=True)
    out["tau"] = {o: choose_tau(sub, out["val_cal"]["new"], out["val_cal"]["old"], o, tie_break)[0]
                  for o in OBJECTIVES}
    return out


def score(fitted: dict, F: Features, rows) -> dict:
    """Raw and calibrated probabilities for ``rows`` -- the one scoring path used everywhere."""
    out = {}
    for t in ("old", "new"):
        raw = fitted["models"][t].predict_proba(_X(F, fitted["params"][t], rows))[:, 1]
        out[f"{t}_raw"], out[f"{t}_cal"] = raw, fitted["calibrators"][t].transform(raw)
    return out


def calibrator_params(cal: PlattScaler) -> dict:
    lr = cal._lr
    return {"coef": float(lr.coef_[0][0]), "intercept": float(lr.intercept_[0])}
