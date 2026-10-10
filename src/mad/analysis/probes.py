"""Linear probes and the representation x target grid.

The study's central question is not "can a probe read revision quality" -- no such label exists
(docs/PLAN.md section 3.2) -- but whether the TRANSITION representation predicts correctness
better than the corresponding single-state one:

        representation      target y_pre            target y_post
        phi_pre             (a) correctness probe   (c) forecast the revision outcome
        phi_post            (b) recall prior state  (d) correctness probe
        phi_paired          (e)                     (f)

    (f) vs (d) and (e) vs (a) are the comparisons the paper turns on.

All cross-validation is grouped by question id. Ungrouped CV would leak, because the same
question contributes three agent rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mad.analysis.stats import auroc

REPRESENTATIONS = ("pre", "post", "diff", "paired")
TARGETS = ("y_pre", "y_post")
DEFAULT_C_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)


class FeatureBank:
    """Aligns activation rows to event rows so ``layer(l)`` returns matched (pre, post) matrices.

    Position-aware: built on mad.analysis.states.StateBank, at ONE named position (default
    ``readout_last``). A state missing at that position is an error, never a substitution.
    """

    def __init__(self, store, events: pd.DataFrame, position: str = "readout_last"):
        from mad.analysis.states import StateBank

        self.store = store
        self.events = events.reset_index(drop=True)
        bank = StateBank(store.run_dir)
        self._pre, _ = bank.rows(self.events, "pre", position)
        self._post, _ = bank.rows(self.events, "post", position)

    def layer(self, layer: int) -> tuple[np.ndarray, np.ndarray]:
        full = self.store.load_layer(layer)
        return full[self._pre], full[self._post]

    def features(self, layer: int, representation: str) -> np.ndarray:
        pre, post = self.layer(layer)
        if representation == "pre":
            return pre
        if representation == "post":
            return post
        if representation == "diff":
            return post - pre
        if representation == "paired":
            return np.hstack([pre, post, post - pre])
        raise ValueError(f"unknown representation {representation!r}")


@dataclass
class Probe:
    model: Pipeline
    layer: int
    representation: str
    target: str
    cv_auroc: float = float("nan")
    best_C: float = float("nan")
    oof: np.ndarray | None = field(default=None, repr=False)

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def _pipeline(C: float, n_components: int | None) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=0)))
    # sklearn>=1.8 deprecates an explicit penalty argument; the default is already L2 ridge
    steps.append(("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")))
    return Pipeline(steps)


def grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float,
    n_splits: int = 5,
    n_components: int | None = None,
) -> np.ndarray:
    """Out-of-fold probabilities under GroupKFold on question id."""
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("need at least 2 question groups for grouped CV")
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = _pipeline(C, n_components).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def fit_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    layer: int = -1,
    representation: str = "post",
    target: str = "y_post",
    C_grid=DEFAULT_C_GRID,
    n_splits: int = 5,
    n_components: int | None = None,
) -> Probe:
    """Select C by grouped CV AUROC, then refit on all supplied rows."""
    y = np.asarray(y).astype(int)
    best = (-np.inf, None, None)
    for C in C_grid:
        oof = grouped_oof(X, y, groups, C, n_splits, n_components)
        mask = ~np.isnan(oof)
        score = auroc(y[mask], oof[mask]) if mask.sum() else float("nan")
        if np.isfinite(score) and score > best[0]:
            best = (score, C, oof)
    cv_auroc, C, oof = best
    model = _pipeline(C, n_components).fit(X, y)
    return Probe(model, layer, representation, target, cv_auroc, C, oof)


def mass_mean_probe(X: np.ndarray, y: np.ndarray, ridge: float = 1.0) -> np.ndarray:
    """Difference-of-means direction, optionally whitened. Cheap, robust, few knobs."""
    y = np.asarray(y).astype(bool)
    direction = X[y].mean(axis=0) - X[~y].mean(axis=0)
    if ridge <= 0:
        return direction
    cov = np.cov(X, rowvar=False) + ridge * np.eye(X.shape[1])
    return np.linalg.solve(cov, direction)


def mass_mean_oof(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, ridge: float = 1.0, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold mass-mean scores under GroupKFold.

    Scoring a mass-mean direction in-sample is meaningless whenever d > n: the difference of
    means separates the training rows perfectly by construction and reports AUROC 1.0.
    """
    y = np.asarray(y).astype(int)
    out = np.full(len(y), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        mu = X[tr].mean(axis=0)
        sd = X[tr].std(axis=0) + 1e-8
        direction = mass_mean_probe((X[tr] - mu) / sd, y[tr], ridge=ridge)
        out[te] = ((X[te] - mu) / sd) @ direction
    return out


def layer_sweep(
    bank: FeatureBank,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    representation: str = "post",
    layers=None,
    C_grid=DEFAULT_C_GRID,
    n_components: int | None = None,
) -> pd.DataFrame:
    layers = list(layers) if layers is not None else list(bank.store.layer_range())
    rows = []
    for layer in layers:
        X = bank.features(layer, representation)
        probe = fit_probe(
            X,
            y,
            groups,
            layer=layer,
            representation=representation,
            C_grid=C_grid,
            n_components=n_components,
        )
        rows.append(
            {
                "layer": layer,
                "representation": representation,
                "cv_auroc": probe.cv_auroc,
                "C": probe.best_C,
            }
        )
    return pd.DataFrame(rows)


def representation_target_grid(
    bank: FeatureBank,
    events: pd.DataFrame,
    *,
    layer: int,
    groups: np.ndarray | None = None,
    representations=REPRESENTATIONS,
    C_grid=DEFAULT_C_GRID,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Fill the grid. Row (representation, target) reports grouped-CV AUROC."""
    groups = events["question_id"].to_numpy() if groups is None else groups
    targets = {
        "y_pre": events["pre_correct"].to_numpy(),
        "y_post": events["post_correct"].to_numpy(),
    }
    rows = []
    for rep in representations:
        X = bank.features(layer, rep)
        for tname, y in targets.items():
            probe = fit_probe(
                X,
                y,
                groups,
                layer=layer,
                representation=rep,
                target=tname,
                C_grid=C_grid,
                n_components=n_components,
            )
            rows.append(
                {
                    "representation": rep,
                    "target": tname,
                    "layer": layer,
                    "cv_auroc": probe.cv_auroc,
                    "C": probe.best_C,
                    "n": len(y),
                    "positive_rate": float(np.mean(y)),
                }
            )
    return pd.DataFrame(rows)
