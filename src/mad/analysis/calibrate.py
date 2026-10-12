"""Calibration.

Calibration is a core result here, not a diagnostic: the gating rule compares two probability
estimates (accept iff g_post - g_pre > tau), so a miscalibrated pair produces a bad decision even
when both probes rank well (docs/PLAN.md section 3.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattScaler:
    """Monotone calibration: never reverses a model's ranking.

    If the fitted slope is not positive -- the scores carry no usable ranking on the calibration
    data, as happened with near-constant scores on the 12B pilot -- the calibrator outputs the
    calibration data's base rate for every score (``constant``), instead of a flipped ranking.
    """

    def __init__(self):
        self._lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        self.constant: float | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> PlattScaler:
        y = np.asarray(y).astype(int)
        self._lr.fit(np.asarray(scores).reshape(-1, 1), y)
        self.constant = float(y.mean()) if self._lr.coef_[0][0] <= 0 else None
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(np.asarray(scores)), self.constant)
        return self._lr.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]


class IsotonicScaler:
    def __init__(self):
        self._ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> IsotonicScaler:
        self._ir.fit(np.asarray(scores), np.asarray(y).astype(float))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self._ir.predict(np.asarray(scores))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        rows.append(
            {
                "bin": b,
                "lo": edges[b],
                "hi": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else np.nan,
                "empirical": float(y[mask].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))
