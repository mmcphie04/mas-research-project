"""Inference utilities.

Two things matter here and are easy to get wrong:

1. Rows are CLUSTERED by question -- three agents share every question -- so the independence
   assumption behind a plain AUROC standard error (and behind DeLong) is violated. The cluster
   bootstrap over questions is the default; DeLong is provided for reference and comparison and
   should be reported as such, never as the primary interval.

2. CONDITIONAL (within-question) AUROC is the guard against the difficulty-detector shortcut
   (docs/RISKS.md risk 1). A probe that only reads item difficulty scores well marginally and at
   chance conditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


@dataclass
class Interval:
    point: float
    lo: float
    hi: float
    n: int = 0

    def __repr__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score)))


def conditional_auroc(y, score, groups) -> tuple[float, int, int]:
    """AUROC computed only over (positive, negative) pairs WITHIN the same question.

    Returns (auroc, n_pairs, n_groups_contributing). Chance-level here with a high marginal
    AUROC is the signature of a probe that has learned item difficulty and nothing else.
    """
    df = pd.DataFrame({"y": np.asarray(y), "s": np.asarray(score), "g": np.asarray(groups)})
    conc = ties = total = 0.0
    used = 0
    for _, grp in df.groupby("g", sort=False):
        pos = grp.loc[grp.y == 1, "s"].to_numpy()
        neg = grp.loc[grp.y == 0, "s"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        used += 1
        diff = pos[:, None] - neg[None, :]
        conc += float((diff > 0).sum())
        ties += float((diff == 0).sum())
        total += diff.size
    if total == 0:
        return float("nan"), 0, 0
    return (conc + 0.5 * ties) / total, int(total), used


def cluster_bootstrap(
    fn, groups, *arrays, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Interval:
    """Bootstrap ``fn(*arrays)`` by resampling GROUPS (questions) with replacement."""
    groups = np.asarray(groups)
    arrays = [np.asarray(a) for a in arrays]
    point = fn(*arrays)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    draws[:] = np.nan
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        try:
            draws[b] = fn(*(a[idx] for a in arrays))
        except ValueError:
            draws[b] = np.nan
    good = draws[~np.isnan(draws)]
    if len(good) == 0:
        return Interval(point, float("nan"), float("nan"), len(uniq))
    lo, hi = np.quantile(good, [alpha / 2, 1 - alpha / 2])
    return Interval(float(point), float(lo), float(hi), len(uniq))


def bootstrap_auroc(y, score, groups, **kw) -> Interval:
    return cluster_bootstrap(lambda a, b: auroc(a, b), groups, y, score, **kw)


def bootstrap_delta_auroc(y, score_a, score_b, groups, **kw) -> Interval:
    """Paired cluster bootstrap of AUROC(a) - AUROC(b) on the same rows."""
    return cluster_bootstrap(
        lambda yy, sa, sb: auroc(yy, sa) - auroc(yy, sb), groups, y, score_a, score_b, **kw
    )


# ------------------------------------------------------------------------- DeLong (reference)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong (Sun & Xu 2014). ``scores`` is (k, n). Returns (aucs, covariance).

    Assumes independent observations; with three agents per question that assumption fails, so
    use this only alongside the cluster bootstrap.
    """
    y = np.asarray(y).astype(int)
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    order = np.argsort(-y, kind="stable")
    y, scores = y[order], scores[:, order]
    m = int(y.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        raise ValueError("DeLong needs both classes present")
    k = scores.shape[0]
    pos, neg = scores[:, :m], scores[:, m:]
    tx = np.stack([_midrank(pos[r]) for r in range(k)])
    ty = np.stack([_midrank(neg[r]) for r in range(k)])
    tz = np.stack([_midrank(scores[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    return aucs, sx / m + sy / n


def delong_paired_test(y, score_a, score_b) -> tuple[float, float, float]:
    """Return (auc_a, auc_b, two-sided p) for the paired AUROC difference."""
    aucs, cov = delong(y, np.vstack([np.asarray(score_a), np.asarray(score_b)]))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2 * stats.norm.sf(abs(z)))


# ------------------------------------------------------------------------------ paired tests


@dataclass
class McNemarResult:
    n_discordant: int
    b_favours_a: int
    b_favours_b: int
    p_value: float
    delta: float

    def __repr__(self) -> str:
        return (
            f"McNemar: delta={self.delta:+.4f}, discordant={self.n_discordant} "
            f"({self.b_favours_a} vs {self.b_favours_b}), p={self.p_value:.4g}"
        )


def mcnemar(correct_a, correct_b) -> McNemarResult:
    """Exact paired test on matched binary outcomes. ``a`` is the system under test."""
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError("mcnemar needs matched arrays")
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n_d = only_a + only_b
    p = 1.0 if n_d == 0 else float(stats.binomtest(only_a, n_d, 0.5).pvalue)
    return McNemarResult(n_d, only_a, only_b, p, float(a.mean() - b.mean()))


def mcnemar_power(n_discordant: int, pi: float = 0.65, alpha: float = 0.05) -> float:
    """Approximate power of the paired test given ``n_discordant`` and a true split ``pi``."""
    if n_discordant <= 0:
        return 0.0
    z = stats.norm.isf(alpha / 2)
    se0, se1 = 0.5, np.sqrt(pi * (1 - pi))
    return float(stats.norm.sf((z * se0 - abs(pi - 0.5) * np.sqrt(n_discordant)) / se1))


def required_discordant(pi: float = 0.65, alpha: float = 0.05, power: float = 0.8) -> int:
    """Discordant pairs needed to detect a ``pi``/(1-pi) split. 65/35 -> ~85; 60/40 -> ~194."""
    z_a, z_b = stats.norm.isf(alpha / 2), stats.norm.isf(1 - power)
    num = z_a * 0.5 + z_b * np.sqrt(pi * (1 - pi))
    return int(np.ceil((num / abs(pi - 0.5)) ** 2))
