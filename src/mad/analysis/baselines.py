"""Baselines the probe has to beat, and the controls that keep it honest.

The metadata baseline is the claim-killer (docs/RISKS.md risk 2): if a handful of debate features
match the probe, nothing has been shown about internal representations. Incremental AUROC over
metadata -- not raw AUROC -- is the paper's core quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mad.analysis.probes import grouped_oof
from mad.analysis.stats import auroc

METADATA_COLUMNS = [
    "n_peers_agree_pre",
    "peer_unanimous",
    "post_equals_peer_majority",
    "n_options",
    "pre_rationale_chars",
    "post_rationale_chars",
    "pre_confidence",
    "post_confidence",
    "revised",
]


def metadata_features(events: pd.DataFrame, include_subject: bool = True) -> pd.DataFrame:
    X = events[[c for c in METADATA_COLUMNS if c in events.columns]].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if include_subject and "subject" in events.columns:
        X = pd.concat([X, pd.get_dummies(events["subject"], prefix="subj", dtype=float)], axis=1)
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def confidence_baselines(events: pd.DataFrame) -> dict[str, np.ndarray]:
    """Cheap signals available without any probe."""
    return {
        "readout_conf_post": events["post_confidence"].to_numpy(dtype=float),
        "readout_conf_pre": events["pre_confidence"].to_numpy(dtype=float),
        "verbalized_post": events["post_verbalized"].to_numpy(dtype=float) / 100.0,
        "verbalized_pre": events["pre_verbalized"].to_numpy(dtype=float) / 100.0,
    }


def metadata_oof(
    events: pd.DataFrame, y: np.ndarray, groups: np.ndarray, C: float = 1.0
) -> np.ndarray:
    X = metadata_features(events).to_numpy(dtype=float)
    return grouped_oof(X, np.asarray(y).astype(int), np.asarray(groups), C=C)


def incremental_auroc(
    y: np.ndarray,
    probe_score: np.ndarray,
    baseline_score: np.ndarray,
    combined_score: np.ndarray | None = None,
) -> dict[str, float]:
    """AUROC of probe, baseline, and (optionally) the two combined."""
    out = {"probe": auroc(y, probe_score), "baseline": auroc(y, baseline_score)}
    out["delta"] = out["probe"] - out["baseline"]
    if combined_score is not None:
        out["combined"] = auroc(y, combined_score)
        out["incremental"] = out["combined"] - out["baseline"]
    return out


def combined_oof(
    events: pd.DataFrame, probe_score: np.ndarray, y: np.ndarray, groups: np.ndarray, C: float = 1.0
) -> np.ndarray:
    """Metadata features plus the probe score, so incremental value can be tested directly."""
    X = metadata_features(events).to_numpy(dtype=float)
    X = np.hstack([X, np.asarray(probe_score).reshape(-1, 1)])
    return grouped_oof(X, np.asarray(y).astype(int), np.asarray(groups), C=C)


def permutation_null(
    fit_score_fn, y: np.ndarray, groups: np.ndarray, n_perm: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Shuffle labels WITHIN nothing -- across all rows -- and refit. AUROC should sit at 0.5.

    ``fit_score_fn(y_shuffled) -> scores`` must retrain the probe on the permuted labels.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    rows = []
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        scores = fit_score_fn(y_perm)
        mask = ~np.isnan(scores)
        rows.append({"perm": i, "auroc": auroc(y_perm[mask], scores[mask])})
    return pd.DataFrame(rows)


def permutation_p_value(observed: float, null_aurocs: np.ndarray) -> float:
    null = np.asarray(null_aurocs)
    null = null[~np.isnan(null)]
    if len(null) == 0:
        return float("nan")
    return float((1 + (null >= observed).sum()) / (1 + len(null)))
