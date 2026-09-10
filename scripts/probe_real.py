"""Train a correctness probe on REAL activations and compare it to logit confidence.

    uv run python scripts/probe_real.py runs/probe-arcc

This is Milestone A: the ground floor of the project. If a linear probe on real activations
cannot beat the model's own forced-choice logit confidence at predicting whether its answer is
correct, then revision gating and confidence-weighted voting are both dead on arrival.

Every AUROC is reported twice: MARGINAL (pooled) and CONDITIONAL (within-question only). The gap
between them is the difficulty-detector diagnostic -- a probe that has merely learned which items
are hard scores well marginally and at chance conditionally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.calibrate import PlattScaler, brier, expected_calibration_error
from mad.analysis.probes import DEFAULT_C_GRID, fit_probe, grouped_oof, mass_mean_probe
from mad.analysis.stats import (
    auroc,
    bootstrap_auroc,
    bootstrap_delta_auroc,
    conditional_auroc,
)
from mad.collect import STATES_FILE
from mad.provenance import require_complete
from mad.store import ActivationStore

KEY = ["run_id", "condition", "question_id", "agent_id", "phase"]


def load_run(run_dir: Path):
    require_complete(run_dir)
    rows = pd.read_parquet(run_dir / STATES_FILE)
    store = ActivationStore(run_dir)
    # align activation index order to the state table explicitly rather than trusting insertion order
    idx = store.index.reset_index(drop=True)
    pos = {tuple(r[k] for k in KEY): i for i, r in idx.iterrows()}
    order = np.array([pos[tuple(r[k] for k in KEY)] for _, r in rows.iterrows()])
    return rows, store, order


def report_pair(name, y, score, groups, seed=0):
    marg = bootstrap_auroc(y, score, groups, n_boot=500, seed=seed)
    cond, n_pairs, n_groups = conditional_auroc(y, score, groups)
    print(f"  {name:<26} marginal {marg}   conditional {cond:.4f}  ({n_pairs} pairs / {n_groups} questions)")
    return marg.point, cond


def main(run_dir: str):
    run_dir = Path(run_dir)
    rows, store, order = load_run(run_dir)
    y = rows["correct"].to_numpy()
    groups = rows["question_id"].to_numpy()
    conf = rows["readout_confidence"].to_numpy()

    print("=" * 92)
    print(f"REAL-ACTIVATION PROBE   {run_dir}")
    print("=" * 92)
    print(f"states {len(rows)}   questions {rows.question_id.nunique()}   "
          f"variants/question {len(rows) / rows.question_id.nunique():.1f}")
    print(f"accuracy {y.mean():.4f}   layers {store.n_layers}   d_model {store.d_model}   "
          f"activations {len(store)}")

    # how much does option order alone move the answer?
    per_q = rows.groupby("question_id")["correct"].nunique()
    print(f"\nquestions whose correctness FLIPS across option orderings: "
          f"{(per_q > 1).mean():.3f}  <- within-question signal available to the probe")
    by_pos = rows.groupby("gold_display")["correct"].agg(["mean", "size"])
    print("accuracy by gold option position:",
          "  ".join(f"{i}:{r['mean']:.3f}(n={int(r['size'])})" for i, r in by_pos.iterrows()))

    print("\n" + "-" * 92)
    print("LAYER SWEEP (grouped 5-fold CV by question)")
    print("-" * 92)
    sweep = []
    for layer in store.layer_range():
        X = store.load_layer(layer)[order]
        pr = fit_probe(X, y, groups, layer=layer, C_grid=DEFAULT_C_GRID)
        ok = ~np.isnan(pr.oof)
        cond, _, _ = conditional_auroc(y[ok], pr.oof[ok], groups[ok])
        sweep.append({"layer": layer, "cv_auroc": pr.cv_auroc, "cond_auroc": cond, "C": pr.best_C})
    sweep = pd.DataFrame(sweep)
    print(sweep.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    best_layer = int(sweep.loc[sweep["cv_auroc"].idxmax(), "layer"])
    print(f"\nbest layer {best_layer} of {store.n_layers} "
          f"({best_layer / store.n_layers:.0%} depth)")

    X = store.load_layer(best_layer)[order]
    probe = fit_probe(X, y, groups, layer=best_layer)
    ok = ~np.isnan(probe.oof)

    print("\n" + "-" * 92)
    print("PROBE vs BASELINES  (out-of-fold; CIs are cluster bootstrap over questions)")
    print("-" * 92)
    report_pair("logit confidence", y, conf, groups)
    report_pair(f"linear probe (L{best_layer})", y[ok], probe.oof[ok], groups[ok])

    mm_dir = mass_mean_probe(X, y)
    report_pair("mass-mean probe", y, X @ mm_dir, groups)

    both = np.column_stack([conf, probe.oof])
    both_oof = grouped_oof(both[ok], y[ok], groups[ok], C=1.0)
    ok2 = ~np.isnan(both_oof)
    report_pair("probe + confidence", y[ok][ok2], both_oof[ok2], groups[ok][ok2])

    print("\n" + "-" * 92)
    print("HEAD TO HEAD: does the probe add anything over logit confidence?")
    print("-" * 92)
    d = bootstrap_delta_auroc(y[ok], probe.oof[ok], conf[ok], groups[ok], n_boot=1000)
    print(f"  delta AUROC (probe - confidence): {d}")
    print(f"  {'PROBE WINS' if d.lo > 0 else 'NOT SIGNIFICANT' if d.hi > 0 else 'CONFIDENCE WINS'}"
          "  (95% cluster-bootstrap CI excludes 0?)")

    print("\n" + "-" * 92)
    print("CONTROLS")
    print("-" * 92)
    rng = np.random.default_rng(0)
    perm_scores = []
    for _ in range(5):
        y_perm = rng.permutation(y)
        oof_p = grouped_oof(X, y_perm, groups, C=probe.best_C)
        m = ~np.isnan(oof_p)
        perm_scores.append(auroc(y_perm[m], oof_p[m]))
    print(f"  label-permutation AUROC over 5 shuffles: "
          f"{np.mean(perm_scores):.4f} +/- {np.std(perm_scores):.4f}   (expect ~0.5)")

    plat = PlattScaler().fit(probe.oof[ok], y[ok])
    p_cal = plat.transform(probe.oof[ok])
    print(f"  probe ECE {expected_calibration_error(probe.oof[ok], y[ok]):.4f} -> "
          f"{expected_calibration_error(p_cal, y[ok]):.4f} after Platt; Brier {brier(p_cal, y[ok]):.4f}")
    print(f"  confidence ECE {expected_calibration_error(conf, y):.4f}")

    sweep.to_csv(run_dir / "layer_sweep.csv", index=False)
    print(f"\nwrote {run_dir / 'layer_sweep.csv'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/probe-arcc")
