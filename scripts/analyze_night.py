"""Control, cross-dataset transfer, and model-scale analyses from the overnight run."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.labels import build_events, label_counts, revision_rates
from mad.analysis.probes import fit_probe
from mad.analysis.stats import auroc, bootstrap_delta_auroc, conditional_auroc
from mad.debate import ROWS_FILE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_real import load_run  # noqa: E402

print("=" * 78)
print("1. PEER ATTRIBUTION  (debate vs self-reconsideration)")
print("=" * 78)
d = build_events(pd.read_parquet(Path("runs/night-debate") / ROWS_FILE))
c = build_events(pd.read_parquet(Path("runs/night-control") / ROWS_FILE))
for name, e in (("debate (sees peers)", d), ("control (no peers)", c)):
    r = revision_rates(e)
    lc = label_counts(e)
    print(
        f"  {name:<22} revision {r['revision_rate']:.3f}  r_C {r['r_C']:.3f}  r_I {r['r_I']:.3f}"
        f"   B={lc['beneficial']} H={lc['harmful']} U={lc['unhelpful']}"
    )
rd, rc = revision_rates(d)["revision_rate"], revision_rates(c)["revision_rate"]
print(
    f"\n  peer-attributable share of revisions: {(rd - rc) / rd:.1%}"
    f"   (baseline instability accounts for {rc / rd:.1%})"
)
ld, lc2 = label_counts(d), label_counts(c)
print(
    f"  harmful per 1200 events: debate {ld['harmful']}  control {lc2['harmful']}"
    f"   -> peers add {ld['harmful'] - lc2['harmful']}"
)
print(
    f"  beneficial per 1200 events: debate {ld['beneficial']}  control {lc2['beneficial']}"
    f"   -> peers add {ld['beneficial'] - lc2['beneficial']}"
)


def probe_vs_conf(run, label, layer_frac=0.85):
    rows, store, order = load_run(Path(run))
    y = rows.correct.to_numpy()
    g = rows.question_id.to_numpy()
    conf = rows.readout_confidence.to_numpy()
    L = int(store.n_layers * layer_frac)
    X = store.load_layer(L)[order]
    pr = fit_probe(X, y, g, layer=L, C_grid=(0.001, 0.1))
    ok = ~np.isnan(pr.oof)
    dlt = bootstrap_delta_auroc(y[ok], pr.oof[ok], conf[ok], g[ok], n_boot=500)
    cnd, _, _ = conditional_auroc(y[ok], pr.oof[ok], g[ok])
    print(
        f"  {label:<34} acc {y.mean():.3f}  probe {pr.cv_auroc:.4f}  conf {auroc(y, conf):.4f}"
        f"  delta {dlt.point:+.4f} [{dlt.lo:+.4f},{dlt.hi:+.4f}]  cond {cnd:.4f}"
    )
    return pr, store, order, rows, L


print("\n" + "=" * 78)
print("2. MODEL SCALE  (same questions, ARC-Challenge readout)")
print("=" * 78)
probe_vs_conf("runs/night-270m", "gemma-3-270m (18 layers, d=640)")
p1, s1, o1, r1, L1 = probe_vs_conf("runs/probe-arcc", "gemma-3-1b  (26 layers, d=1152)")

print("\n" + "=" * 78)
print("3. CROSS-DATASET TRANSFER  (train ARC-Challenge -> test ARC-Easy)")
print("=" * 78)
rows_e, store_e, order_e = load_run(Path("runs/night-arce"))
ye = rows_e.correct.to_numpy()
ge = rows_e.question_id.to_numpy()
Xe = store_e.load_layer(L1)[order_e]
if store_e.d_model != s1.d_model:
    print("  incompatible widths; skipped")
else:
    s_tr = p1.score(Xe)
    ce = rows_e.readout_confidence.to_numpy()
    cnd, _, _ = conditional_auroc(ye, s_tr, ge)
    dlt = bootstrap_delta_auroc(ye, s_tr, ce, ge, n_boot=500)
    print(
        f"  ARC-Easy accuracy {ye.mean():.3f}   (in-domain ARC-Challenge probe was {p1.cv_auroc:.4f})"
    )
    print(f"  transferred probe AUROC {auroc(ye, s_tr):.4f}   conditional {cnd:.4f}")
    print(f"  ARC-Easy confidence AUROC {auroc(ye, ce):.4f}")
    print(f"  transferred probe - confidence  {dlt.point:+.4f} [{dlt.lo:+.4f}, {dlt.hi:+.4f}]")
    print(f"  retention: {auroc(ye, s_tr) / p1.cv_auroc:.1%} of in-domain AUROC")
