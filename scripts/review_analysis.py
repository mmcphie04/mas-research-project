"""Analysis addressing the external review: honest layer selection, symmetric calibration,
richer output-only baselines, and non-redundant paired features.

    uv run python scripts/review_analysis.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.calibrate import PlattScaler, expected_calibration_error
from mad.analysis.labels import build_events
from mad.analysis.probes import FeatureBank, fit_probe, grouped_oof
from mad.analysis.stats import auroc, bootstrap_delta_auroc, conditional_auroc
from mad.debate import ROWS_FILE
from mad.store import ActivationStore
from scripts.probe_real import load_run  # noqa

C_GRID = (0.001, 0.1)


def split_by_question(qids, seed=11, fracs=(0.5, 0.2, 0.3)):
    uq = np.array(sorted(set(qids)))
    rng = np.random.default_rng(seed); rng.shuffle(uq)
    a, b = int(fracs[0] * len(uq)), int((fracs[0] + fracs[1]) * len(uq))
    m = {q: ("train" if i < a else "val" if i < b else "test") for i, q in enumerate(uq)}
    return np.array([m[q] for q in qids])


print("=" * 84)
print("A. DID THE DOUBLE-SUFFIX BUG MATTER?  (Milestone A, same questions, same model)")
print("=" * 84)
for tag, run in (("buggy  (suffix twice)", "runs/probe-arcc"), ("fixed  (suffix once)", "runs/probe-arcc-fixed")):
    rows, store, order = load_run(Path(run))
    y, g = rows.correct.to_numpy(), rows.question_id.to_numpy()
    conf = rows.readout_confidence.to_numpy()
    L = int(store.n_layers * 0.92)
    pr = fit_probe(store.load_layer(L)[order], y, g, layer=L, C_grid=C_GRID)
    ok = ~np.isnan(pr.oof)
    d = bootstrap_delta_auroc(y[ok], pr.oof[ok], conf[ok], g[ok], n_boot=400)
    print(f"  {tag:<24} n={len(y)} acc={y.mean():.3f}  probe {pr.cv_auroc:.4f}  "
          f"conf {auroc(y, conf):.4f}  delta {d.point:+.4f} [{d.lo:+.4f},{d.hi:+.4f}]")

print("\n" + "=" * 84)
print("B. DEBATE PROBE with honest layer selection (train+val pick the layer, test scores once)")
print("=" * 84)
ev = build_events(pd.read_parquet(Path("runs/night2-debate") / ROWS_FILE)).reset_index(drop=True)
store = ActivationStore("runs/night2-debate")
bank = FeatureBank(store, ev)
rev = (ev.revised == 1).to_numpy()
sp = split_by_question(ev.question_id.to_numpy())
y = ev.post_correct.to_numpy(); g = ev.question_id.to_numpy()
tr = rev & np.isin(sp, ["train", "val"]); te = rev & (sp == "test")
print(f"  revision events: {tr.sum()} train+val, {te.sum()} test")

best = (-1, None)
for L in range(0, store.n_layers + 1, 2):
    pr = fit_probe(bank.features(L, "post")[tr], y[tr], g[tr], layer=L, C_grid=C_GRID)
    if pr.cv_auroc > best[0]:
        best = (pr.cv_auroc, L)
L = best[1]
print(f"  layer chosen on train+val only: {L}  (val AUROC {best[0]:.4f})")

print("\n  paired-representation variants, scored ONCE on held-out test:")
res = {}
for rep in ("pre", "post", "diff", "pre_post", "pre_diff", "paired"):
    if rep == "pre_post":
        Xf = np.hstack([bank.features(L, "pre"), bank.features(L, "post")])
    elif rep == "pre_diff":
        Xf = np.hstack([bank.features(L, "pre"), bank.features(L, "diff")])
    else:
        Xf = bank.features(L, rep)
    m = fit_probe(Xf[tr], y[tr], g[tr], layer=L, C_grid=C_GRID)
    s = m.score(Xf[te]); res[rep] = s
    c, npair, _ = conditional_auroc(y[te], s, g[te])
    print(f"    {rep:<10} test AUROC {auroc(y[te], s):.4f}   conditional {c:.4f} ({npair} pairs)")

print("\n" + "=" * 84)
print("C. OUTPUT-ONLY BASELINE, given the same supervised fit as the probe")
print("=" * 84)
cols = [c for c in ("post_confidence", "pre_confidence", "post_rationale_chars", "n_options") if c in ev]
Xo = ev[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
oof_o = grouped_oof(Xo[tr], y[tr], g[tr], C=1.0)
mo = ~np.isnan(oof_o)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=1.0, max_iter=2000))]).fit(Xo[tr], y[tr])
s_out = pipe.predict_proba(Xo[te])[:, 1]
raw = ev.post_confidence.to_numpy()
print(f"  raw top-probability          test AUROC {auroc(y[te], raw[te]):.4f}")
print(f"  supervised output-only       test AUROC {auroc(y[te], s_out):.4f}")
print(f"  probe (best paired variant)  test AUROC {auroc(y[te], res['pre_diff']):.4f}")
d = bootstrap_delta_auroc(y[te], res["pre_diff"], s_out, g[te], n_boot=600)
print(f"  probe - supervised output-only  {d}")

print("\n  calibration, both fitted on train+val and measured on test:")
pl_p = PlattScaler().fit(res["pre_diff"][: 0] if False else
                        fit_probe(np.hstack([bank.features(L, "pre"), bank.features(L, "diff")])[tr],
                                  y[tr], g[tr], layer=L, C_grid=C_GRID).oof[~np.isnan(
                                  fit_probe(np.hstack([bank.features(L, "pre"), bank.features(L, "diff")])[tr],
                                            y[tr], g[tr], layer=L, C_grid=C_GRID).oof)],
                        y[tr][~np.isnan(fit_probe(np.hstack([bank.features(L, "pre"), bank.features(L, "diff")])[tr],
                                                  y[tr], g[tr], layer=L, C_grid=C_GRID).oof)])
pl_o = PlattScaler().fit(oof_o[mo], y[tr][mo])
print(f"    probe  ECE {expected_calibration_error(pl_p.transform(res['pre_diff']), y[te]):.4f}")
print(f"    output ECE {expected_calibration_error(pl_o.transform(s_out), y[te]):.4f}")
print(f"    raw top-probability ECE {expected_calibration_error(raw[te], y[te]):.4f}")
