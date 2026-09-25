"""Stage 3+4 on real debate data: can a probe predict revision quality?

    uv run python scripts/analyze_debate.py runs/night-debate

Everything here is offline. Because there is exactly one debate round, the gate only ever picks
between two answers that were already generated, so no model is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.baselines import metadata_oof
from mad.analysis.gating import apply_gate, select_threshold
from mad.analysis.labels import build_events
from mad.analysis.oracles import evaluate, policy_accept, random_accept
from mad.analysis.probes import FeatureBank, fit_probe
from mad.analysis.stats import auroc, bootstrap_delta_auroc, conditional_auroc, mcnemar
from mad.debate import ROWS_FILE
from mad.store import ActivationStore

SWEEP_C = (0.001, 0.1)


def main(run_dir: str) -> None:
    run_dir = Path(run_dir)
    rows = pd.read_parquet(run_dir / ROWS_FILE)
    ev = build_events(rows).reset_index(drop=True)
    store = ActivationStore(run_dir)
    bank = FeatureBank(store, ev)

    rev = (ev["revised"] == 1).to_numpy()
    g_all = ev["question_id"].to_numpy()
    print(f"events {len(ev)}   revisions {rev.sum()}   "
          f"B={int((ev.label=='beneficial').sum())} H={int((ev.label=='harmful').sum())} "
          f"U={int((ev.label=='unhelpful').sum())}")

    # ---- layer sweep on the deployment population: revision events only, target = post correct
    print("\nLAYER SWEEP  (revision events only, target: is the NEW answer correct)")
    y_rev = ev.loc[rev, "post_correct"].to_numpy()
    g_rev = g_all[rev]
    best = (-1, None)
    for layer in store.layer_range():
        if layer % 2 and layer != store.n_layers:
            continue
        X = bank.features(layer, "post")[rev]
        pr = fit_probe(X, y_rev, g_rev, layer=layer, C_grid=SWEEP_C)
        print(f"  layer {layer:>2}  cv_auroc {pr.cv_auroc:.4f}")
        if pr.cv_auroc > best[0]:
            best = (pr.cv_auroc, layer)
    L = best[1]
    print(f"  -> best layer {L}")

    # ---- representation x target grid at the winning layer
    print(f"\nREPRESENTATION x TARGET GRID  (layer {L}, revision events, grouped CV)")
    targets = {"y_pre (was old answer right)": ev.loc[rev, "pre_correct"].to_numpy(),
               "y_post (is new answer right)": y_rev}
    scores = {}
    for repname in ("pre", "post", "diff", "paired"):
        X = bank.features(L, repname)[rev]
        line = f"  {repname:<7}"
        for tname, y in targets.items():
            pr = fit_probe(X, y, g_rev, layer=L, C_grid=SWEEP_C)
            line += f"   {tname.split()[0]}={pr.cv_auroc:.4f}"
            if tname.startswith("y_post"):
                scores[repname] = pr.oof
        print(line)

    # ---- baselines on the same rows
    print("\nBASELINES vs PROBE  (target: is the NEW answer correct, revision events only)")
    conf = ev.loc[rev, "post_confidence"].to_numpy()
    meta = metadata_oof(ev[rev], y_rev, g_rev)
    probe = scores["post"]
    ok = ~np.isnan(probe) & ~np.isnan(meta)
    for name, sc in (("readout confidence", conf), ("debate metadata", meta),
                     ("probe: post state", probe), ("probe: paired", scores["paired"])):
        s = np.asarray(sc)
        m = ok & ~np.isnan(s)
        c, npair, ngrp = conditional_auroc(y_rev[m], s[m], g_rev[m])
        print(f"  {name:<20} AUROC {auroc(y_rev[m], s[m]):.4f}   conditional {c:.4f} ({npair} pairs)")

    for label, other in (("confidence", conf), ("metadata", meta)):
        d = bootstrap_delta_auroc(y_rev[ok], probe[ok], np.asarray(other)[ok], g_rev[ok], n_boot=600)
        verdict = "PROBE WINS" if d.lo > 0 else ("inconclusive" if d.hi > 0 else "BASELINE WINS")
        print(f"  probe - {label:<11} {d}  {verdict}")

    # ---- does gating on the probe beat ungated debate?
    print("\nGATING  (tau tuned on a validation half, evaluated on the held-out half)")
    qs = ev["question_id"].unique()
    val_ids = set(qs[: len(qs) // 2])
    full = np.full(len(ev), np.nan)
    full[np.flatnonzero(rev)] = probe
    ev = ev.assign(_score=full)
    va, te = ev[ev.question_id.isin(val_ids)], ev[~ev.question_id.isin(val_ids)]

    for name, sc_col in (("probe", "_score"), ("readout confidence", "post_confidence")):
        sv = va[sc_col].fillna(1.0).to_numpy()
        st = te[sc_col].fillna(1.0).to_numpy()
        tau, _ = select_threshold(va, None, sv, rule="single")
        acc = apply_gate(te, None, st, tau, rule="single")
        gated = evaluate(te, acc, f"gated-{name}")
        mad = evaluate(te, policy_accept(te, "mad_standard"), "mad")
        from mad.analysis.labels import apply_policy
        a = apply_policy(te, acc).sort_values("question_id")
        b = apply_policy(te, policy_accept(te, "mad_standard")).sort_values("question_id")
        t = mcnemar(a.panel_correct.to_numpy(), b.panel_correct.to_numpy())
        print(f"  {name:<19} tau={tau:.3f}  panel {gated['panel_accuracy']:.4f} vs MAD "
              f"{mad['panel_accuracy']:.4f}  ({t.delta*100:+.2f} pts, {t.n_discordant} disc, p={t.p_value:.3f})")

    rnd = random_accept(te, rate=0.5, seed=0)
    print(f"  {'random gate @50%':<19}            panel "
          f"{evaluate(te, rnd, 'rand')['panel_accuracy']:.4f}")
    print(f"  {'oracle (ceiling)':<19}            panel "
          f"{evaluate(te, policy_accept(te, 'oracle_full'), 'orc')['panel_accuracy']:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/night-debate")
