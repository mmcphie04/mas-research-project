"""Does the PRE-REVISION state predict what the agent is about to do?

    uv run python scripts/peer_exposure_analysis.py

Three standardised states per agent:
  pre            after its original response
  peer_exposure  after it has read the peers, BEFORE it generates a revision
  post           after it has generated the revision

The post state only exists once the answer is decided, so a probe on it may be reading a
decision rather than predicting one. peer_exposure is the state where a prediction is still a
prediction. The sharpest form of the question is on initially-correct agents: can the activation,
before any revision text exists, tell that this agent is about to abandon a correct answer?

Same discipline as the reanalysis: repeated question-grouped splits, layer and C chosen inside
training rows only, a fairly-fitted output baseline, and a constant base-rate control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mad.analysis.calibrate import expected_calibration_error
from mad.analysis.stats import auroc
from mad.debate import ROWS_FILE
from mad.store import ActivationStore

RUN = "runs/night3-debate"
N_REPS, C_GRID = 8, (1e-4, 1e-3, 1e-2, 1e-1)


def pipe(C):
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, max_iter=3000))])


def pick(Xf, y, g, layers):
    gk = GroupKFold(n_splits=3)
    bl, bs = None, -1
    for L in layers:
        X = Xf(L)
        s = [auroc(y[b], pipe(1e-2).fit(X[a], y[a]).predict_proba(X[b])[:, 1])
             for a, b in gk.split(X, y, g) if len(np.unique(y[a])) > 1]
        if s and np.nanmean(s) > bs:
            bs, bl = np.nanmean(s), L
    X = Xf(bl)
    bC, bc = None, -1
    for C in C_GRID:
        s = [auroc(y[b], pipe(C).fit(X[a], y[a]).predict_proba(X[b])[:, 1])
             for a, b in gk.split(X, y, g) if len(np.unique(y[a])) > 1]
        if s and np.nanmean(s) > bc:
            bc, bC = np.nanmean(s), C
    return bl, bC


def main():
    rows = pd.read_parquet(f"{RUN}/{ROWS_FILE}")
    store = ActivationStore(RUN)
    idx = store.index.reset_index(drop=True)
    pos = {(r.question_id, r.agent_id, r.phase): i for i, r in idx.iterrows()}

    w = rows.pivot_table(index=["question_id", "agent_id"], columns="phase",
                         values=["readout_answer", "correct", "readout_confidence"],
                         aggfunc="first").reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    w["revised"] = (w.readout_answer_pre != w.readout_answer_post).astype(int)
    w["switched_at_exposure"] = (w.readout_answer_pre != w.readout_answer_peer_exposure).astype(int)

    ii = {p: np.array([pos[(r.question_id, r.agent_id, p)] for r in w.itertuples()])
          for p in ("pre", "peer_exposure", "post")}
    cache = {}

    def F(name, L):
        if L not in cache:
            f = store.load_layer(L)
            cache[L] = {p: f[i] for p, i in ii.items()}
        c = cache[L]
        return {"pre": c["pre"], "exposure": c["peer_exposure"], "post": c["post"],
                "pre+exposure": np.hstack([c["pre"], c["peer_exposure"]]),
                "pre+post": np.hstack([c["pre"], c["post"]]),
                "all_three": np.hstack([c["pre"], c["peer_exposure"], c["post"]])}[name]

    LAYERS = [int(store.n_layers * f) for f in (0.25, 0.5, 0.65, 0.8, 0.92, 1.0)]
    y = w.correct_post.to_numpy().astype(int)
    g = w.question_id.to_numpy()
    rev = w.revised.to_numpy() == 1
    ic = w.correct_pre.to_numpy() == 1
    Xout = w[["readout_confidence_pre", "readout_confidence_peer_exposure",
              "readout_answer_pre", "readout_answer_peer_exposure",
              "switched_at_exposure"]].to_numpy(dtype=float)

    print(f"events {len(w)}  questions {w.question_id.nunique()}  revisions {rev.sum()}  "
          f"initially correct {ic.sum()}  base rate y_post {y.mean():.3f}")
    print(f"agent already shows a different answer at peer_exposure: {w.switched_at_exposure.mean():.3f}")
    print(f"  of those, {(w[w.switched_at_exposure==1].revised==1).mean():.3f} go on to actually revise")
    print(f"  of the rest, {(w[w.switched_at_exposure==0].revised==1).mean():.3f} revise anyway\n")

    uq = np.array(sorted(set(g)))
    out = []
    for rep in range(N_REPS):
        rng = np.random.default_rng(500 + rep)
        te = np.isin(g, rng.permutation(uq)[: int(0.3 * len(uq))])
        tr = ~te
        for name in ("pre", "exposure", "post", "pre+exposure", "pre+post", "all_three"):
            L, C = pick(lambda L, n=name, m=tr: F(n, L)[m], y[tr], g[tr], LAYERS)
            X = F(name, L)
            p = pipe(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            for sub, m in (("all", np.ones(te.sum(), bool)), ("revised", rev[te]),
                           ("init_correct", ic[te])):
                if m.sum() > 15 and len(np.unique(y[te][m])) > 1:
                    out.append({"rep": rep, "family": "activation", "representation": name,
                                "subset": sub, "layer": L,
                                "auroc": auroc(y[te][m], p[m]),
                                "brier": brier_score_loss(y[te][m], p[m]),
                                "ece": expected_calibration_error(p[m], y[te][m])})
        po = pipe(1e-2).fit(Xout[tr], y[tr]).predict_proba(Xout[te])[:, 1]
        const = np.full(te.sum(), y[tr].mean())
        for fam, p in (("output_only", po), ("constant", const)):
            for sub, m in (("all", np.ones(te.sum(), bool)), ("revised", rev[te]),
                           ("init_correct", ic[te])):
                if m.sum() > 15 and len(np.unique(y[te][m])) > 1:
                    out.append({"rep": rep, "family": fam, "representation": fam, "subset": sub,
                                "layer": -1, "auroc": auroc(y[te][m], p[m]),
                                "brier": brier_score_loss(y[te][m], p[m]),
                                "ece": expected_calibration_error(p[m], y[te][m])})
        print(f"  rep {rep} done", flush=True)

    df = pd.DataFrame(out)
    df.to_csv("docs/peer_exposure_results.csv", index=False)
    for sub in ("all", "revised", "init_correct"):
        print(f"\n=== subset: {sub} ===")
        print(df[df.subset == sub].groupby(["family", "representation"])[["auroc", "brier", "ece"]]
              .agg(["mean", "std"]).round(4).to_string())
    print("\nwrote docs/peer_exposure_results.csv")


if __name__ == "__main__":
    main()
