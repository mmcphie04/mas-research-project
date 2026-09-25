"""Exploratory reanalysis: all events, repeated question-grouped splits, fair baselines.

    uv run python scripts/reanalysis.py

Differences from the earlier analysis, all of them corrections:
  * every question x agent event, not only the ~38% that changed answer. Correctness is defined
    whether or not the agent revised.
  * repeated outer splits grouped by question, reporting the distribution rather than one number.
  * layer and regularisation chosen inside training data only, on every repetition.
  * [pre, post] as the primary paired form. [pre, post-pre] is the same information reparameterised.
  * the output-only baseline gets the same supervised fit, splits and tuning as the probe.
  * a CONSTANT base-rate predictor is included, because a predictor that emits the base rate for
    every event has near-perfect calibration and zero ability to rank. Low ECE alone is not signal.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mad.analysis.calibrate import expected_calibration_error
from mad.analysis.labels import apply_policy, build_events
from mad.analysis.oracles import evaluate, policy_accept
from mad.analysis.stats import auroc
from mad.debate import ROWS_FILE
from mad.store import ActivationStore

RUN = Path("runs/night2-debate")
N_REPS, C_GRID = 8, (1e-4, 1e-3, 1e-2, 1e-1)
OUT = Path("docs/reanalysis_results.json")


def pipe(C):
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=C, max_iter=3000, solver="lbfgs"))])


def inner_pick(X, y, g, layers_fn, layer_choices):
    """Pick layer then C, both with grouped CV inside the training rows only."""
    gk = GroupKFold(n_splits=3)
    best_L, best = None, -1
    for L in layer_choices:
        XL = layers_fn(L)
        s = []
        for tr, te in gk.split(XL, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            s.append(auroc(y[te], pipe(1e-2).fit(XL[tr], y[tr]).predict_proba(XL[te])[:, 1]))
        m = np.nanmean(s) if s else np.nan
        if np.isfinite(m) and m > best:
            best, best_L = m, L
    XL = layers_fn(best_L)
    best_C, bc = None, -1
    for C in C_GRID:
        s = []
        for tr, te in gk.split(XL, y, g):
            if len(np.unique(y[tr])) < 2:
                continue
            s.append(auroc(y[te], pipe(C).fit(XL[tr], y[tr]).predict_proba(XL[te])[:, 1]))
        m = np.nanmean(s) if s else np.nan
        if np.isfinite(m) and m > bc:
            bc, best_C = m, C
    return best_L, best_C, best


def metrics(y, p):
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"auroc": auroc(y, p), "brier": brier_score_loss(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "ece": expected_calibration_error(p, y), "sd_pred": float(np.std(p))}


def main():
    ev = build_events(pd.read_parquet(RUN / ROWS_FILE)).reset_index(drop=True)
    store = ActivationStore(RUN)
    idx = store.index.reset_index(drop=True)
    pos = {(r.question_id, r.agent_id, r.phase): i for i, r in idx.iterrows()}
    ipre = np.array([pos[(e.question_id, e.agent_id, "pre")] for e in ev.itertuples()])
    ipost = np.array([pos[(e.question_id, e.agent_id, "post")] for e in ev.itertuples()])
    cache = {}

    def feats(rep, L):
        if L not in cache:
            f = store.load_layer(L)
            cache[L] = (f[ipre], f[ipost])
        pre, post = cache[L]
        return {"pre": pre, "post": post, "pre_post": np.hstack([pre, post]),
                "pre_diff": np.hstack([pre, post - pre])}[rep]

    LAYERS = [int(store.n_layers * f) for f in (0.25, 0.5, 0.65, 0.8, 0.92, 1.0)]
    out_cols = [c for c in ("pre_confidence", "post_confidence", "pre_rationale_chars",
                            "post_rationale_chars", "n_options", "n_peers_agree_pre",
                            "peer_unanimous", "post_equals_peer_majority", "pre_answer",
                            "post_answer", "revised") if c in ev.columns]
    Xout = ev[out_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)

    full = RUN / "readout_full.parquet"
    if full.exists():
        fp = pd.read_parquet(full)
        w = fp.pivot_table(index=["question_id", "agent_id"], columns="phase",
                           values=["top", "margin", "entropy"]).reset_index()
        w.columns = ["_".join(c).strip("_") for c in w.columns]
        ev2 = ev.merge(w, on=["question_id", "agent_id"], how="left")
        extra = [c for c in ev2.columns if c.split("_")[0] in ("top", "margin", "entropy")]
        Xout = np.hstack([Xout, ev2[extra].fillna(0).to_numpy(dtype=float)])
        print(f"output baseline enriched with {len(extra)} recovered distribution features")
    else:
        print("NOTE: readout_full.parquet absent; output baseline is top-probability + metadata only")

    qids = ev.question_id.to_numpy()
    y_pre, y_post = ev.pre_correct.to_numpy(), ev.post_correct.to_numpy()
    rev = (ev.revised == 1).to_numpy()
    print(f"events {len(ev)}  revisions {rev.sum()}  questions {len(set(qids))}  "
          f"base rate y_post {y_post.mean():.3f}")

    rows, gates = [], []
    uq = np.array(sorted(set(qids)))
    for rep in range(N_REPS):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(uq)
        test_q = set(perm[: int(0.3 * len(uq))])
        te = np.isin(qids, list(test_q))
        tr = ~te
        probs = {}
        for target, yv in (("y_pre", y_pre), ("y_post", y_post)):
            for rep_name in ("pre", "post", "pre_post", "pre_diff"):
                L, C, vs = inner_pick(None, yv[tr], qids[tr],
                                      lambda L, r=rep_name, m=tr: feats(r, L)[m], LAYERS)
                X = feats(rep_name, L)
                p = pipe(C).fit(X[tr], yv[tr]).predict_proba(X[te])[:, 1]
                probs[(target, rep_name)] = p
                m = metrics(yv[te], p)
                rows.append({"rep": rep, "family": "activation", "target": target,
                             "representation": rep_name, "layer": L, "C": C,
                             "inner_auroc": vs, "subset": "all", **m})
                for sub, mask in (("revised", rev[te]),
                                  ("init_correct", y_pre[te] == 1),
                                  ("init_incorrect", y_pre[te] == 0)):
                    if mask.sum() > 20 and len(np.unique(yv[te][mask])) > 1:
                        rows.append({"rep": rep, "family": "activation", "target": target,
                                     "representation": rep_name, "layer": L, "C": C,
                                     "inner_auroc": vs, "subset": sub,
                                     **metrics(yv[te][mask], p[mask])})
            # output-only and constant, same splits and fitting opportunity
            bo, bc = -1, None
            for C in C_GRID:
                s = [auroc(yv[tr][b], pipe(C).fit(Xout[tr][a], yv[tr][a]).predict_proba(Xout[tr][b])[:, 1])
                     for a, b in GroupKFold(n_splits=3).split(Xout[tr], yv[tr], qids[tr])]
                if np.nanmean(s) > bo:
                    bo, bc = np.nanmean(s), C
            po = pipe(bc).fit(Xout[tr], yv[tr]).predict_proba(Xout[te])[:, 1]
            probs[(target, "output")] = po
            rows.append({"rep": rep, "family": "output_only", "target": target,
                         "representation": "output", "layer": -1, "C": bc,
                         "inner_auroc": bo, "subset": "all", **metrics(yv[te], po)})
            const = np.full(te.sum(), yv[tr].mean())
            rows.append({"rep": rep, "family": "constant", "target": target,
                         "representation": "base_rate", "layer": -1, "C": np.nan,
                         "inner_auroc": np.nan, "subset": "all", **metrics(yv[te], const)})
            Xc = np.hstack([feats("pre_post", LAYERS[-2]), Xout])
            pc = pipe(1e-3).fit(Xc[tr], yv[tr]).predict_proba(Xc[te])[:, 1]
            probs[(target, "combined")] = pc
            rows.append({"rep": rep, "family": "combined", "target": target,
                         "representation": "act+output", "layer": LAYERS[-2], "C": 1e-3,
                         "inner_auroc": np.nan, "subset": "all", **metrics(yv[te], pc)})

        # ---- the actual decision: keep the original answer or take the proposed one
        sub = ev[te].reset_index(drop=True)
        for src in ("pre_post", "output", "combined"):
            gp, gn = probs[("y_pre", src)], probs[("y_post", src)]
            best = max((np.mean(np.where(np.asarray(gn - gp > t) | ~rev[te],
                                          y_post[te], y_pre[te])), t)
                        for t in np.linspace(-0.3, 0.3, 25))
            acc = np.asarray(gn - gp > best[1]) | ~rev[te]
            panel = apply_policy(sub, acc)["panel_correct"].mean()
            gates.append({"rep": rep, "source": src, "tau": best[1],
                          "agent_acc": float(best[0]), "panel_acc": float(panel)})
        for name, pol in (("accept_all", "mad_standard"), ("reject_all", "round0_plurality"),
                          ("oracle", "oracle_full")):
            a = policy_accept(sub, pol)
            gates.append({"rep": rep, "source": name, "tau": np.nan,
                          "agent_acc": float(np.mean(np.where(a, y_post[te], y_pre[te]))),
                          "panel_acc": float(evaluate(sub, a)["panel_accuracy"])})
        print(f"  rep {rep} done", flush=True)

    df, gd = pd.DataFrame(rows), pd.DataFrame(gates)
    df.to_csv("docs/reanalysis_events.csv", index=False)
    gd.to_csv("docs/reanalysis_gating.csv", index=False)

    def show(d, keys, cols=("auroc", "brier", "ece", "sd_pred")):
        agg = d.groupby(list(keys))[list(cols)].agg(["mean", "std"]).round(4)
        print(agg.to_string())

    print("\n=== ALL EVENTS, target = proposed answer correct ===")
    show(df[(df.target == "y_post") & (df.subset == "all")], ["family", "representation"])
    print("\n=== by subset (activation, pre_post) ===")
    show(df[(df.family == "activation") & (df.representation == "pre_post") & (df.target == "y_post")],
         ["subset"])
    print("\n=== OLD vs NEW gating (mean over reps) ===")
    print(gd.groupby("source")[["agent_acc", "panel_acc"]].agg(["mean", "std"]).round(4).to_string())
    OUT.write_text(json.dumps({"n_events": int(len(ev)), "n_revisions": int(rev.sum()),
                               "n_questions": int(len(uq)), "n_reps": N_REPS,
                               "layers_considered": LAYERS}, indent=2))
    print(f"\nwrote docs/reanalysis_events.csv, docs/reanalysis_gating.csv, {OUT}")


if __name__ == "__main__":
    main()
