"""Reanalysis of the 1B debate run with every choice made on training questions only.

    uv run python scripts/reanalysis.py --run runs/night2-debate --out artifacts/review/reanalysis

Corrects two problems in the previous version of this script:

1. **The gate threshold was chosen on the test labels.** The old code picked tau by maximising
   accuracy over the held-out rows' true labels, then reported that accuracy -- an optimistically
   biased number, documented at the time as "tuned on training questions", which it was not.
   Here tau is chosen on out-of-fold predictions for the TRAINING questions and then applied,
   unchanged, to the test questions.
2. **Repeated overlapping splits were described as replications.** Here the outer loop is grouped
   K-fold over questions, so every question is held out exactly once per repetition and the
   pooled held-out predictions support a paired bootstrap over questions. Extra repetitions are
   reported as stability, not as independent evidence.

The output-only baseline is the strong one: the full answer-choice distribution before and after
the revision, the probability of the OLD and the NEW candidate under each context, margins,
entropy, panel agreement, response lengths and categorical answer positions -- fitted both as a
regularised linear model and as one modest nonlinear model, with the same training-only
selection discipline as the activation probe.

Decision rule, for a revised answer only: accept iff P(new correct) - P(old correct) > tau.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis.calibrate import PlattScaler, expected_calibration_error  # noqa: E402
from mad.analysis.labels import apply_policy, build_events  # noqa: E402
from mad.analysis.oracles import panel_optimal_accept, policy_accept  # noqa: E402
from mad.analysis.stats import auroc, conditional_auroc  # noqa: E402
from mad.debate import ROWS_FILE  # noqa: E402
from mad.store import ActivationStore  # noqa: E402

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
TAU_GRID = np.round(np.linspace(-0.5, 0.5, 41), 4)
LAYER_FRACS = (0.25, 0.5, 0.65, 0.8, 0.92, 1.0)
HGB_GRID = ({"max_iter": 100, "max_depth": 3}, {"max_iter": 300, "max_depth": 3})


# ----------------------------------------------------------------------------- features


def _lp_matrix(full: pd.DataFrame, phase: str, keys: pd.DataFrame, width: int) -> np.ndarray:
    """(n_events, width) answer-choice log-probs for one phase, padded with -30."""
    f = full[full.phase == phase].set_index(["question_id", "agent_id"])["lp"]
    out = np.full((len(keys), width), -30.0)
    for i, (q, a) in enumerate(zip(keys.question_id, keys.agent_id, strict=True)):
        v = np.array([float(x) for x in f.loc[(q, a)].split(",")])
        out[i, : len(v)] = v
    return out


def output_features(ev: pd.DataFrame, full: pd.DataFrame | None) -> tuple[np.ndarray, list[str]]:
    """Everything observable WITHOUT internal activations, available at post-revision time."""
    cols: dict[str, np.ndarray] = {}
    width = int(ev.n_options.max())
    old = ev.pre_answer.to_numpy().astype(int)
    new = ev.post_answer.to_numpy().astype(int)
    if full is not None:
        lp_pre, lp_post = _lp_matrix(full, "pre", ev, width), _lp_matrix(full, "post", ev, width)
        idx = np.arange(len(ev))
        for ctx, lp in (("pre", lp_pre), ("post", lp_post)):
            p = np.exp(lp)
            srt = np.sort(lp, axis=1)
            cols[f"p_old_{ctx}"] = p[idx, old]
            cols[f"p_new_{ctx}"] = p[idx, new]
            cols[f"margin_{ctx}"] = srt[:, -1] - srt[:, -2]
            cols[f"entropy_{ctx}"] = -(p * np.clip(lp, -30, 0)).sum(axis=1)
            for j in range(width):
                cols[f"lp{j}_{ctx}"] = lp[:, j]
    else:
        cols["p_top_pre"] = ev.pre_confidence.to_numpy()
        cols["p_top_post"] = ev.post_confidence.to_numpy()
    # panel agreement among the OTHER agents' first answers -- observable to a later selector
    pre_tab = ev.pivot(index="question_id", columns="agent_id", values="pre_answer")
    agree_old, agree_new = [], []
    for q, a, o, n in zip(ev.question_id, ev.agent_id, old, new, strict=True):
        others = [pre_tab.loc[q, b] for b in pre_tab.columns if b != a]
        agree_old.append(sum(x == o for x in others))
        agree_new.append(sum(x == n for x in others))
    cols["panel_agree_old"] = np.array(agree_old, float)
    cols["panel_agree_new"] = np.array(agree_new, float)
    cols["revised"] = ev.revised.to_numpy(float)
    cols["len_pre"] = ev.pre_rationale_chars.to_numpy(float)
    cols["len_post"] = ev.post_rationale_chars.to_numpy(float)
    for j in range(width):  # categorical answer positions
        cols[f"old_is_{j}"] = (old == j).astype(float)
        cols[f"new_is_{j}"] = (new == j).astype(float)
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


# ----------------------------------------------------------------------------- models


def _lin(C):
    return Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=C, max_iter=4000))])


def _cv_auroc(make, X, y, g, k=3):
    s = []
    for a, b in GroupKFold(n_splits=k).split(X, y, g):
        if len(np.unique(y[a])) > 1 and len(np.unique(y[b])) > 1:
            s.append(auroc(y[b], make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]))
    return float(np.mean(s)) if s else -np.inf


def select_model(family: str, Xf, y, g):
    """Choose hyperparameters with grouped CV on the rows given -- which are training rows only.

    ``Xf`` is a callable layer -> matrix for the activation families, else a matrix.
    Returns (factory, chosen_params, inner_score).
    """
    if family in ("activation", "combined"):
        best = (-np.inf, None, None)
        for L in Xf.layers:
            s = _cv_auroc(lambda: _lin(1e-2), Xf(L), y, g)
            if s > best[0]:
                best = (s, L, None)
        L = best[1]
        X = Xf(L)
        bc = max(C_GRID, key=lambda C: _cv_auroc(lambda C=C: _lin(C), X, y, g))
        return (lambda C=bc: _lin(C)), {"layer": int(L), "C": bc}, best[0]
    if family == "output_linear":
        bc = max(C_GRID, key=lambda C: _cv_auroc(lambda C=C: _lin(C), Xf, y, g))
        return (lambda C=bc: _lin(C)), {"C": bc}, None
    if family == "output_hgb":
        def mk(p):
            return lambda: HistGradientBoostingClassifier(learning_rate=0.05, random_state=0, **p)
        bp = max(HGB_GRID, key=lambda p: _cv_auroc(mk(p), Xf, y, g))
        return mk(bp), dict(bp), None
    raise ValueError(family)


def oof(make, X, y, g, k=5):
    """Out-of-fold probabilities over the given (training) rows."""
    p = np.full(len(y), np.nan)
    for a, b in GroupKFold(n_splits=k).split(X, y, g):
        p[b] = make().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return p


def choose_tau(p_new, p_old, y_new, y_old, revised) -> float:
    """Tau maximising agent accuracy on TRAINING revised events, from out-of-fold predictions."""
    def acc(t):
        take = (p_new - p_old > t) | ~revised
        return np.mean(np.where(take, y_new, y_old))
    scores = [acc(t) for t in TAU_GRID]
    return float(TAU_GRID[int(np.argmax(scores))])  # ties -> smallest tau, deterministic


# ----------------------------------------------------------------------------- pipeline


class LayerFeatures:
    """Activation features for a fixed set of rows, by layer."""

    def __init__(self, store: ActivationStore, ipre, ipost, rows, extra: np.ndarray | None = None):
        self.store, self.ipre, self.ipost, self.rows, self.extra = store, ipre, ipost, rows, extra
        self.layers = sorted({int(store.n_layers * f) for f in LAYER_FRACS})
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, L):
        if L not in self._cache:
            f = self.store.load_layer(L)
            X = np.hstack([f[self.ipre], f[self.ipost]])
            if self.extra is not None:
                X = np.hstack([X, self.extra])
            self._cache[L] = X
        return self._cache[L][self.rows]

    def subset(self, rows):
        out = LayerFeatures(self.store, self.ipre, self.ipost, rows, self.extra)
        out.layers, out._cache = self.layers, self._cache
        return out


def fit_fold(families, feats, y_old, y_new, g, revised, tr) -> dict:
    """Everything a gate needs, decided using training rows only. Test labels never enter."""
    out = {}
    for fam in families:
        F = feats[fam]
        Ftr = F.subset(tr) if isinstance(F, LayerFeatures) else F[tr]
        models, oofs, params = {}, {}, {}
        for tname, y in (("old", y_old), ("new", y_new)):
            make, prm, _ = select_model(fam, Ftr, y[tr], g[tr])
            X = Ftr(prm["layer"]) if isinstance(Ftr, LayerFeatures) else Ftr
            oofs[tname] = oof(make, X, y[tr], g[tr])
            models[tname] = make().fit(X, y[tr])
            params[tname] = prm
        tau = choose_tau(oofs["new"], oofs["old"], y_new[tr], y_old[tr], revised[tr])
        cal = {t: PlattScaler().fit(oofs[t], (y_new if t == "new" else y_old)[tr]) for t in oofs}
        out[fam] = {"models": models, "params": params, "tau": tau, "cal": cal}
    return out


def predict_fold(fitted, feats, te) -> dict:
    preds = {}
    for fam, f in fitted.items():
        F = feats[fam]
        p = {}
        for t in ("old", "new"):
            X = F.subset(te)(f["params"][t]["layer"]) if isinstance(F, LayerFeatures) else F[te]
            raw = f["models"][t].predict_proba(X)[:, 1]
            p[t] = f["cal"][t].transform(raw)
        preds[fam] = p
    return preds


def panel_paired_bootstrap(ev, acc_a, acc_b, n_boot=2000, seed=0):
    """Paired difference in panel accuracy, resampling questions."""
    a = apply_policy(ev, acc_a).set_index("question_id").panel_correct
    b = apply_policy(ev, acc_b).set_index("question_id").panel_correct.reindex(a.index)
    d = (a - b).to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), [float(x) for x in np.quantile(bs, [0.025, 0.975])]


def run(run_dir: Path, out_dir: Path, seeds=(0, 1, 2), k_outer=5, families=None) -> dict:
    families = families or ("output_linear", "output_hgb", "activation", "combined")
    rows = pd.read_parquet(run_dir / ROWS_FILE)
    rows = rows[rows.phase.isin(["pre", "post"])]
    ev = build_events(rows).reset_index(drop=True)
    full_path = run_dir / "readout_full.parquet"
    full = pd.read_parquet(full_path) if full_path.exists() else None
    Xout, out_names = output_features(ev, full)

    store = ActivationStore(run_dir)
    idx = store.index.reset_index(drop=True)
    pos = {(r.question_id, r.agent_id, r.phase): i for i, r in idx.iterrows()
           if getattr(r, "position", "readout_last") == "readout_last"}
    ipre = np.array([pos[(e.question_id, e.agent_id, "pre")] for e in ev.itertuples()])
    ipost = np.array([pos[(e.question_id, e.agent_id, "post")] for e in ev.itertuples()])
    all_rows = np.arange(len(ev))
    feats = {
        "output_linear": Xout, "output_hgb": Xout,
        "activation": LayerFeatures(store, ipre, ipost, all_rows),
        # unscaled here: every model's own pipeline standardises using its training rows only
        "combined": LayerFeatures(store, ipre, ipost, all_rows, extra=Xout),
    }
    y_old = ev.pre_correct.to_numpy().astype(int)
    y_new = ev.post_correct.to_numpy().astype(int)
    g = ev.question_id.to_numpy()
    rev = (ev.revised == 1).to_numpy()

    report = {"run": str(run_dir), "n_events": int(len(ev)), "n_questions": int(ev.question_id.nunique()),
              "n_revised": int(rev.sum()), "output_features": out_names,
              "full_option_vectors": full is not None, "families": list(families), "seeds": []}
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []

    for seed in seeds:
        uq = np.array(sorted(set(g)))
        rng = np.random.default_rng(seed)
        fold_of = dict(zip(rng.permutation(uq), np.arange(len(uq)) % k_outer, strict=True))
        fold = np.array([fold_of[q] for q in g])
        P = {fam: {"old": np.full(len(ev), np.nan), "new": np.full(len(ev), np.nan)} for fam in families}
        gate = {fam: np.ones(len(ev), bool) for fam in families}
        chosen = []
        for f in range(k_outer):
            tr, te = fold != f, fold == f
            fitted = fit_fold(families, feats, y_old, y_new, g, rev, tr)
            preds = predict_fold(fitted, feats, te)
            for fam in families:
                for t in ("old", "new"):
                    P[fam][t][te] = preds[fam][t]
                gate[fam][te] = (preds[fam]["new"] - preds[fam]["old"] > fitted[fam]["tau"]) | ~rev[te]
                chosen.append({"seed": seed, "fold": f, "family": fam, "tau": fitted[fam]["tau"],
                               "params_old": fitted[fam]["params"]["old"],
                               "params_new": fitted[fam]["params"]["new"]})
            print(f"  seed {seed} fold {f} done", flush=True)

        res = {"seed": seed, "choices": chosen, "families": {}}
        for fam in families:
            fr = {}
            for t, y in (("old", y_old), ("new", y_new)):
                p = P[fam][t]
                c, npair, ngrp = conditional_auroc(y, p, g)
                fr[f"{t}_correct"] = {
                    "auroc_all": auroc(y, p), "auroc_changed": auroc(y[rev], p[rev]),
                    "auroc_within_question": c, "within_pairs": npair,
                    "brier": brier_score_loss(y, p), "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
                    "ece": expected_calibration_error(p, y), "sd_pred": float(np.std(p)),
                }
            acc = gate[fam]
            fr["gate"] = {
                "agent_accuracy": float(np.mean(np.where(acc, y_new, y_old))),
                "panel_accuracy": float(apply_policy(ev, acc).panel_correct.mean()),
                "beneficial_kept": int((acc & (ev.label == "beneficial")).sum()),
                "harmful_blocked": int((~acc & (ev.label == "harmful")).sum()),
                "revisions_accepted": int((acc & rev).sum()),
            }
            res["families"][fam] = fr
        const = {t: np.full(len(ev), y.mean()) for t, y in (("old", y_old), ("new", y_new))}
        res["constant"] = {f"{t}_correct": {"auroc_all": 0.5, "brier": brier_score_loss(y, const[t]),
                                            "ece": expected_calibration_error(const[t], y)}
                           for t, y in (("old", y_old), ("new", y_new))}
        ref = {"accept_all": policy_accept(ev, "mad_standard"),
               "reject_all": policy_accept(ev, "round0_plurality"),
               "agent_optimal": policy_accept(ev, "oracle_full"),
               "panel_optimal": panel_optimal_accept(ev)}
        res["reference_gates"] = {k: {"agent_accuracy": float(np.mean(np.where(v, y_new, y_old))),
                                      "panel_accuracy": float(apply_policy(ev, v).panel_correct.mean())}
                                  for k, v in ref.items()}
        res["paired_panel_differences"] = {}
        for a, b in (("combined", "output_linear"), ("combined", "output_hgb"),
                     ("activation", "output_linear"), ("activation", "output_hgb"),
                     ("activation", "accept_all"), ("combined", "accept_all"),
                     ("output_linear", "accept_all"), ("output_hgb", "accept_all")):
            ga = gate[a] if a in gate else ref[a]
            gb = gate[b] if b in gate else ref[b]
            m, ci = panel_paired_bootstrap(ev, ga, gb, seed=seed)
            res["paired_panel_differences"][f"{a} - {b}"] = {"mean": m, "ci95": ci}
        report["seeds"].append(res)
        for fam in families:
            for i in range(len(ev)):
                pred_rows.append((seed, fam, ev.question_id[i], int(ev.agent_id[i]), int(rev[i]),
                                  y_old[i], y_new[i], P[fam]["old"][i], P[fam]["new"][i], bool(gate[fam][i])))

    pd.DataFrame(pred_rows, columns=["seed", "family", "question_id", "agent_id", "revised", "y_old",
                                     "y_new", "p_old", "p_new", "accept"]).to_csv(
        out_dir / "heldout_predictions.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/night2-debate")
    ap.add_argument("--out", default="artifacts/review/reanalysis")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args(argv)
    rep = run(Path(args.run), Path(args.out), seeds=tuple(args.seeds))
    s0 = rep["seeds"][0]
    print(json.dumps({"reference": s0["reference_gates"], "paired": s0["paired_panel_differences"],
                      "gates": {f: v["gate"] for f, v in s0["families"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
