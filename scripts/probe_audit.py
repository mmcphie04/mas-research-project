"""Probe training audit: a bounded comparison of estimators on FIXED layers and position
(CPU only; exploratory -- the 320 existing questions have all been seen).

    python scripts/probe_audit.py cv          # pilot, grouped 5-fold CV, three training sets
    python scripts/probe_audit.py validation  # fit on the pilot, score the 200 validation questions
    python scripts/probe_audit.py curves      # learning curves (training questions subsampled)
    python scripts/probe_audit.py stability   # bootstrap the training questions; score/selection stability
    python scripts/probe_audit.py paired      # paired question-bootstrap: each activation estimator minus
                                              # the matched output-only estimator (same training set)

Fixed representation: R1's position (readout_last, letter-centered, standardized) at layers 21,
27, 33 (a subset of R1's band; per-layer scores normalized with training-only references and
averaged). No other layers or positions are searched.

Estimators (each fitted per layer; hyperparameters chosen by grouped 3-fold inner CV over
TRAINING questions only; all states of a question stay together):
  meandiff          mean(correct) - mean(incorrect): the current R1 estimator
  lr                L2 logistic regression, C in {1e-4, 1e-3, 1e-2, 1e-1} by inner log-loss
  pca{k}+lr         PCA to k = 16/64/128 components, rescaled to unit variance, then lr
  pca64raw+lr       PCA 64 WITHOUT rescaling (checks whether equal-variance rescaling amplifies noise)
  pairwise          logistic regression on within-question differences (correct - incorrect),
                    no intercept, every question given equal total weight; C by inner pairwise log-loss
  pca64+pairwise    the same on 64 rescaled components
  wq_avgdiff        the average within-question difference (docs/R1_INVESTIGATION.md): a direction,
                    not a fitted ranking model
Output-only baselines with the same estimators (meandiff, lr, pairwise, wq_avgdiff) on the
full output features (imputation, indicators, standardization fitted on training rows).

Calibration: Platt scaling fitted on the inner out-of-fold scores at the chosen settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import candidate_scoring as CS  # noqa: E402

OUT = ROOT / "artifacts/val12b/probe_audit"
PILOT, VAL = ROOT / "runs/cache/probe12b", ROOT / "runs/cache/val12b"
LAYERS, POSITION = (21, 27, 33), "readout_last"
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
TRAIN_SETS = {"originals": ("round0",), "originals+debate": ("round0", "debate"),
              "all_pilot_conditions": ("round0", "debate", "debate_answer_only", "misleading_one", "self_reconsider")}
ACT = ("meandiff", "lr", "pca16+lr", "pca64+lr", "pca128+lr", "pca64raw+lr", "pairwise", "pca64+pairwise", "wq_avgdiff")
OUTP = ("out:meandiff", "out:lr", "out:pairwise", "out:wq_avgdiff")


# ----------------------------------------------------------------------------- data


class Bank:
    """States, labels, groups, the fixed activation layers, and the output features."""

    def __init__(self, cache):
        self.D = CS.StateData.from_cache(cache)
        self.X = {}
        for L in LAYERS:
            X, present = self.D.X(L, POSITION)
            self.X[L] = np.array(X, np.float32)
        self.present = present
        self.M = self.D.output(("of", "ol", "ot"))[0]


def _letter_std(X, letters, tr):
    """Letter centering and standardization fitted on training rows ``tr``."""
    Xc = X.astype(np.float32, copy=True)
    g = Xc[tr].mean(0)
    means = {int(a): Xc[tr & (letters == a)].mean(0) for a in np.unique(letters[tr]) if (tr & (letters == a)).sum() >= 5}
    for a in np.unique(letters):
        Xc[letters == a] -= means.get(int(a), g)
    mu, sd = Xc[tr].mean(0), Xc[tr].std(0) + 1e-6
    return (Xc - mu) / sd


def _out_std(M, tr):
    imp = CS.Imputer().fit(M[tr])
    Z = imp.transform(M)
    s = CS.Standardizer().fit(Z[tr])
    return s.transform(Z)


def _pairs(y, groups, rows):
    """Within-question (correct, incorrect) pairs among ``rows``; weight 1/(pairs in question)."""
    ii, jj, W = [], [], []
    for q in np.unique(groups[rows]):
        idx = np.flatnonzero(rows & (groups == q))
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        if len(pos) and len(neg):
            n = len(pos) * len(neg)
            ii.append(np.repeat(pos, len(neg)))
            jj.append(np.tile(neg, len(pos)))
            W.append(np.full(n, 1.0 / n))
    if not ii:
        return np.array([], int), np.array([], int), np.array([])
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(W)


# ----------------------------------------------------------------------------- estimators


def _fit(kind, Z, y, groups, rows, C):
    """A linear scorer (w, b) fitted on ``rows``."""
    from sklearn.linear_model import LogisticRegression
    if kind == "meandiff":
        return Z[rows & (y == 1)].mean(0) - Z[rows & (y == 0)].mean(0), 0.0
    if kind == "wq_avgdiff":
        return CS.within_question_direction(Z[rows], y[rows], groups[rows]), 0.0
    if kind == "lr":
        m = LogisticRegression(C=C, max_iter=3000).fit(Z[rows], y[rows])
        return m.coef_[0], float(m.intercept_[0])
    if kind == "pairwise":
        i, j, w = _pairs(y, groups, rows)
        if len(i) == 0:
            return np.zeros(Z.shape[1]), 0.0
        d = Z[i] - Z[j]
        m = LogisticRegression(C=C, fit_intercept=False, max_iter=3000).fit(
            np.vstack([d, -d]), np.r_[np.ones(len(d)), np.zeros(len(d))], sample_weight=np.r_[w, w])
        return m.coef_[0], 0.0
    raise ValueError(kind)


def _pca(Z, rows, k, rescale):
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(k, int(rows.sum()) - 1), svd_solver="randomized", random_state=0).fit(Z[rows])
    P = p.transform(Z)
    if rescale:
        P = P / np.sqrt(p.explained_variance_ + 1e-12)
    return P


def _loss(kind, s, y, groups, rows):
    """Held-out loss for choosing C: log-loss of sigmoid(s) (lr), pairwise log-loss (pairwise)."""
    if kind == "pairwise":
        i, j, w = _pairs(y, groups, rows)
        if len(i) == 0:
            return 0.0
        d = s[i] - s[j]
        return float(np.sum(w * np.logaddexp(0, -d)) / w.sum())
    p = np.clip(1 / (1 + np.exp(-s[rows])), 1e-9, 1 - 1e-9)
    return float(-np.mean(y[rows] * np.log(p) + (1 - y[rows]) * np.log(1 - p)))


def _design(est, B, L, tr):
    """The feature matrix for one estimator at one layer (fitted on training rows only)."""
    if est.startswith("out:"):
        return _out_std(B.M, tr)
    Z = _letter_std(B.X[L], B.D.letters, tr)
    if est.startswith("pca"):
        k = int(est[3:].split("+")[0].replace("raw", ""))
        Z = _pca(Z, tr, k, rescale="raw" not in est)
    return Z


def fit_score(est, B, train, target=None, inner_k=3, seed=0):
    """Fit ``est`` on B's ``train`` rows (activation layers averaged); score ``target`` (a Bank
    with the same features; default B itself). Returns test z, Platt p, train in-sample z, chosen C."""
    target = target or B
    kind = est.split("+")[-1].replace("out:", "")
    layers = (None,) if est.startswith("out:") else LAYERS
    folds = np.full(len(B.D.y), -1)
    folds[train] = CS.make_folds(B.D.groups[train], inner_k, seed)
    zs_te, zs_tr, refs, chosen = [], [], [], []
    for L in layers:
        if target is B:
            Z = _design(est, B, L, train)
            Zt = Z
        else:  # fit preprocessing on B's training rows, apply the same transform to the target
            Zt, Z = _transfer_design(est, B, target, L, train)
        Cs = C_GRID if kind in ("lr", "pairwise") else (None,)
        best, oof_best = None, None
        for C in Cs:
            oof = np.full(len(B.D.y), np.nan)
            losses = []
            for f in range(inner_k):
                a, b = train & (folds != f), train & (folds == f)
                Zi = _design(est, B, L, a) if target is B else _transfer_design(est, B, B, L, a)[1]
                w, b0 = _fit(kind, Zi, B.D.y, B.D.groups, a, C)
                oof[b] = Zi[b] @ w + b0
                losses.append(_loss(kind, oof, B.D.y, B.D.groups, b))
            if best is None or np.mean(losses) < best[0]:
                best, oof_best = (np.mean(losses), C), oof
        w, b0 = _fit(kind, Z, B.D.y, B.D.groups, train, best[1])
        mu, sd = np.nanmean(oof_best[train]), np.nanstd(oof_best[train]) + 1e-9
        zs_te.append((Zt @ w + b0 - mu) / sd)
        zs_tr.append((Z @ w + b0 - mu) / sd)
        refs.append((oof_best - mu) / sd)
        chosen.append(best[1])
    z_te, z_tr, ref = np.mean(zs_te, 0), np.mean(zs_tr, 0), np.nanmean(refs, 0)
    from mad.analysis.calibrate import PlattScaler
    pl = PlattScaler().fit(ref[train], B.D.y[train])
    return z_te, pl.transform(z_te), z_tr, chosen


def _transfer_design(est, B, T, L, tr):
    """Preprocessing fitted on B's rows ``tr``, applied to B and to target T. Returns (T matrix, B matrix)."""
    if est.startswith("out:"):
        imp = CS.Imputer().fit(B.M[tr])
        s = CS.Standardizer().fit(imp.transform(B.M[tr]))
        return s.transform(imp.transform(T.M)), s.transform(imp.transform(B.M))
    Xb, Xt = B.X[L], T.X[L]
    g = Xb[tr].mean(0)
    lb, lt = B.D.letters, T.D.letters
    means = {int(a): Xb[tr & (lb == a)].mean(0) for a in np.unique(lb[tr]) if (tr & (lb == a)).sum() >= 5}
    cb = Xb - np.stack([means.get(int(a), g) for a in lb])
    ct = Xt - np.stack([means.get(int(a), g) for a in lt])
    mu, sd = cb[tr].mean(0), cb[tr].std(0) + 1e-6
    Zb, Zt = (cb - mu) / sd, (ct - mu) / sd
    if est.startswith("pca"):
        from sklearn.decomposition import PCA
        k = int(est[3:].split("+")[0].replace("raw", ""))
        p = PCA(n_components=min(k, int(tr.sum()) - 1), svd_solver="randomized", random_state=0).fit(Zb[tr])
        Zb, Zt = p.transform(Zb), p.transform(Zt)
        if "raw" not in est:
            Zb, Zt = Zb / np.sqrt(p.explained_variance_ + 1e-12), Zt / np.sqrt(p.explained_variance_ + 1e-12)
    return Zt, Zb


# ----------------------------------------------------------------------------- evaluation


def evaluate(D, z, p, folds, cond, rows_q=None, n_boot=1000):
    qs = np.array(sorted(set(D.qid))) if rows_q is None else rows_q
    W = CS.boot_weights(len(qs), n_boot, 0)
    e = CS.evaluate_scores(D, z, folds, cond, qs, W, {q: i for i, q in enumerate(qs)})
    dec = CS.decisions(D, cond, {"s": np.where(np.isin(D.qid, qs), z, np.nan)})
    dec = dec[dec.index.isin(qs)]
    rows = np.isin(D.qid, qs) & (D.states.condition == cond).to_numpy()
    cal = CS.calibration(D, p, rows, base_rate=float(np.mean(D.y))) if p is not None else None
    return {"ranking": {k: v["value"] for k, v in e["metrics"].items()},
            "ci": {k: v["ci95"] for k, v in e["metrics"].items()},
            "best_six": int(dec["correct:best_six[s]"].sum()), "best_first": int(dec["correct:best_original[s]"].sum()),
            "pooled6": int(dec["correct:pooled6_vote"].sum()), "n_questions": int(len(dec)),
            "calibration_revised_states": cal}


def _task_cv(est):
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        B = Bank(PILOT)
        D = B.D
        folds = CS.make_folds(D.groups, 5, 0)
        out = {}
        for ts, conds in TRAIN_SETS.items():
            inset = D.states.condition.isin(conds).to_numpy()
            z, p, ztr_eval = np.full(len(D.y), np.nan), np.full(len(D.y), np.nan), []
            chosen = []
            for f in range(5):
                train, test = (folds != f) & inset, folds == f
                zt, pt, ztr, c = fit_score(est, B, train, seed=f)
                z[test], p[test] = zt[test], pt[test]
                chosen.append(c)
                trq = np.array(sorted(set(D.qid[folds != f])))
                ztr_eval.append(evaluate(D, ztr, None, np.zeros(len(D.y), int), "debate", trq, n_boot=0)["ranking"])
            out[ts] = {"heldout": evaluate(D, z, p, folds, "debate"),
                       "train_in_sample_mean": {k: float(np.nanmean([r[k] for r in ztr_eval if r[k] is not None]))
                                                for k in ztr_eval[0]},
                       "chosen_C_by_fold": chosen}
        return est, out


def _task_validation(est):
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        B, V = Bank(PILOT), Bank(VAL)
        out = {}
        for ts, conds in TRAIN_SETS.items():
            inset = B.D.states.condition.isin(conds).to_numpy()
            z, p, _, c = fit_score(est, B, inset, target=V, seed=7)
            out[ts] = {c_: evaluate(V.D, z, p, np.zeros(len(V.D.y), int), c_) for c_ in ("debate", "independent")}
            out[ts]["chosen_C"] = c
            out[ts]["_z"] = z
        return est, out


def _task_curve(est, n_train, rep):
    """Train on n_train randomly drawn pilot questions (all conditions); score the 200 validation
    questions."""
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        B, V = Bank(PILOT), Bank(VAL)
        qs = np.unique(B.D.qid)
        pick = np.random.default_rng(1000 * n_train + rep).choice(qs, n_train, replace=False)
        train = np.isin(B.D.qid, pick)
        z, p, ztr, _ = fit_score(est, B, train, target=V, seed=rep)
        r = {c: evaluate(V.D, z, p, np.zeros(len(V.D.y), int), c, n_boot=0) for c in ("debate", "independent")}
        tr = evaluate(B.D, ztr, None, np.zeros(len(B.D.y), int), "debate", np.array(sorted(pick)), n_boot=0)
        return {"estimator": est, "n_train_questions": n_train, "rep": rep,
                "informative_train_questions": int(sum(0 < B.D.y[train & (B.D.qid == q)].sum() < (train & (B.D.qid == q)).sum()
                                                        for q in pick)),
                **{f"val_{c}_{k}": r[c]["ranking"][k] for c in r for k in ("six", "revision_auc", "state_auc_post", "panel_post")},
                **{f"val_{c}_best_six": r[c]["best_six"] for c in r},
                "train_six": tr["ranking"]["six"], "train_state_auc_post": tr["ranking"]["state_auc_post"]}


def _task_stability(est, rep):
    """Bootstrap the 120 pilot questions (all conditions), fit, score validation."""
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        B, V = Bank(PILOT), Bank(VAL)
        qs = np.unique(B.D.qid)
        draw = np.random.default_rng(rep).choice(qs, len(qs), replace=True)
        uniq = np.unique(draw)  # a bootstrap of questions; duplicates add weight we approximate by inclusion
        train = np.isin(B.D.qid, uniq)
        z, _, _, _ = fit_score(est, B, train, target=V, seed=rep)
        return est, rep, z


def main(argv=None) -> int:
    from joblib import Parallel, delayed
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cv", "validation", "curves", "stability", "paired"])
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    if a.cmd == "paired":
        (OUT / "paired.json").write_text(json.dumps(paired(), indent=1, default=float))
    elif a.cmd in ("cv", "validation"):
        task = _task_cv if a.cmd == "cv" else _task_validation
        res = dict(Parallel(n_jobs=a.jobs)(delayed(task)(e) for e in ACT + OUTP))
        if a.cmd == "validation":  # scores of the 200 validation questions (compact; no activations)
            V = CS.StateData.from_cache(VAL)
            S = V.states.reset_index(drop=True)[["question_id", "agent_id", "phase", "condition"]].copy()
            for e, r in res.items():
                for ts in TRAIN_SETS:
                    S[f"{e}|{ts}"] = np.round(r[ts].pop("_z"), 5)
            S.to_csv(OUT / "validation_scores.csv.gz", index=False)
        (OUT / f"{a.cmd}.json").write_text(json.dumps(res, indent=1, default=float))
    elif a.cmd == "curves":
        ests = ("meandiff", "lr", "pca64+lr", "pairwise", "wq_avgdiff", "out:lr", "out:pairwise")
        rows = Parallel(n_jobs=a.jobs)(delayed(_task_curve)(e, n, r) for e in ests for n in (24, 48, 72, 96, 120)
                                       for r in range(5 if n < 120 else 1))
        pd.DataFrame(rows).to_csv(OUT / "learning_curves.csv", index=False)
    else:
        ests = ("meandiff", "lr", "pca16+lr", "pca64+lr", "pca64raw+lr", "pairwise", "wq_avgdiff", "out:lr", "out:pairwise")
        res = Parallel(n_jobs=a.jobs)(delayed(_task_stability)(e, r) for e in ests for r in range(10))
        V = CS.StateData.from_cache(VAL)
        summary = stability_summary(V, res)
        (OUT / "stability.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"{a.cmd} done in {time.time() - t:.0f}s")
    return 0


def paired(n_boot=2000, baselines=("out:lr", "out:pairwise")):
    """Each activation estimator minus each matched output-only estimator (same training set), on
    the 200 validation questions: differences in six-candidate AUROC, revision AUROC, and
    best-of-six / best-of-first-three correct counts, with a shared question bootstrap."""
    V = CS.StateData.from_cache(VAL)
    S = pd.read_csv(OUT / "validation_scores.csv.gz")
    assert (S.question_id.astype(str).to_numpy() == V.states.question_id.astype(str).to_numpy()).all()
    qs = np.array(sorted(set(V.qid)))
    W = CS.boot_weights(len(qs), n_boot, 0)
    qindex = {q: i for i, q in enumerate(qs)}
    folds = np.zeros(len(V.y), int)
    cache = {}

    def metrics(col, cond):
        if (col, cond) not in cache:
            z = S[col].to_numpy(float)
            e = CS.evaluate_scores(V, z, folds, cond, qs, W, qindex, return_vals=True)
            d = CS.decisions(V, cond, {"s": z}).reindex(qs)
            cache[(col, cond)] = {"six": e["vals"]["six"], "revision_auc": e["vals"]["revision_auc"],
                                  "best_six": d["correct:best_six[s]"].to_numpy(float),
                                  "best_first": d["correct:best_original[s]"].to_numpy(float)}
        return cache[(col, cond)]

    out = []
    for base in baselines:
        for e in ACT:
            for ts in TRAIN_SETS:
                for cond in ("debate", "independent"):
                    a, b = metrics(f"{e}|{ts}", cond), metrics(f"{base}|{ts}", cond)
                    row = {"estimator": e, "baseline": base, "train_set": ts, "condition": cond}
                    for k in ("six", "revision_auc"):
                        row[k] = CS.summarize(a[k] - b[k])
                    for k in ("best_six", "best_first"):
                        dq = a[k] - b[k]
                        row[k + "_correct_diff"] = {"value": float(dq.sum()), "ci95": [float(x) * len(qs) for x in np.quantile((W[1:] @ dq) / W[1:].sum(1), [0.025, 0.975])],
                                                    "questions_differ": int((dq != 0).sum())}
                    out.append(row)
    return {"n_questions": int(len(qs)), "n_boot": n_boot, "comparisons": out}


def stability_summary(V, res):
    """Across bootstrap fits: score correlation, selection agreement, and the systematic
    revised-minus-first offset (mean) vs its fit-to-fit variation (instability)."""
    st = V.states.reset_index(drop=True)
    pre = st[st.phase == "pre"]
    n_first = pre.groupby("question_id").answer.nunique()
    out = {}
    for est in sorted({e for e, _, _ in res}):
        Z = np.vstack([z for e, _, z in res if e == est])
        cors = np.corrcoef(Z)[np.triu_indices(len(Z), 1)]
        picks = {}
        for c in ("debate", "independent"):
            P6, P3 = [], []
            for z in Z:
                d = CS.decisions(V, c, {"s": z})
                P6.append(d["best_six[s]"].to_numpy())
                P3.append(d["best_original[s]"].to_numpy())
            P6, P3 = np.array(P6), np.array(P3)
            agree = lambda P: float(np.mean([np.mean(P[:, j] == pd.Series(P[:, j]).mode()[0]) for j in range(P.shape[1])]))  # noqa: E731
            picks[c] = {"best_six_agreement_with_modal_pick": agree(P6), "best_first_agreement_with_modal_pick": agree(P3),
                        "best_six_correct_range": [int(min(row.sum() for row in (P6 == d["gold"].to_numpy()))),
                                                   int(max(row.sum() for row in (P6 == d["gold"].to_numpy())))]}
        offs = {}
        for c in ("debate", "independent"):
            post = st[(st.phase == "post") & (st.condition == c)]
            ip = [int(pre.index[(pre.question_id == q) & (pre.agent_id == a)][0]) for q, a in zip(post.question_id, post.agent_id, strict=True)]
            delta = Z[:, post.index.to_numpy()] - Z[:, ip]
            un = (n_first.reindex(post.question_id).to_numpy() == 1)
            per_fit_un, per_fit_dis = delta[:, un].mean(1), delta[:, ~un].mean(1)
            offs[c] = {"unanimous_mean": float(per_fit_un.mean()), "unanimous_sd_across_fits": float(per_fit_un.std()),
                       "disputed_mean": float(per_fit_dis.mean()), "disputed_sd_across_fits": float(per_fit_dis.std()),
                       "state_delta_sd_across_fits_mean": float(delta.std(0).mean()),
                       "state_delta_sd_across_states_mean": float(delta.std(1).mean())}
        out[est] = {"fits": int(len(Z)), "score_correlation_between_fits": {"median": float(np.median(cors)), "min": float(cors.min())},
                    "selection": picks, "revised_minus_first": offs}
    return out


if __name__ == "__main__":
    raise SystemExit(main())
