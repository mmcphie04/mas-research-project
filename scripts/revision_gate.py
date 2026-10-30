"""EXPLORATORY revision-gate feasibility (2026-10-30 request, sections 2-4).

    python scripts/revision_gate.py --out artifacts/revision_gate

2. inventory of revision events in every Gemma-3-12B dataset (main250 test: counts only);
3. oracle gate ceilings: agent and panel accuracy under accept-all, reject-all, a perfect gate
   and random gates right 60% / 70% of the time on helpful/harmful revisions (wrong->wrong
   revisions accepted with probability 1/2), as exact expected values;
4. output-feature gate: L2 logistic regression and a shallow gradient-boosted tree predicting
   "helpful" among helpful/harmful revisions from the fixed features
   (``revision_events.FEATURES``), with 5-fold question-grouped cross-validation, the
   hyperparameter and the acceptance threshold tuned inside the training folds; the gate then
   decides every changed answer of the held-out questions, and is simulated on agent and panel
   accuracy against accept-all, reject-all and three fixed rules.

main250's test questions never enter sections 3-4. Every interval: 95% percentile, 10,000
question-grouped resamples (seed 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.analysis import revision_events as R  # noqa: E402

N_BOOT, SEED, K_OUTER, K_INNER = 10000, 0, 5, 4
LR_C = [0.01, 0.1, 1.0, 10.0]
GB_DEPTH = [1, 2, 3]
LABEL = "EXPLORATORY"


# ----------------------------------------------------------------------------- bootstrap


def cluster_mean(values: np.ndarray, groups: np.ndarray, n_boot=N_BOOT) -> dict:
    """Mean of per-row values, questions (groups) resampled with replacement."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"n": 0}
    g, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, values, len(g)), np.bincount(inv, None, len(g))
    idx = np.random.default_rng(SEED).integers(0, len(g), (n_boot, len(g)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return {"n": int(len(values)), "questions": int(len(g)), "mean": float(values.mean()),
            "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def auroc(scores, labels) -> float:
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    r = pd.Series(scores).rank().to_numpy()
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def cluster_stat(fn, groups, *arrays, n_boot=N_BOOT) -> dict:
    groups = np.asarray(groups)
    g, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(g))]
    point = fn(*arrays)
    rng = np.random.default_rng(SEED)
    b = []
    for _ in range(n_boot):
        rows = np.concatenate([members[i] for i in rng.integers(0, len(g), len(g))])
        b.append(fn(*[a[rows] for a in arrays]))
    b = np.array(b)
    b = b[np.isfinite(b)]
    return {"n": int(len(groups)), "questions": int(len(g)), "mean": float(point),
            "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


# ----------------------------------------------------------------------------- section 3


def panels(ev: pd.DataFrame):
    for (d, c, q), g in ev.groupby(["dataset", "condition", "question_id"], sort=True):
        g = g.sort_values("agent_id")
        yield d, c, q, g


def oracle(ev: pd.DataFrame) -> dict:
    out = {}
    sel = ev[((ev.dataset == "val12b") & (ev.condition == "debate")) | (ev.dataset == "main250")]
    for (d, c), sub in sel.groupby(["dataset", "condition"], sort=False):
        rows = []
        for _, _, q, g in panels(sub):
            b, a, gold = g.before.tolist(), g.after.tolist(), int(g.gold.iloc[0])
            r = {"question_id": q}
            r["reject_all"] = R.panel_expectation(b, a, gold, [0.0] * len(b))
            r["accept_all"] = R.panel_expectation(b, a, gold, [1.0] * len(b))
            r["perfect"] = R.perfect_gate(b, a, gold)
            r["random_60"] = R.random_gate(b, a, gold, 0.6)
            r["random_70"] = R.random_gate(b, a, gold, 0.7)
            rows.append(r)
        qs = np.array([r["question_id"] for r in rows])
        res = {"questions": len(rows), "panel_k": int(sub.panel_k.iloc[0]), "policies": {}}
        for pol in ("reject_all", "accept_all", "perfect", "random_60", "random_70"):
            ag = np.array([r[pol][0] for r in rows])
            pa = np.array([r[pol][1] for r in rows])
            pa_acc = np.array([r["accept_all"][1] for r in rows])
            res["policies"][pol] = {"agent": cluster_mean(ag, qs), "panel": cluster_mean(pa, qs),
                                    "panel_minus_accept_all": cluster_mean(pa - pa_acc, qs)}
        out[f"{d} {c}"] = res
    return out


# ----------------------------------------------------------------------------- section 4


def training_events(ev: pd.DataFrame) -> pd.DataFrame:
    keep = ev.apply(lambda r: (r.dataset, r.condition) in R.TRAIN_SOURCES, axis=1)
    return ev[keep].reset_index(drop=True)


def question_folds(qids, k, seed) -> dict:
    u = np.array(sorted(set(qids)))
    perm = np.random.default_rng(seed).permutation(len(u))
    return {q: int(i % k) for i, q in zip(perm, u, strict=True)}


def make_model(kind: str, h):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                             LogisticRegression(C=h, max_iter=5000))
    return HistGradientBoostingClassifier(max_depth=h, max_iter=100, learning_rate=0.1, min_samples_leaf=20,
                                          random_state=0)


def _fit_predict(kind, h, Xtr, ytr, Xte):
    m = make_model(kind, h)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def best_threshold(p: np.ndarray, helpful: np.ndarray) -> float:
    """Threshold maximizing (helpful accepted - harmful accepted) on the given events; accept iff
    p >= t. Ties: the middle of the best candidates."""
    cand = np.unique(np.concatenate([p, [np.inf]]))
    net = np.array([((p >= t) & (helpful == 1)).sum() - ((p >= t) & (helpful == 0)).sum() for t in cand])
    best = cand[net == net.max()]
    return float(best[len(best) // 2])


def inner_tune(kind, grid, X, y, qids):
    folds = question_folds(qids, K_INNER, SEED + 1)
    f = np.array([folds[q] for q in qids])
    scores, oof_by_h = {}, {}
    for h in grid:
        oof = np.zeros(len(y))
        for k in range(K_INNER):
            tr, te = f != k, f == k
            oof[te] = _fit_predict(kind, h, X[tr], y[tr], X[te])
        scores[h], oof_by_h[h] = auroc(oof, y), oof
    h = max(grid, key=lambda x: (scores[x], -grid.index(x)))
    return h, best_threshold(oof_by_h[h], y), scores


def cross_validate(tev: pd.DataFrame) -> pd.DataFrame:
    """Out-of-fold P(helpful) for every changed answer; each fold's model sees only the other
    folds' helpful/harmful events."""
    ch = tev[tev.changed == 1].copy()
    folds = question_folds(tev.question_id, K_OUTER, SEED)
    ch["fold"] = ch.question_id.map(folds)
    X = ch[R.FEATURES].to_numpy(float)
    hh = ch.kind.isin(["helpful", "harmful"]).to_numpy()
    y = (ch.kind == "helpful").to_numpy(int)
    for kind, grid in (("logistic", LR_C), ("boosted_tree", GB_DEPTH)):
        p = np.full(len(ch), np.nan)
        acc = np.zeros(len(ch), int)
        tuned = []
        for k in range(K_OUTER):
            tr = (ch.fold != k).to_numpy() & hh
            te = (ch.fold == k).to_numpy()
            h, t, sc = inner_tune(kind, grid, X[tr], y[tr], ch.question_id.to_numpy()[tr])
            p[te] = _fit_predict(kind, h, X[tr], y[tr], X[te])
            acc[te] = (p[te] >= t).astype(int)
            tuned.append({"fold": k, "hyperparameter": h, "threshold": t,
                          "inner_auroc": {str(a): round(b, 4) for a, b in sc.items()}})
        ch[f"p_{kind}"], ch[f"accept_{kind}"] = p, acc
        ch.attrs[f"tuning_{kind}"] = tuned
    return ch


def rule_accept(r) -> dict:
    return {"accept_all": 1, "reject_all": 0,
            "a_new_is_peer_majority": int(r.new_is_peer_majority == 1),
            "b_conviction_reject_if_conf_before_ge_90": int(not (r.conf_before >= 90)),
            "c_mad_conformist_keep_if_old_is_prev_majority": int(r.before != r.prev_round_majority)}


GATES = ["accept_all", "reject_all", "a_new_is_peer_majority", "b_conviction_reject_if_conf_before_ge_90",
         "c_mad_conformist_keep_if_old_is_prev_majority", "logistic", "boosted_tree", "perfect"]


def simulate(tev: pd.DataFrame, ch: pd.DataFrame) -> pd.DataFrame:
    dec = {(r.dataset, r.condition, r.question_id, r.agent_id): r for r in ch.itertuples(index=False)}
    rows = []
    for d, c, q, g in panels(tev):
        b, a, gold = g.before.tolist(), g.after.tolist(), int(g.gold.iloc[0])
        row = {"dataset": d, "condition": c, "question_id": q}
        for gate in GATES:
            if gate == "perfect":
                row[gate] = R.perfect_gate(b, a, gold)
                continue
            p = []
            for r in g.itertuples(index=False):
                if r.before == r.after:
                    p.append(0.0)
                    continue
                x = dec[(d, c, q, r.agent_id)]
                p.append(float(getattr(x, f"accept_{gate}") if gate in ("logistic", "boosted_tree")
                               else rule_accept(x)[gate]))
            row[gate] = R.panel_expectation(b, a, gold, p)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_sim(sim: pd.DataFrame) -> dict:
    def block(s):
        qs = s.question_id.to_numpy()
        acc_all = np.array([x[1] for x in s.accept_all])
        return {"panels": len(s), **{g: {"agent": cluster_mean(np.array([x[0] for x in s[g]]), qs),
                                         "panel": cluster_mean(np.array([x[1] for x in s[g]]), qs),
                                         "panel_minus_accept_all": cluster_mean(np.array([x[1] for x in s[g]]) - acc_all, qs)}
                                     for g in GATES}}
    out = {"all panels": block(sim)}
    for (d, c), s in sim.groupby(["dataset", "condition"], sort=False):
        out[f"{d} {c}"] = block(s)
    return out


def coefficients(tev: pd.DataFrame) -> dict:
    hh = tev[(tev.changed == 1) & tev.kind.isin(["helpful", "harmful"])]
    X, y = hh[R.FEATURES].to_numpy(float), (hh.kind == "helpful").to_numpy(int)
    h, _, scores = inner_tune("logistic", LR_C, X, y, hh.question_id.to_numpy())
    m = make_model("logistic", h).fit(X, y)
    imp = m.named_steps["simpleimputer"]
    names = list(R.FEATURES) + [f"missing:{R.FEATURES[i]}" for i in imp.indicator_.features_]
    coef = m.named_steps["logisticregression"].coef_[0]
    return {"C": h, "grouped_cv_auroc_by_C": {str(k): round(v, 4) for k, v in scores.items()},
            "standardized_coefficients": dict(sorted(zip(names, coef.round(4).tolist(), strict=True),
                                                     key=lambda kv: -abs(kv[1]))),
            "intercept": float(m.named_steps["logisticregression"].intercept_[0]),
            "note": "fitted on all non-test helpful/harmful events; positive = more likely helpful"}


def feasibility(ev: pd.DataFrame) -> dict:
    tev = training_events(ev)
    ch = cross_validate(tev)
    hh = ch[ch.kind.isin(["helpful", "harmful"])]
    y = (hh.kind == "helpful").to_numpy(int)
    qs = hh.question_id.to_numpy()
    models = {}
    for kind in ("logistic", "boosted_tree"):
        p, a = hh[f"p_{kind}"].to_numpy(), hh[f"accept_{kind}"].to_numpy()
        models[kind] = {"auroc": cluster_stat(auroc, qs, p, y), "accuracy": cluster_mean((a == y).astype(float), qs),
                        "accept_rate_helpful_harmful": float(a.mean()), "accept_rate_wrong_wrong":
                        float(ch[ch.kind == "wrong_wrong"][f"accept_{kind}"].mean()),
                        "tuning": ch.attrs[f"tuning_{kind}"]}
        models[kind]["auroc_by_protocol"] = {
            str(w): cluster_stat(auroc, qs[hh.protocol == w], p[hh.protocol.to_numpy() == w], y[hh.protocol.to_numpy() == w])
            for w in (150, 250)}
    base = {"helpful": int(y.sum()), "harmful": int(len(y) - y.sum()), "wrong_wrong": int((ch.kind == "wrong_wrong").sum()),
            "questions": int(tev.question_id.nunique()), "majority_class_accuracy": float(max(y.mean(), 1 - y.mean()))}
    return {"data": base, "models": models, "simulation": summarize_sim(simulate(tev, ch)),
            "logistic_coefficients": coefficients(tev),
            "sources": sorted(f"{d} {c}" for d, c in R.TRAIN_SOURCES)}


# ----------------------------------------------------------------------------- markdown


def ci(x, nd=3):
    if not x or "mean" not in x:
        return "—"
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"


def md_inventory(inv, ev) -> str:
    L = [f"# Revision events in every Gemma-3-12B dataset ({LABEL})", "",
         "Helpful = wrong → right; harmful = right → wrong; wrong → wrong = changed between wrong answers. "
         "main250 test questions: counts only (reserved).", "",
         "| dataset | condition | protocol (words) | panel | part | questions | changed | helpful | harmful | wrong→wrong | questions with ≥1 helpful/harmful | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in inv:
        L.append(f"| {r['dataset']} | {r['condition']} | {r['protocol']} | {r['panel_k']} | {r['part']} | {r['questions']} | "
                 f"{r['changed']} | {r['helpful']} | {r['harmful']} | {r['wrong_wrong']} | "
                 f"{r['questions_with_helpful_or_harmful']} | {r['note']} |")
    nt = [r for r in inv if not r["part"].startswith("test")]
    peer = [r for r in nt if (r["dataset"], r["condition"]) in R.TRAIN_SOURCES]
    tot = lambda rs, k: sum(r[k] for r in rs)  # noqa: E731
    L += ["", "**Available for training (main250 test excluded):**", "",
          f"- all conditions: {tot(nt, 'helpful')} helpful + {tot(nt, 'harmful')} harmful = "
          f"{tot(nt, 'helpful') + tot(nt, 'harmful')} events ({tot(nt, 'wrong_wrong')} wrong→wrong);",
          f"- revisions after seeing peers (the gate's sources): {tot(peer, 'helpful')} helpful + {tot(peer, 'harmful')} "
          f"harmful = {tot(peer, 'helpful') + tot(peer, 'harmful')} events ({tot(peer, 'wrong_wrong')} wrong→wrong), "
          f"{ev[ev.apply(lambda r: (r.dataset, r.condition) in R.TRAIN_SOURCES, axis=1)].question_id.nunique()} questions.",
          "", "The gate uses only revisions made after seeing peers (its features describe the peers); "
          "self-reconsideration, independent re-answers and pilot250's 4-question second round are listed but not used. "
          "pilot250's 12 questions are validation-pool questions also in val12b; resampling and folds group by question id "
          "across datasets."]
    return "\n".join(L) + "\n"


POL = {"reject_all": "reject every revision", "accept_all": "accept every revision", "perfect": "perfect gate",
       "random_60": "random gate, 60% right", "random_70": "random gate, 70% right"}


def md_oracle(o) -> str:
    L = [f"# Oracle gate ceilings ({LABEL})", "",
         "Exact expected values per panel. Perfect gate: accepts helpful, rejects harmful, and for wrong→wrong "
         "revisions takes the accept/reject combination best for the panel. Random gate: right with probability "
         "0.6 / 0.7 on each helpful/harmful revision independently; wrong→wrong accepted with probability ½. "
         "main250: train+validation questions only.", ""]
    for name, r in o.items():
        L += [f"## {name} (panel {r['panel_k']}, {r['questions']} questions)", "",
              "| policy | agent accuracy | panel accuracy | panel − accept-all |", "|---|---|---|---|"]
        for p, v in r["policies"].items():
            L.append(f"| {POL[p]} | {ci(v['agent'])} | {ci(v['panel'])} | {ci(v['panel_minus_accept_all'])} |")
        L.append("")
    return "\n".join(L)


GNAME = {"accept_all": "accept all", "reject_all": "reject all", "a_new_is_peer_majority": "(a) accept iff new = peer majority",
         "b_conviction_reject_if_conf_before_ge_90": "(b) conviction: reject if confidence before ≥ 90",
         "c_mad_conformist_keep_if_old_is_prev_majority": "(c) MAD-Conformist: keep if old = previous majority",
         "logistic": "logistic gate (CV)", "boosted_tree": "boosted-tree gate (CV)", "perfect": "perfect gate (oracle)"}


def md_feasibility(f) -> str:
    d = f["data"]
    L = [f"# Output-feature revision gate: feasibility ({LABEL})", "",
         f"Revisions after seeing peers, all Gemma-3-12B datasets except main250 test: {d['helpful']} helpful, "
         f"{d['harmful']} harmful, {d['wrong_wrong']} wrong→wrong, {d['questions']} questions. Target: helpful among "
         f"helpful/harmful (majority-class accuracy {d['majority_class_accuracy']:.3f}). Features (fixed): "
         f"{', '.join(R.FEATURES)}. 5-fold question-grouped cross-validation; C (logistic, {LR_C}) or depth (tree, "
         f"{GB_DEPTH}) and the acceptance threshold tuned in 4 inner grouped folds (threshold: most helpful-minus-harmful "
         "accepted on inner out-of-fold predictions). The gate decides every changed answer of the held-out questions.",
         "", "## Prediction (out-of-fold, helpful/harmful events)", "",
         "| model | AUROC | AUROC, 150-word data | AUROC, 250-word data | accuracy (tuned threshold) | accepts helpful/harmful | accepts wrong→wrong |",
         "|---|---|---|---|---|---|---|"]
    for k, m in f["models"].items():
        L.append(f"| {k} | {ci(m['auroc'])} | {ci(m['auroc_by_protocol']['150'])} | {ci(m['auroc_by_protocol']['250'])} | "
                 f"{ci(m['accuracy'])} | {100 * m['accept_rate_helpful_harmful']:.0f}% | {100 * m['accept_rate_wrong_wrong']:.0f}% |")
    L += ["", "## Simulated agent and panel accuracy", ""]
    for name, b in f["simulation"].items():
        L += [f"### {name} ({b['panels']} panels)", "", "| gate | agent accuracy | panel accuracy | panel − accept-all |",
              "|---|---|---|---|"]
        for g in GATES:
            L.append(f"| {GNAME[g]} | {ci(b[g]['agent'])} | {ci(b[g]['panel'])} | {ci(b[g]['panel_minus_accept_all'])} |")
        L.append("")
    c = f["logistic_coefficients"]
    L += ["## Logistic coefficients (standardized; fitted on all events above)", "",
          f"C = {c['C']} (grouped CV AUROC by C: {c['grouped_cv_auroc_by_C']}); intercept {c['intercept']:.3f}. "
          "Positive = more likely helpful.", "", "| feature | coefficient |", "|---|---|"]
    L += [f"| {k} | {v:+.3f} |" for k, v in c["standardized_coefficients"].items()]
    return "\n".join(L) + "\n"


def _default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    L = R.load_events(a.root)
    inv = R.inventory(L)
    (out / "inventory.json").write_text(json.dumps({"label": LABEL, "rows": inv}, indent=1, default=_default) + "\n")
    (out / "inventory.md").write_text(md_inventory(inv, L["events"]))
    o = oracle(L["events"])
    (out / "oracle.json").write_text(json.dumps({"label": LABEL, **o}, indent=1, default=_default) + "\n")
    (out / "oracle.md").write_text(md_oracle(o))
    f = feasibility(L["events"])
    (out / "feasibility.json").write_text(json.dumps({"label": LABEL, **f}, indent=1, default=_default) + "\n")
    (out / "feasibility.md").write_text(md_feasibility(f))
    print("written:", sorted(p.name for p in out.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
