"""Why does the frozen R1 activation selector pick the same option from the six pooled answers as
from the three first answers alone? (CPU only; exploratory: motivated by the validation tranche.)

    python scripts/investigate_r1.py reproduce   # exported scores from raw vectors; indexing, letters, calibration
    python scripts/investigate_r1.py shifts      # post-minus-pre score shifts by agreement, correctness, condition, layer
    python scripts/investigate_r1.py compare     # current vs within-question scorers x training sets, output-matched
    python scripts/investigate_r1.py counterfactual  # remove the context shift and recount the six-way picks

Outputs: artifacts/val12b/investigation/. Uncertainty: bootstrap over questions (10,000 draws),
since an agent's first and revised states, and the three agents of a question, are not
independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import candidate_scoring as CS  # noqa: E402

OUT = ROOT / "artifacts/val12b/investigation"
FROZEN = ROOT / "runs/frozen/pilot12b_scorers_3857a72.pkl"
VAL_CACHE, SELF_CACHE, PILOT_CACHE = ROOT / "runs/cache/val12b", ROOT / "runs/cache/val12b_self", ROOT / "runs/cache/probe12b"
LETTERS = "ABCDEFGHIJ"


def _boot_mean(values, groups, n_boot=10000, seed=0):
    """Mean of per-state values with a question-clustered bootstrap interval."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    if len(values) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0, "questions": 0}
    u, inv = np.unique(groups, return_inverse=True)
    s = np.bincount(inv, weights=values, minlength=len(u))
    c = np.bincount(inv, minlength=len(u)).astype(float)
    W = CS.boot_weights(len(u), n_boot, seed)[1:]
    m = (W @ s) / np.maximum(W @ c, 1e-12)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "n": int(len(values)), "questions": int(len(u))}


def _frozen():
    F = pickle.load(open(FROZEN, "rb"))
    return F, hashlib.sha256(FROZEN.read_bytes()).hexdigest()


def _picks(D, z, cond):
    """Per question: the option chosen by the highest score among first answers, revised answers,
    and all six; and whether the six-way pick is a revised state."""
    st = D.states.reset_index(drop=True)
    pre, post = st[st.phase == "pre"], st[(st.phase == "post") & (st.condition == cond)]
    rows = []
    for q, g0 in pre.groupby("question_id"):
        g1 = post[post.question_id == q]
        if len(g1) != len(g0):
            continue
        g0, g1 = g0.sort_values("agent_id"), g1.sort_values("agent_id")
        a0, a1 = g0.answer.to_numpy(), g1.answer.to_numpy()
        s0, s1 = z[g0.index], z[g1.index]
        k6 = int(np.nanargmax(np.r_[s0, s1]))
        rows.append({"question_id": q, "best_original": int(a0[np.nanargmax(s0)]), "best_revised": int(a1[np.nanargmax(s1)]),
                     "best_six": int(np.r_[a0, a1][k6]), "six_pick_is_revised_state": int(k6 >= 3),
                     "n_distinct_first": len(set(a0)), "any_revision": int((a0 != a1).any())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 1. reproduce


def reproduce():
    from mad.analysis.states import StateBank
    F, sha = _frozen()
    act = F["models"]["R1"]["act"]
    D = CS.StateData.from_cache(VAL_CACHE)
    allrows = np.ones(len(D.y), bool)
    exp = pd.read_csv(ROOT / "artifacts/val12b/frozen/scores_R1.csv.gz").set_index("row_id").loc[D.states.row_id]
    raw_c, z_c, p_c = act.score(D, allrows)
    rep = {"frozen_sha256": sha, "states": int(len(D.y))}
    rep["cache_path_vs_export"] = {k: float(np.nanmax(np.abs(v - exp[f"act:{k}"].to_numpy())))
                                   for k, v in (("raw", raw_c), ("z", z_c), ("p", p_c))}
    # independent path: raw vectors from the ARCHIVE views (not the cache), hand-written formula
    banks = {c: StateBank(ROOT / f"runs/val12b/val12b-{c}") for c in ("debate", "independent")}
    st = D.states
    view = np.where(st.phase == "pre", "debate", st.condition)
    keys = st.assign(condition=view)[["condition", "question_id", "agent_id"]]
    acc = np.zeros(len(st))
    for L in act.layers:
        X = np.zeros((len(st), 3840), np.float32)
        for v in ("debate", "independent"):
            for ph in ("pre", "post"):
                m = (view == v) & (st.phase == ph).to_numpy()
                if m.any():
                    X[m] = banks[v].matrix(keys[m], ph, "readout_last", L)[0]
        p = act.probes[L]
        Xc = X - np.stack([p.letter_means.get(int(a), p.global_mean) for a in st.answer])
        s = ((Xc - p.std.mu) / p.std.sd) @ p.w
        acc += (s - act.layer_norm[L].mu) / act.layer_norm[L].sd
    z_manual = (acc / len(act.layers) - act.norm.mu) / act.norm.sd
    lr = act.platt._lr
    p_manual = 1 / (1 + np.exp(-(lr.coef_[0][0] * z_manual + lr.intercept_[0])))
    rep["archive_manual_vs_export"] = {"z": float(np.max(np.abs(z_manual - exp["act:z"].to_numpy()))),
                                       "p": float(np.max(np.abs(p_manual - exp["act:p"].to_numpy())))}
    # shared round 0: the independent view's first-answer vectors equal the debate view's
    k0 = st[st.phase == "pre"][["question_id", "agent_id"]].assign(condition="independent")
    a = banks["independent"].matrix(k0, "pre", "readout_last", 24)[0]
    b = banks["debate"].matrix(k0.assign(condition="debate"), "pre", "readout_last", 24)[0]
    rep["round0_identical_across_views"] = bool(np.array_equal(a, b))
    # answer letters used for letter centering vs the letter written in the response
    tr = {}
    for c in ("debate", "independent"):
        for line in open(ROOT / f"runs/val12b/val12b-{c}/transcripts.jsonl"):
            r = json.loads(line)
            tr[(str(r["question_id"]), r["agent_id"], r["phase"], "round0" if r["phase"] == "pre" else c)] = r["text"]
    agree = tot = 0
    for _, r in st[st.answer_source == "stated"].iterrows():
        m = re.findall(r"Answer:\s*\(?([A-J])\)?", tr[(r.question_id, r.agent_id, r.phase, r.condition)])
        if m:
            tot += 1
            agree += int(LETTERS.index(m[-1]) == r.answer)
    rep["answer_letter_matches_response_text"] = {"matched": agree, "checked": tot}
    rep["post_index"] = {"pre_rows": int((st.phase == "pre").sum()),
                         "post_rows": {c: int(((st.phase == "post") & (st.condition == c)).sum()) for c in ("debate", "independent")},
                         "every_post_has_matching_pre": bool(all(((st.phase == "pre") & (st.question_id == q) & (st.agent_id == a)).any()
                                                                 for q, a in zip(st[st.phase == "post"].question_id,
                                                                                 st[st.phase == "post"].agent_id, strict=True)))}
    rep["validation_z"] = {"mean": float(np.nanmean(z_c)), "sd": float(np.nanstd(z_c))}
    picks = {c: _picks(D, z_c, c) for c in ("debate", "independent")}
    rep["option_changes"] = {c: {"questions": int(len(pk)),
                                 "revised_only_vs_first": int((pk.best_revised != pk.best_original).sum()),
                                 "six_vs_first": int((pk.best_six != pk.best_original).sum()),
                                 "six_pick_is_a_revised_state": int(pk.six_pick_is_revised_state.sum())}
                             for c, pk in picks.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reproduce.json").write_text(json.dumps(rep, indent=1))
    picks["debate"].to_csv(OUT / "picks_debate.csv", index=False)
    print(json.dumps(rep, indent=1))


# ----------------------------------------------------------------------------- 2. shifts


def _pairs_table(D, z, cond, layer_z=None):
    """One row per agent in ``cond``: first and revised scores, and descriptors of the question
    (initial agreement) and of the change."""
    st = D.states.reset_index(drop=True)
    pre_idx = {(q, int(a)): i for i, (q, a) in enumerate(zip(st.question_id, st.agent_id, strict=True))
               if st.phase.iloc[i] == "pre"}
    n_first = st[st.phase == "pre"].groupby("question_id").answer.nunique()
    out = []
    for i in np.flatnonzero(((st.phase == "post") & (st.condition == cond)).to_numpy()):
        r = st.iloc[i]
        ip = pre_idx[(r.question_id, int(r.agent_id))]
        row = {"question_id": r.question_id, "agent_id": int(r.agent_id), "condition": cond,
               "agreement": {1: "unanimous", 2: "two_one", 3: "three_way"}[int(n_first[r.question_id])],
               "changed": int(st.answer.iloc[ip] != r.answer), "first_correct": int(st.correct.iloc[ip]),
               "revised_correct": int(r.correct), "z_first": float(z[ip]), "z_revised": float(z[i])}
        row["delta"] = row["z_revised"] - row["z_first"]
        for L, zl in (layer_z or {}).items():
            row[f"delta_L{L}"] = float(zl[i] - zl[ip])
        out.append(row)
    return pd.DataFrame(out)


def _layer_scores(act, D):
    """Each layer's normalized score (the quantity the ensemble averages)."""
    out = {}
    for L in act.layers:
        X, present = D.X(L, act.position)
        p = act.probes[L]
        s = np.full(len(D.y), np.nan)
        s[present] = p.score(X[present], D.letters[present])
        out[L] = act.layer_norm[L](s)
    return out


def shifts():
    F, sha = _frozen()
    act, out_full = F["models"]["R1"]["act"], F["models"]["R1"]["out_lr[full]"]
    res, tables = {"frozen_sha256": sha}, []
    for cache, conds in ((VAL_CACHE, ("debate", "independent")), (SELF_CACHE, ("self_reconsider",))):
        D = CS.StateData.from_cache(cache)
        allrows = np.ones(len(D.y), bool)
        z, zo = act.score(D, allrows)[1], out_full.score(D, allrows)[1]
        lz = _layer_scores(act, D)
        for c in conds:
            t = _pairs_table(D, z, c, lz)
            t["delta_output"] = _pairs_table(D, zo, c).delta.to_numpy()
            tables.append(t)
    T = pd.concat(tables, ignore_index=True)
    T.to_csv(OUT / "shift_pairs.csv", index=False)
    # the same on the pilot's out-of-fold scores (partition 0): did the training data already show it?
    Dp = CS.StateData.from_cache(PILOT_CACHE)
    sp = pd.read_csv(ROOT / "artifacts/pilot12b/probe_validation/scores_R1.csv.gz")
    sp = sp[sp.seed == 0].set_index("row_id").loc[Dp.states.row_id]
    Tp = pd.concat([_pairs_table(Dp, sp["act:z"].to_numpy(), c) for c in Dp.conditions], ignore_index=True)
    Tp.to_csv(OUT / "shift_pairs_pilot_oof.csv", index=False)

    def summ(df, value="delta"):
        r = {}
        for c, g in df.groupby("condition"):
            r[c] = {"all": _boot_mean(g[value], g.question_id)}
            for a, h in g.groupby("agreement"):
                r[c][a] = _boot_mean(h[value], h.question_id)
                for ch, hh in h.groupby("changed"):
                    for fc, hhh in hh.groupby("first_correct"):
                        r[c][f"{a}|{'changed' if ch else 'unchanged'}|first_{'right' if fc else 'wrong'}"] = \
                            _boot_mean(hhh[value], hhh.question_id)
            dis = g[g.agreement != "unanimous"]
            un = g[g.agreement == "unanimous"]
            if len(dis) and len(un):  # disputed minus unanimous, clustered by question
                r[c]["disputed_minus_unanimous"] = _diff_boot(un, dis, value)
        return r

    res["activation_validation"] = summ(T)
    res["output_validation"] = summ(T, "delta_output")
    res["activation_pilot_oof"] = summ(Tp)
    res["by_layer_validation"] = {c: {f"L{L}": {"unanimous": _boot_mean(g[g.agreement == "unanimous"][f"delta_L{L}"],
                                                                         g[g.agreement == "unanimous"].question_id)["mean"],
                                                 "disputed": _boot_mean(g[g.agreement != "unanimous"][f"delta_L{L}"],
                                                                        g[g.agreement != "unanimous"].question_id)["mean"]}
                                       for L in act.layers}
                                  for c, g in T.groupby("condition")}
    (OUT / "shifts.json").write_text(json.dumps(res, indent=1))
    for k in ("activation_validation", "output_validation", "activation_pilot_oof"):
        print("==", k)
        for c, r in res[k].items():
            print(f"  {c:18s} all {r['all']['mean']:+.3f}  "
                  + "  ".join(f"{a} {r[a]['mean']:+.3f} [{r[a]['ci95'][0]:+.2f},{r[a]['ci95'][1]:+.2f}] (n={r[a]['n']})"
                              for a in ("unanimous", "two_one", "three_way") if a in r))
    print("== by layer (debate): unanimous / disputed",
          {k: (round(v["unanimous"], 2), round(v["disputed"], 2)) for k, v in res["by_layer_validation"]["debate"].items()})


def _diff_boot(a, b, value, n_boot=10000, seed=1):
    """mean(b) - mean(a) over two sets of questions, resampled by question."""
    qa, qb = a.question_id.to_numpy(), b.question_id.to_numpy()
    qs = np.unique(np.r_[qa, qb])
    idx = {q: i for i, q in enumerate(qs)}
    ia, ib = np.array([idx[q] for q in qa]), np.array([idx[q] for q in qb])
    W = CS.boot_weights(len(qs), n_boot, seed)
    sa = W[:, ia] @ a[value].to_numpy() / np.maximum(W[:, ia].sum(1), 1e-12)
    sb = W[:, ib] @ b[value].to_numpy() / np.maximum(W[:, ib].sum(1), 1e-12)
    d = sb - sa
    return {"value": float(d[0]), "ci95": [float(np.quantile(d[1:], 0.025)), float(np.quantile(d[1:], 0.975))]}


# ----------------------------------------------------------------------------- 3. compare


TRAIN_SETS = {"originals": ("round0",), "originals+debate": ("round0", "debate"),
              "all_pilot_conditions": ("round0", "debate", "debate_answer_only", "misleading_one", "self_reconsider")}
R1_LAYERS = "readout_last:letter:{m}:15-39"


def _scorers(D, train, seed):
    """Activation and output scorers with matched estimators: across-question vs within-question
    mean difference, on the same rows, preprocessing fitted on training rows (grouped by question)."""
    return {"act_across (R1)": CS.ActivationScorer(R1_LAYERS.format(m="meandiff"), seed=seed).fit(D, train),
            "act_within_question": CS.ActivationScorer(R1_LAYERS.format(m="wqmeandiff"), seed=seed).fit(D, train),
            "out_across": CS.TabularScorer("out_across", ("of", "ol", "ot"), "meandiff", seed=seed).fit(D, train),
            "out_within_question": CS.TabularScorer("out_within_question", ("of", "ol", "ot"), "wqmeandiff", seed=seed).fit(D, train),
            "out_lr[full]": CS.TabularScorer("out_lr[full]", ("of", "ol", "ot"), "lr", None, seed=seed).fit(D, train)}


def _evaluate(D, z, folds, cond, n_boot, seed):
    qs = np.array(sorted(set(D.qid)))
    W = CS.boot_weights(len(qs), n_boot, seed)
    e = CS.evaluate_scores(D, z, folds, cond, qs, W, {q: i for i, q in enumerate(qs)})
    dec = CS.decisions(D, cond, {"s": z})
    pk = _picks(D, z, cond)
    return {"ranking": {k: v["value"] for k, v in e["metrics"].items()},
            "ranking_ci": {k: v["ci95"] for k, v in e["metrics"].items()},
            "best_six_correct": int(dec["correct:best_six[s]"].sum()), "best_first_correct": int(dec["correct:best_original[s]"].sum()),
            "pooled6_vote": int(dec["correct:pooled6_vote"].sum()), "n_questions": int(len(dec)),
            "six_option_differs_from_first": int((pk.best_six != pk.best_original).sum())}


def compare(n_boot=2000):
    Dp = CS.StateData.from_cache(PILOT_CACHE)
    res = {"note": "exploratory: estimators and training sets chosen after inspecting the validation tranche",
           "pilot_cv": {}, "validation": {}}
    # (a) pilot, grouped 5-fold CV (one partition), evaluated on natural debate
    folds = CS.make_folds(Dp.groups, 5, 0)
    for ts, conds in TRAIN_SETS.items():
        inset = Dp.states.condition.isin(conds).to_numpy()
        z = {}
        for f in range(5):
            train, test = (folds != f) & inset, folds == f
            for n, m in _scorers(Dp, train, seed=1000 + f).items():
                z.setdefault(n, np.full(len(Dp.y), np.nan))[test] = m.score(Dp, test)[1][test]
        res["pilot_cv"][ts] = {n: _evaluate(Dp, v, folds, "debate", n_boot, 0) for n, v in z.items()}
        print("pilot CV", ts, {n: (round(r["ranking"]["six"], 3), round(r["ranking"]["revision_auc"], 3), r["best_six_correct"],
                                   r["six_option_differs_from_first"]) for n, r in res["pilot_cv"][ts].items()}, flush=True)
    # (b) fit on the whole pilot, apply to the 200 validation questions (exploratory)
    Dv = CS.StateData.from_cache(VAL_CACHE)
    allv = np.ones(len(Dv.y), bool)
    for ts, conds in TRAIN_SETS.items():
        inset = Dp.states.condition.isin(conds).to_numpy()
        ms = _scorers(Dp, inset, seed=7)
        res["validation"][ts] = {}
        for c in ("debate", "independent"):
            res["validation"][ts][c] = {n: _evaluate(Dv, m.score(Dv, allv)[1], np.zeros(len(Dv.y), int), c, n_boot, 0)
                                        for n, m in ms.items()}
        print("validation", ts, {n: (round(r["ranking"]["six"], 3), round(r["ranking"]["revision_auc"], 3), r["best_six_correct"],
                                     r["six_option_differs_from_first"]) for n, r in res["validation"][ts]["debate"].items()}, flush=True)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reproduce", "shifts", "compare", "counterfactual"])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    {"reproduce": reproduce, "shifts": shifts, "compare": lambda: compare(a.n_boot), "counterfactual": counterfactual}[a.cmd]()
    return 0



def counterfactual():
    """Does the agreement/context shift explain the unchanged selections? Remove it and recount.
    (a) per-question correction: subtract each question's mean post-minus-pre shift from its three
        revised scores (label-free, uses only that question's states);
    (b) per-agreement-group correction: subtract the group's mean shift.
    Exploratory: defined after seeing the validation scores."""
    F, sha = _frozen()
    act = F["models"]["R1"]["act"]
    res = {"frozen_sha256": sha}
    for cache, conds in ((VAL_CACHE, ("debate", "independent")), (SELF_CACHE, ("self_reconsider",))):
        D = CS.StateData.from_cache(cache)
        z = act.score(D, np.ones(len(D.y), bool))[1]
        st = D.states.reset_index(drop=True)
        for c in conds:
            t = _pairs_table(D, z, c)
            post_i = {(q, a): i for i, (q, a) in enumerate(zip(st.question_id, st.agent_id, strict=True))
                      if st.phase.iloc[i] == "post" and st.condition.iloc[i] == c}
            out = {}
            for name, shift in (("none", pd.Series(0.0, index=t.index)),
                                ("per_question_mean_shift", t.groupby("question_id").delta.transform("mean")),
                                ("per_agreement_group_shift", t.groupby("agreement").delta.transform("mean"))):
                zc = z.copy()
                for (q, a, s_) in zip(t.question_id, t.agent_id, shift, strict=True):
                    zc[post_i[(q, a)]] -= s_
                pk = _picks(D, zc, c)
                dec = CS.decisions(D, c, {"s": zc})
                ev = CS.evaluate_scores(D, zc, np.zeros(len(D.y), int), c, np.array(sorted(set(D.qid))),
                                        CS.boot_weights(len(set(D.qid)), 0, 0), {q: i for i, q in enumerate(sorted(set(D.qid)))})
                out[name] = {"six_option_differs_from_first": int((pk.best_six != pk.best_original).sum()),
                             "six_pick_is_revised_state": int(pk.six_pick_is_revised_state.sum()),
                             "best_six_correct": int(dec["correct:best_six[s]"].sum()),
                             "best_first_correct": int(dec["correct:best_original[s]"].sum()),
                             "six_auroc": ev["metrics"]["six"]["value"], "n_questions": int(len(pk))}
            # where does the max sit, by agreement (uncorrected)?
            pk = _picks(D, z, c).merge(t.groupby("question_id").agreement.first().reset_index(), on="question_id")
            out["six_pick_from_revision_by_agreement"] = pk.groupby("agreement").six_pick_is_revised_state.agg(["sum", "size"]).to_dict("index")
            res[c] = out
            print(c, {k: v for k, v in out.items()}, flush=True)
    (OUT / "counterfactual.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    raise SystemExit(main())
