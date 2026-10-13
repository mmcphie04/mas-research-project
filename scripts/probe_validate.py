"""Bounded, corrected comparison of two fixed activation recipes (CPU; the pilot pool only).

    python scripts/probe_validate.py fit      --cache runs/cache/probe12b --work runs/cache/probe12b/validate
    python scripts/probe_validate.py evaluate --cache runs/cache/probe12b --work runs/cache/probe12b/validate \\
        --out artifacts/pilot12b/probe_validation

Fixed development candidates (motivated by the earlier pilot screen; not preregistered
discoveries; no third recipe is chosen from these results):

  R1  readout_last:letter:meandiff:15-39       layers 15,18,21,24,27,30,33,36,39
  R2  before_stated_answer:letter:pca16:21-33  layers 21,24,27,30,33

Both read the state AFTER the agent has written its answer (the readout prompt contains the
response; the before-answer position is inside it, and letter centering uses the chosen letter).
They are retrospective candidate scorers, not pre-generation gates.

Scorers, all fitted per outer fold on the training questions of all conditions pooled (see
mad.analysis.candidate_scoring for the training-only preprocessing and normalizers):

  act               the activation recipe
  out_lr[of+ot]     logistic, C=1: output summaries + agreement + option text (continuity)
  out_lr[full]      logistic, C chosen from {0.001, 0.01, 0.1, 1} by grouped inner-CV log-loss:
                    output summaries + agreement + per-letter log-probabilities + option text
  out_hgb[full]     gradient boosting on the same full features (nonlinear comparator)
  comb_lr[full+act] out_lr[full]'s features plus the activation z, same C grid and rule
  eq[act+out_lr_full], eq[act+out_lr_of+ot]   post hoc equal-weight sums of training-normalized z
  act_rout          R1 only, diagnostic: activations with output features linearly regressed out
  act_fb            R2 only: activation z, or out_lr[full] z where the before-answer state is
                    absent (the all-question operational fallback)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import candidate_scoring as CS  # noqa: E402

RECIPES = {"R1": "readout_last:letter:meandiff:15-39", "R2": "before_stated_answer:letter:pca16:21-33"}
SEEDS = (0, 1, 2, 10, 11, 12)
CONDITIONS = ("debate", "debate_answer_only", "self_reconsider", "misleading_one")
PRIMARY = "debate"
K_OUTER = 5


# ----------------------------------------------------------------------------- fitting


def _scorers(D, recipe_key, train, seed, fold):
    """Fit every scorer for one outer fold. Reference folds use a seed derived from (seed, fold)."""
    rs = 1000 * seed + fold
    recipe = RECIPES[recipe_key]
    act = CS.ActivationScorer(recipe, seed=rs).fit(D, train)
    out_ofot = CS.TabularScorer("out_lr[of+ot]", ("of", "ot"), "lr", C=1.0, seed=rs).fit(D, train)
    out_full = CS.TabularScorer("out_lr[full]", ("of", "ol", "ot"), "lr", C=None, seed=rs).fit(D, train)
    out_hgb = CS.TabularScorer("out_hgb[full]", ("of", "ol", "ot"), "hgb", seed=rs).fit(D, train)
    fb = out_full if recipe_key == "R2" else None
    m = {"act": act, "out_lr[of+ot]": out_ofot, "out_lr[full]": out_full, "out_hgb[full]": out_hgb}
    m["comb_lr[full+act]"] = CS.CombinedScorer("comb_lr[full+act]", act, ("of", "ol", "ot"), C=None, seed=rs,
                                               fallback=fb).fit(D, train)
    m["eq[act+out_lr_full]"] = CS.SumScorer("eq[act+out_lr_full]", act, out_full, fb).fit(D, train)
    m["eq[act+out_lr_of+ot]"] = CS.SumScorer("eq[act+out_lr_of+ot]", act, out_ofot, fb).fit(D, train)
    if recipe_key == "R1":
        p, _, meth, layers = CS.parse_recipe(recipe)
        m["act_rout"] = CS.ActivationScorer(f"{p}:letter+rout:{meth}:{layers[0]}-{layers[-1]}", seed=rs).fit(D, train)
    else:
        m["act_fb"] = CS.FallbackScorer("act_fb", act, out_full).fit(D, train)
    return m


def _orthogonality(D, scorer, train, test):
    """How much of a scorer's held-out z is linearly predictable from the full output features:
    OLS fitted on its TRAINING reference z, R^2 on held-out rows; and the largest absolute
    held-out correlation with any single output feature."""
    M, names = D.output(("of", "ol", "ot"))
    imp = CS.Imputer().fit(M[train])
    std = CS.Standardizer().fit(imp.transform(M[train]))
    Z = np.hstack([np.ones((len(M), 1)), std.transform(imp.transform(M))])
    ref = scorer.ref_z
    a = train & np.isfinite(ref)
    B, *_ = np.linalg.lstsq(Z[a], ref[a], rcond=None)
    z = scorer.score(D, test)[1]
    b = test & np.isfinite(z)
    pred = Z[b] @ B
    r2 = 1 - np.sum((z[b] - pred) ** 2) / np.sum((z[b] - z[b].mean()) ** 2)
    Mi = imp.transform(M)[:, : M.shape[1]]
    cors = [abs(np.corrcoef(z[b], Mi[b, j])[0, 1]) for j in range(M.shape[1]) if Mi[b, j].std() > 0]
    return {"heldout_r2_from_outputs": float(r2), "max_abs_corr_with_one_output": float(np.nanmax(cors))}


def fit_task(cache, work, recipe_key, seed):
    import sklearn.decomposition  # noqa: F401
    import sklearn.ensemble  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2):
        t = time.time()
        D = CS.StateData.from_cache(cache)
        folds = CS.make_folds(D.groups, K_OUTER, seed)
        oof, params, taus, align, ortho = {}, {}, {}, {}, {}
        for f in range(K_OUTER):
            train, test = folds != f, folds == f
            m = _scorers(D, recipe_key, train, seed, f)
            for name, sc in m.items():
                raw, z, p = sc.score(D, test)
                for k, v in (("raw", raw), ("z", z), ("p", p)):
                    oof.setdefault(f"{name}|{k}", np.full(len(D.y), np.nan))[test] = v[test]
                params.setdefault(str(f), {})[name] = sc.params()
                zt = z[test & np.isfinite(z)]
                align.setdefault(str(f), {})[name] = {"heldout_z_mean": float(zt.mean()), "heldout_z_sd": float(zt.std())}
            gate_scorer = m["act_fb"] if recipe_key == "R2" else m["act"]
            taus[str(f)] = CS.tune_gate_tau(D, PRIMARY, gate_scorer.ref_z, np.unique(D.qid[train]))
            for name in ("act", "act_rout"):
                if name in m:
                    ortho.setdefault(str(f), {})[name] = _orthogonality(D, m[name], train, test)
        res = {"recipe_key": recipe_key, "recipe": RECIPES[recipe_key], "seed": seed, "folds": folds, "oof": oof,
               "params": params, "taus": taus, "alignment": align, "orthogonality": ortho,
               "secs": round(time.time() - t)}
        Path(work).mkdir(parents=True, exist_ok=True)
        with open(Path(work) / f"{recipe_key}_seed{seed}.pkl", "wb") as fh:
            pickle.dump(res, fh)
        return f"{recipe_key} seed {seed}: {res['secs']}s"


def loco_task(cache, recipe_key, seed):
    """Leave one condition out: the activation scorer is refitted without the held-out
    condition's revised states (training questions only, as always) and scored on it; revision
    AUROC within (fold) cells, against the scorer trained with every condition."""
    import sklearn.decomposition  # noqa: F401
    import sklearn.linear_model  # noqa: F401
    from threadpoolctl import threadpool_limits
    with threadpool_limits(2):
        D = CS.StateData.from_cache(cache)
        folds = CS.make_folds(D.groups, K_OUTER, seed)
        qs = np.array(sorted(set(D.qid)))
        qindex = {q: i for i, q in enumerate(qs)}
        W = CS.boot_weights(len(qs), 0, seed)
        out = {}
        for cond in D.conditions:
            held = ((D.states.condition == cond) & (D.states.phase == "post")).to_numpy()
            z_with, z_held = np.full(len(D.y), np.nan), np.full(len(D.y), np.nan)
            for f in range(K_OUTER):
                train, test = folds != f, folds == f
                rs = 1000 * seed + f
                a = CS.ActivationScorer(RECIPES[recipe_key], seed=rs).fit(D, train)
                b = CS.ActivationScorer(RECIPES[recipe_key], seed=rs).fit(D, train & ~held)
                z_with[test], z_held[test] = a.score(D, test)[1][test], b.score(D, test)[1][test]
            elig = CS.eligible_questions(D, D.present(CS.parse_recipe(RECIPES[recipe_key])[0]), cond)
            out[cond] = {k: {m: v["value"] for m, v in CS.evaluate_scores(D, z, folds, cond, elig, W, qindex)["metrics"].items()}
                         for k, z in (("trained_with", z_with), ("held_out", z_held))}
        return recipe_key, seed, out


# ----------------------------------------------------------------------------- evaluation


def _boot_diff(a, b, n_boot=10000, seed=0):
    d = (np.asarray(a) - np.asarray(b)).astype(float)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"mean": float(d.mean()), "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]}


def _paired(df, a, b, seed):
    ca, cb = df[f"correct:{a}"], df[f"correct:{b}"]
    win, loss = df.index[(ca == 1) & (cb == 0)], df.index[(ca == 0) & (cb == 1)]
    return {"a": a, "b": b, "a_correct": int(ca.sum()), "b_correct": int(cb.sum()), **_boot_diff(ca, cb, seed=seed),
            "wins": len(win), "losses": len(loss), "win_ids": list(map(str, win)), "loss_ids": list(map(str, loss))}


SELECT_SCORERS = ("act", "out_lr[full]", "out_hgb[full]", "out_lr[of+ot]", "comb_lr[full+act]",
                  "eq[act+out_lr_full]", "eq[act+out_lr_of+ot]")
MATCHED_OUTPUT = "out_lr[full]"


def decision_summary(dec: pd.DataFrame, scorers, seed) -> dict:
    out = {"n_questions": int(len(dec)),
           "counts": {c.split(":", 1)[1]: int(dec[c].sum()) for c in dec.columns if c.startswith("correct:")},
           "random_tie_expectation": {v: float(dec[f"{v}_random_tie"].sum())
                                      for v in ("original_vote", "revised_vote", "pooled6_vote")},
           "debate_introduced_correct": int(dec.debate_introduced_correct.sum()),
           "any_round0_correct": int(dec.any_round0_correct.sum()), "any_revised_correct": int(dec.any_revised_correct.sum()),
           "paired": [], "origin": {}, "gate_agents": {}}
    for s in scorers:
        P = out["paired"]
        P.append(_paired(dec, f"best_six[{s}]", "pooled6_vote", seed))
        P.append(_paired(dec, f"best_six[{s}]", f"best_original[{s}]", seed))
        P.append(_paired(dec, f"best_original[{s}]", "original_vote", seed))
        P.append(_paired(dec, f"gate0[{s}]", "original_vote", seed))
        P.append(_paired(dec, f"gate0[{s}]", "pooled6_vote", seed))
        if s != MATCHED_OUTPUT:
            for rule in ("best_six", "best_original", "gate0"):
                P.append(_paired(dec, f"{rule}[{s}]", f"{rule}[{MATCHED_OUTPUT}]", seed))
        corr = dec[f"correct:best_six[{s}]"] == 1
        origin = dec[f"best_six_origin[{s}]"]
        out["origin"][s] = {"from_revision": int((origin == "revision").sum()),
                            "new_option": int(dec[f"best_six_new_option[{s}]"].sum()),
                            "correct_and_new_option": int((corr & (dec[f"best_six_new_option[{s}]"] == 1)).sum()),
                            "correct_new_option_ids": list(map(str, dec.index[corr & (dec[f"best_six_new_option[{s}]"] == 1)]))}
        for g in ("gate0", "gate_tuned"):
            cols = [c for c in dec.columns if c.startswith(f"{g}_") and c.endswith(f"[{s}]")]
            if cols:
                out["gate_agents"][f"{g}[{s}]"] = {c.split("[")[0][len(g) + 1:]: int(dec[c].sum()) for c in cols}
    if "correct:gate_tuned[act]" in dec.columns or "correct:gate_tuned[act_fb]" in dec.columns:
        k = "act" if "correct:gate_tuned[act]" in dec.columns else "act_fb"
        out["paired"].append(_paired(dec, f"gate_tuned[{k}]", "original_vote", seed))
    return out


def evaluate(cache, work, out, n_boot):
    D = CS.StateData.from_cache(cache)
    qs = np.array(sorted(set(D.qid)))
    qindex = {q: i for i, q in enumerate(qs)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for rk in RECIPES:
        metrics, decs, score_rows, params = {}, [], [], {}
        for seed in SEEDS:
            r = pickle.load(open(Path(work) / f"{rk}_seed{seed}.pkl", "rb"))
            folds, oof = r["folds"], r["oof"]
            names = sorted({k.split("|")[0] for k in oof})
            z = {n: oof[f"{n}|z"] for n in names}
            p = {n: oof[f"{n}|p"] for n in names}
            W = CS.boot_weights(len(qs), n_boot, seed)
            present = D.present(CS.parse_recipe(RECIPES[rk])[0])
            ms = {}
            for cond in CONDITIONS:
                elig = CS.eligible_questions(D, present, cond)
                pops = {"eligible": elig}
                if rk == "R2":
                    pops["all_fallback"] = qs
                for pop, qset in pops.items():
                    key = f"{cond}|{pop}"
                    use = [n for n in names if not (pop == "all_fallback" and n == "act")]
                    if pop == "eligible":
                        use = [n for n in use if n != "act_fb"]
                    ms[key] = {"n_questions": int(len(qset)),
                               "missing_states": int((~present & CS.population_masks(D, cond)["six"]).sum()),
                               "scorers": {n: CS.evaluate_scores(D, z[n], folds, cond, qset, W, qindex) for n in use}}
                    for n in use:
                        rows = np.isin(D.qid, qset) & (D.states.phase == "post").to_numpy() & (D.states.condition == cond).to_numpy()
                        base = float(D.y[folds != 0].mean())
                        ms[key]["scorers"][n]["calibration_post"] = CS.calibration(D, p[n], rows, base)
                    sel = [s for s in SELECT_SCORERS if s in use] + (["act_fb"] if "act_fb" in use else [])
                    if pop == "all_fallback":
                        sel = [s for s in sel if s != "act"]
                    sub_z = {s: np.where(np.isin(D.qid, qset), z[s], np.nan) for s in sel}
                    gate_key = "act_fb" if pop == "all_fallback" else "act"
                    taus = None
                    dec = CS.decisions(D, cond, sub_z)
                    if gate_key in sel:  # one validation-tuned gate: per question, its fold's tau
                        tq = {q: r["taus"][str(int(folds[np.flatnonzero(D.qid == q)[0]]))] for q in qset}
                        tuned = []
                        for f in range(K_OUTER):
                            qf = [q for q in qset if folds[np.flatnonzero(D.qid == q)[0]] == f]
                            zf = {gate_key: np.where(np.isin(D.qid, qf), z[gate_key], np.nan)}
                            dfx = CS.decisions(D, cond, zf, taus={gate_key: r["taus"][str(f)]})
                            tuned.append(dfx.loc[dfx.index.isin(qf), [c for c in dfx.columns if "gate_tuned" in c]])
                        taus = tq
                        dec = dec.join(pd.concat(tuned))
                    dec = dec[dec.index.isin(qset)]
                    ms[key]["decisions"] = decision_summary(dec, [s for s in sel], seed)
                    ms[key]["gate_tau_by_fold"] = r["taus"] if taus is not None else None
                    decs.append(dec.reset_index().assign(seed=seed, condition=cond, population=pop))
            metrics[str(seed)] = {"populations": ms, "alignment": r["alignment"], "orthogonality": r["orthogonality"]}
            params[str(seed)] = {"folds": {q: int(folds[np.flatnonzero(D.qid == q)[0]]) for q in qs}, "fitted": r["params"],
                                 "gate_tau": r["taus"]}
            sc = pd.DataFrame({"row_id": D.states.row_id, "seed": seed, "fold": folds})
            for k, v in oof.items():
                sc[k.replace("|", ":")] = np.round(v, 6)
            score_rows.append(sc)
            print(f"{rk} seed {seed} evaluated", flush=True)
        (out / f"metrics_{rk}.json").write_text(json.dumps({"recipe": RECIPES[rk], "layers": CS.parse_recipe(RECIPES[rk])[3],
                                                            "n_boot": n_boot, "seeds": metrics}, indent=1, default=float))
        (out / f"params_{rk}.json").write_text(json.dumps({"recipe": RECIPES[rk], "seeds": params}, default=float))
        pd.concat(decs).to_csv(out / f"decisions_{rk}.csv.gz", index=False)
        pd.concat(score_rows).to_csv(out / f"scores_{rk}.csv.gz", index=False)
    st = D.states[["row_id", "question_id", "agent_id", "phase", "condition", "answer", "gold_display", "correct", "subject"]].copy()
    for pos in CS.POSITIONS:
        st[f"present:{pos}"] = D.present(pos).astype(int)
    st.to_csv(out / "states.csv", index=False)


METRICS = ("state_auc_pre", "state_auc_post", "panel_pre", "panel_post", "six", "revision_auc")
DIFFS = (("act", "out_lr[full]"), ("act", "out_hgb[full]"), ("comb_lr[full+act]", "out_lr[full]"),
         ("eq[act+out_lr_full]", "out_lr[full]"), ("eq[act+out_lr_of+ot]", "out_lr[of+ot]"))


def _med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (float(np.median(v)), float(min(v)), float(max(v))) if v else (np.nan, np.nan, np.nan)


def summarize_cmd(cache, work, out, n_boot):
    """Paired metric differences (same bootstrap draws as ``evaluate``) and the summary tables."""
    D = CS.StateData.from_cache(cache)
    qs = np.array(sorted(set(D.qid)))
    qindex = {q: i for i, q in enumerate(qs)}
    out = Path(out)
    summary, md = {}, []
    for rk in RECIPES:
        M = json.loads((out / f"metrics_{rk}.json").read_text())
        present = D.present(CS.parse_recipe(RECIPES[rk])[0])
        diffs = {}
        for seed in SEEDS:
            r = pickle.load(open(Path(work) / f"{rk}_seed{seed}.pkl", "rb"))
            z = {k.split("|")[0]: v for k, v in r["oof"].items() if k.endswith("|z")}
            W = CS.boot_weights(len(qs), n_boot, seed)
            for cond in CONDITIONS:
                pops = {"eligible": CS.eligible_questions(D, present, cond)}
                if rk == "R2":
                    pops["all_fallback"] = qs
                for pop, qset in pops.items():
                    vals = {}
                    for a, b in DIFFS:
                        a2 = "act_fb" if (pop == "all_fallback" and a == "act") else a
                        for n in (a2, b):
                            if n not in vals:
                                vals[n] = CS.evaluate_scores(D, z[n], r["folds"], cond, qset, W, qindex, True)["vals"]
                        diffs.setdefault(f"{cond}|{pop}", {}).setdefault(f"{a2} - {b}", {}).setdefault(str(seed), {
                            m: CS.summarize(vals[a2][m] - vals[b][m]) for m in METRICS})
        summary[rk] = {"recipe": RECIPES[rk], "layers": CS.parse_recipe(RECIPES[rk])[3], "metric_differences": diffs}
        seeds = [str(s) for s in SEEDS]
        md += [f"## {rk}: `{RECIPES[rk]}` (layers {CS.parse_recipe(RECIPES[rk])[3]})", ""]
        for key in [k for k in M["seeds"][seeds[0]]["populations"]]:
            P = {s: M["seeds"][s]["populations"][key] for s in seeds}
            md += [f"### {key} ({P[seeds[0]]['n_questions']} questions; absent states {P[seeds[0]]['missing_states']})", "",
                   "Ranking (median over 6 partitions, range):", "",
                   "| scorer | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            for sc in P[seeds[0]]["scorers"]:
                cells = []
                for m in METRICS:
                    md_, lo, hi = _med([P[s]["scorers"][sc]["metrics"][m]["value"] for s in seeds])
                    cells.append(f"{md_:.3f} ({lo:.2f}–{hi:.2f})")
                md.append(f"| {sc} | " + " | ".join(cells) + " |")
            c0 = P[seeds[0]]["scorers"].get("act", next(iter(P[seeds[0]]["scorers"].values())))["counts"]
            md += ["", "Counts (partition 0): " + "; ".join(
                f"{m} {c0[m]['pairs']} pairs" + (f" / {c0[m]['questions']} questions" if "questions" in c0[m] else "")
                for m in METRICS) + f"; helpful {c0['revision_auc']['helpful']}, harmful {c0['revision_auc']['harmful']},"
                f" wrong-to-wrong {c0['revision_auc']['wrong_to_wrong']}", "",
                "Paired differences (median over partitions; partitions whose 95% interval excludes 0, above / below):", "",
                "| difference | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
            for dk, per in diffs.get(key, {}).items():
                cells = []
                for m in METRICS:
                    v = [per[s][m]["value"] for s in seeds]
                    up = sum(per[s][m]["ci95"][0] > 0 for s in seeds)
                    dn = sum(per[s][m]["ci95"][1] < 0 for s in seeds)
                    cells.append(f"{np.median(v):+.3f} ({up}/{dn})")
                md.append(f"| {dk} | " + " | ".join(cells) + " |")
            dec = {s: P[s]["decisions"] for s in seeds}
            rules = sorted(dec[seeds[0]]["counts"])
            md += ["", "Panel decisions, questions correct per partition (original vote / pooled six do not depend on the partition):", "",
                   "| rule | " + " | ".join(seeds) + " |", "|---|" + "---|" * len(seeds)]
            for rule in rules:
                md.append(f"| {rule} | " + " | ".join(str(dec[s]["counts"].get(rule, "")) for s in seeds) + " |")
            rt = dec[seeds[0]]["random_tie_expectation"]
            md += ["", f"Random tie-break expectation: original {rt['original_vote']:.2f}, revised {rt['revised_vote']:.2f}, "
                   f"pooled six {rt['pooled6_vote']:.2f}. Questions with a correct round-0 answer {dec[seeds[0]]['any_round0_correct']}, "
                   f"with a correct revised answer {dec[seeds[0]]['any_revised_correct']}, correct only after revision "
                   f"{dec[seeds[0]]['debate_introduced_correct']}.", "",
                   "Paired decision differences (median; wins−losses per partition):", "",
                   "| comparison | median diff | wins/losses by partition |", "|---|---|---|"]
            pairs_keys = [(p_["a"], p_["b"]) for p_ in dec[seeds[0]]["paired"]]
            for a, b in pairs_keys:
                pp = [next(x for x in dec[s]["paired"] if x["a"] == a and x["b"] == b) for s in seeds]
                md.append(f"| {a} − {b} | {np.median([x['mean'] for x in pp]):+.3f} | "
                          + ", ".join(f"{x['wins']}/{x['losses']}" for x in pp) + " |")
            md += ["", "Origin of best-of-six picks (partition 0): " + "; ".join(
                f"{k}: {v['from_revision']} from revisions, {v['new_option']} new options ({v['correct_and_new_option']} correct)"
                for k, v in dec[seeds[0]]["origin"].items()), ""]
            ga = dec[seeds[0]]["gate_agents"]
            if ga:
                md += ["Gate, agent level (partition 0): " + "; ".join(
                    f"{k}: " + ", ".join(f"{kk} {vv}" for kk, vv in v.items()) for k, v in ga.items()), ""]
            cal = {sc: [P[s]["scorers"][sc]["calibration_post"] for s in seeds] for sc in P[seeds[0]]["scorers"]}
            md += ["Calibration of revised-state probabilities (median over partitions): " + "; ".join(
                f"{sc} Brier {np.median([c['brier'] for c in v]):.3f} (constant {np.median([c['constant_brier'] for c in v]):.3f}),"
                f" ECE {np.median([c['ece'] for c in v]):.3f}" for sc, v in cal.items()), ""]
        al = [M["seeds"][s]["alignment"] for s in seeds]
        sds = {sc: _med([a[f][sc]["heldout_z_sd"] for a in al for f in a]) for sc in al[0]["0"]}
        md += ["Alignment: held-out z SD (1 = reference scale), median (range) over folds × partitions: " + "; ".join(
            f"{k} {v[0]:.2f} ({v[1]:.2f}–{v[2]:.2f})" for k, v in sds.items()), ""]
        og = [M["seeds"][s]["orthogonality"] for s in seeds]
        if og and og[0]:
            for sc in og[0]["0"]:
                r2 = _med([o[f][sc]["heldout_r2_from_outputs"] for o in og for f in o])
                mc = _med([o[f][sc]["max_abs_corr_with_one_output"] for o in og for f in o])
                md.append(f"Orthogonality, {sc}: held-out R² from all output features {r2[0]:.2f} ({r2[1]:.2f}–{r2[2]:.2f}); "
                          f"largest |r| with one output feature {mc[0]:.2f} ({mc[1]:.2f}–{mc[2]:.2f}).")
            md.append("")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    (out / "summary.md").write_text("# Corrected comparison: generated summary\n\nProduced by `scripts/probe_validate.py summarize`. "
                                    "Partitions are re-splits of the same 120 questions, not replications. Intervals are "
                                    "conditional on the fitted scores and do not account for the search that chose the recipes.\n\n"
                                    + "\n".join(md) + "\n")


def freeze(cache, frozen, out):
    """Fit every scorer of both recipes on ALL pilot questions (all conditions pooled) and save
    them for application to new questions without refitting. The pickle holds fitted
    parameters (weights), so it stays local (runs/); its SHA-256 goes to the preregistration."""
    D = CS.StateData.from_cache(cache)
    train = np.ones(len(D.y), bool)
    models = {rk: _scorers(D, rk, train, seed=0, fold=0) for rk in RECIPES}
    feature_names = {rk: {n: getattr(m, "names", None) for n, m in ms.items()} for rk, ms in models.items()}
    Path(frozen).parent.mkdir(parents=True, exist_ok=True)
    with open(frozen, "wb") as fh:
        pickle.dump({"models": models, "trained_on": {"cache": str(cache), "questions": int(len(set(D.qid))),
                                                       "states": int(len(D.y)), "conditions": D.conditions},
                     "feature_names": feature_names, "pilot_ref_z": {rk: {n: m.ref_z for n, m in ms.items()}
                                                                     for rk, ms in models.items()}}, fh)
    h = hashlib.sha256(Path(frozen).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    rec = {"frozen_file": str(frozen), "sha256": h, "code_commit": commit, "code_dirty": dirty,
           "command": "python scripts/probe_validate.py freeze", "trained_on_questions": int(len(set(D.qid))),
           "trained_on_states": int(len(D.y)), "conditions": D.conditions,
           "recipes": {rk: {"recipe": RECIPES[rk], "layers": CS.parse_recipe(RECIPES[rk])[3]} for rk in RECIPES},
           "params": {rk: {n: m.params() for n, m in ms.items()} for rk, ms in models.items()}}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "frozen_scorers.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"frozen scorers: {frozen} sha256 {h}")


def apply_frozen(frozen, cache, out, primary="debate", n_boot=2000):
    """Score a NEW archive with frozen scorers (one fitted model: no cross-fold issue) and
    evaluate: ranking metrics and decisions per condition, by question bootstrap. Refuses if
    the output-feature columns differ from those the scorers were trained on."""
    F = pickle.load(open(frozen, "rb"))
    D = CS.StateData.from_cache(cache)
    qs = np.array(sorted(set(D.qid)))
    qindex = {q: i for i, q in enumerate(qs)}
    W = CS.boot_weights(len(qs), n_boot, 0)
    folds = np.zeros(len(D.y), int)
    allrows = np.ones(len(D.y), bool)
    report = {"frozen_sha256": hashlib.sha256(Path(frozen).read_bytes()).hexdigest(), "conditions": D.conditions,
              "n_questions": int(len(qs)), "recipes": {}}
    for rk, models in F["models"].items():
        for n, m in models.items():
            want = F["feature_names"][rk][n]
            if want is not None and m.features(D)[1] != want:
                raise SystemExit(f"{rk}/{n}: output feature columns differ from training; refusing to apply")
        z = {n: m.score(D, allrows)[1] for n, m in models.items()}
        present = D.present(CS.parse_recipe(RECIPES[rk])[0])
        rep = {}
        for cond in D.conditions:
            elig = CS.eligible_questions(D, present, cond)
            names = [n for n in z if n != "act_fb"]
            sel = [n for n in SELECT_SCORERS if n in z]
            dec = CS.decisions(D, cond, {n: np.where(np.isin(D.qid, elig), z[n], np.nan) for n in sel})
            dec = dec[dec.index.isin(elig)]
            rep[cond] = {"n_questions": int(len(elig)),
                         "scorers": {n: CS.evaluate_scores(D, z[n], folds, cond, elig, W, qindex) for n in names},
                         "decisions": decision_summary(dec, sel, 0)}
            # distribution shift: frozen probabilities vs outcomes (calibration-in-the-large)
            rows = np.isin(D.qid, elig) & (D.states.condition == cond).to_numpy()
            rep[cond]["calibration_in_the_large"] = {
                n: {"mean_p": float(np.nanmean(models[n].score(D, rows)[2][rows])), "observed": float(D.y[rows].mean())}
                for n in names}
        report["recipes"][rk] = rep
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "frozen_application.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"applied {frozen} to {cache}: {len(qs)} questions, conditions {D.conditions}")


def provenance(cache, out, argv):
    cache, out = Path(cache), Path(out)
    meta = json.loads((cache / "meta.json").read_text())
    root, prefix = Path(meta["root"]), meta["prefix"]

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            while b := fh.read(1 << 22):
                h.update(b)
        return h.hexdigest()

    layers = sorted({L for r in RECIPES.values() for L in CS.parse_recipe(r)[3]})
    import sklearn
    rec = {"code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "code_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()),
           "command": " ".join(argv),
           "packages": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit-learn": sklearn.__version__},
           "archive": {"root": str(root), "prefix": prefix,
                       "verify_json_sha256": sha(root / f"{prefix}-verify.json"),
                       "questions_sha256": sha(root / f"{prefix}-questions.jsonl"),
                       "chunk_completion_sha256": {p.parent.name: sha(p) for p in sorted(root.glob(f"{prefix}-*-c*/completion.json"))}},
           "cache": {"keys.parquet": sha(cache / "keys.parquet"), "states.parquet": sha(cache / "states.parquet"),
                     **{f"L{L:02d}.npy": sha(cache / f"L{L:02d}.npy") for L in layers}},
           "recipes": {k: {"recipe": v, "layers": CS.parse_recipe(v)[3]} for k, v in RECIPES.items()},
           "seeds": list(SEEDS), "k_outer": K_OUTER, "k_ref": 5, "C_grid": list(CS.C_GRID)}
    (out / "provenance.json").write_text(json.dumps(rec, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "evaluate", "summarize", "loco", "freeze", "apply", "provenance"])
    ap.add_argument("--frozen", default="runs/frozen/pilot12b_scorers.pkl")
    ap.add_argument("--cache", default="runs/cache/probe12b")
    ap.add_argument("--work", default="runs/cache/probe12b/validate")
    ap.add_argument("--out", default="artifacts/pilot12b/probe_validation")
    ap.add_argument("--recipes", nargs="+", default=list(RECIPES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        from joblib import Parallel, delayed
        for msg in Parallel(n_jobs=args.jobs)(delayed(fit_task)(args.cache, args.work, rk, s)
                                              for rk in args.recipes for s in args.seeds):
            print(msg, flush=True)
    elif args.cmd == "evaluate":
        evaluate(args.cache, args.work, args.out, args.n_boot)
    elif args.cmd == "summarize":
        summarize_cmd(args.cache, args.work, args.out, args.n_boot)
    elif args.cmd == "loco":
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.jobs)(delayed(loco_task)(args.cache, rk, s) for rk in args.recipes for s in args.seeds)
        rep = {}
        for rk, seed, o in res:
            rep.setdefault(rk, {})[str(seed)] = o
        Path(args.out, "leave_condition_out.json").write_text(json.dumps(rep, indent=1))
        for rk, per in rep.items():
            for cond in next(iter(per.values())):
                w = [per[s][cond]["trained_with"]["revision_auc"] for s in per]
                h = [per[s][cond]["held_out"]["revision_auc"] for s in per]
                print(f"{rk} {cond:20s} revision AUROC with {np.median(w):.3f} ({min(w):.2f}-{max(w):.2f})  "
                      f"held out {np.median(h):.3f} ({min(h):.2f}-{max(h):.2f})", flush=True)
    elif args.cmd == "freeze":
        freeze(args.cache, args.frozen, args.out)
    elif args.cmd == "apply":
        apply_frozen(args.frozen, args.cache, args.out, n_boot=args.n_boot)
    else:
        provenance(args.cache, args.out, sys.argv if argv is None else ["probe_validate.py", *argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
