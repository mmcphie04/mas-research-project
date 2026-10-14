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


def _scorers(D, recipe_key, train, seed, fold, recipes=None):
    """Fit every scorer for one outer fold. Reference folds use a seed derived from (seed, fold)."""
    rs = 1000 * seed + fold
    recipe = (recipes or RECIPES)[recipe_key]
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
                    v = [x for x in (per[s][m]["value"] for s in seeds) if x is not None]
                    up = sum((per[s][m]["ci95"][0] or 0) > 0 for s in seeds)
                    dn = sum((per[s][m]["ci95"][1] or 0) < 0 for s in seeds)
                    cells.append(f"{np.median(v):+.3f} ({up}/{dn})" if v else "n/a")
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


def freeze(cache, frozen, out, recipes=None):
    """Fit every scorer of both fixed recipes on ALL pilot questions (all conditions pooled) and
    save them for application to new questions without refitting. The pickle holds fitted
    parameters (weights), so it stays local (runs/); its SHA-256 goes to the preregistration.
    Nothing in a frozen scorer's prediction path refers to the training rows."""
    recipes = recipes or RECIPES
    D = CS.StateData.from_cache(cache) if not isinstance(cache, CS.StateData) else cache
    train = np.ones(len(D.y), bool)
    models = {rk: _scorers(D, rk, train, seed=0, fold=0, recipes=recipes) for rk in recipes}
    schema = {rk: {n: (m.schema(D) if hasattr(m, "schema") else None) for n, m in ms.items()} for rk, ms in models.items()}
    params = {rk: {n: m.params() for n, m in ms.items()} for rk, ms in models.items()}
    for ms in models.values():  # drop training-length arrays: the prediction path must not need them
        for m in ms.values():
            for attr in ("ref_z", "train"):
                if hasattr(m, attr):
                    setattr(m, attr, None)
    Path(frozen).parent.mkdir(parents=True, exist_ok=True)
    with open(frozen, "wb") as fh:
        pickle.dump({"models": models, "recipes": dict(recipes), "feature_names": schema,
                     "train_base_rate": float(D.y.mean()),
                     "trained_on": {"questions": int(len(set(D.qid))), "states": int(len(D.y)),
                                    "conditions": D.conditions}}, fh)
    h = hashlib.sha256(Path(frozen).read_bytes()).hexdigest()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip())
    rec = {"frozen_file": str(frozen), "sha256": h, "code_commit": commit, "code_dirty": dirty,
           "command": "python scripts/probe_validate.py freeze", "trained_on_questions": int(len(set(D.qid))),
           "trained_on_states": int(len(D.y)), "train_base_rate": float(D.y.mean()), "conditions": D.conditions,
           "recipes": {rk: {"recipe": r, "layers": CS.parse_recipe(r)[3]} for rk, r in recipes.items()},
           "feature_names": schema, "params": params}
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "frozen_scorers.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"frozen scorers: {frozen} sha256 {h}")
    return h


TARGETS = {"selection_accuracy": 0.02, "revision_auc": 0.05}


def classify(diff: dict, target: float, n: int, secs_per_question: float | None = None) -> dict:
    """Development reading of one paired difference against a practical target (not a powered
    test): promising (point estimate reaches the target; 'demonstrated' only if the interval is
    above 0), uncompetitive (the interval's upper end is below the target), else inconclusive --
    with the fresh sample needed for an interval half-width of target/2 (n scales as 1/hw^2)."""
    v, (lo, hi) = diff.get("value", diff.get("mean")), diff["ci95"]
    if v is None or lo is None:
        return {"reading": "undefined (no comparisons)"}
    if v >= target:
        reading = "promising: interval above 0" if lo > 0 else "promising, provisional: interval includes 0"
    elif hi < target:
        reading = "uncompetitive at the target"
    else:
        reading = "inconclusive"
    out = {"reading": reading, "value": v, "ci95": [lo, hi], "target": target}
    if reading == "inconclusive":
        hw = (hi - lo) / 2
        need = int(np.ceil(n * (hw / (target / 2)) ** 2))
        out["questions_needed_for_halfwidth_target_over_2"] = need
        if secs_per_question:
            out["projected_raw_gpu_hours"] = round(need * secs_per_question / 3600, 1)
    return out


def _dict_nan_to_none(x):
    if isinstance(x, dict):
        return {k: _dict_nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_dict_nan_to_none(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _source_comparison(D, a_cond, b_cond, z_by, questions, seed=0):
    """Paired comparison of two six-answer pools that share the same first three answers
    (e.g. debate vs independent), on identical questions: coverage, pooled-six vote, best of
    six and best first answer by each scorer, per question and as paired differences."""
    da = CS.decisions(D, a_cond, z_by)
    db = CS.decisions(D, b_cond, z_by)
    q = sorted(set(questions) & set(da.index) & set(db.index))
    da, db = da.loc[q], db.loc[q]
    per = pd.DataFrame(index=pd.Index(q, name="question_id"))
    per["any_round0_correct"] = da.any_round0_correct
    per[f"coverage_six:{a_cond}"] = ((da.any_round0_correct + da.any_revised_correct) > 0).astype(int)
    per[f"coverage_six:{b_cond}"] = ((db.any_round0_correct + db.any_revised_correct) > 0).astype(int)
    per[f"new_correct_option:{a_cond}"] = da.debate_introduced_correct
    per[f"new_correct_option:{b_cond}"] = db.debate_introduced_correct
    per[f"pooled6_vote:{a_cond}"] = da["correct:pooled6_vote"]
    per[f"pooled6_vote:{b_cond}"] = db["correct:pooled6_vote"]
    per["original_vote"] = da["correct:original_vote"]
    rows = {"n_questions": len(q), "comparisons": []}
    comps = [("coverage_six", f"coverage_six:{a_cond}", f"coverage_six:{b_cond}"),
             ("pooled6_vote", f"pooled6_vote:{a_cond}", f"pooled6_vote:{b_cond}")]
    for s in z_by:
        per[f"best_six[{s}]:{a_cond}"] = da[f"correct:best_six[{s}]"]
        per[f"best_six[{s}]:{b_cond}"] = db[f"correct:best_six[{s}]"]
        per[f"best_original[{s}]"] = da[f"correct:best_original[{s}]"]
        comps.append((f"best_six[{s}]", f"best_six[{s}]:{a_cond}", f"best_six[{s}]:{b_cond}"))
        for c in (a_cond, b_cond):
            comps.append((f"best_six[{s}]:{c} - best_original[{s}]", f"best_six[{s}]:{c}", f"best_original[{s}]"))
    for label, x, y in comps:
        d = (per[x] - per[y]).to_numpy(float)
        win, loss = per.index[d > 0], per.index[d < 0]
        rows["comparisons"].append({"comparison": label, "a": x, "b": y, "a_count": int(per[x].sum()),
                                    "b_count": int(per[y].sum()), **_boot_diff(per[x], per[y], seed=seed),
                                    "wins": len(win), "losses": len(loss),
                                    "win_ids": list(map(str, win)), "loss_ids": list(map(str, loss))})
    return per, rows


def _ops(D):
    """Missingness, parse and cap rates per (condition, phase): operational, outcome-free."""
    st = D.states
    out = {}
    for (c, ph), g in st.groupby(["condition", "phase"]):
        idx = g.index.to_numpy()
        out[f"{c}|{ph}"] = {"states": int(len(g)),
                            "answer_fallback_rate": float((g.answer_source != "stated").mean()),
                            "capped_rate": float((g.stop_reason == "length").mean()),
                            "absent_by_position": {pos: int((~D.present(pos)[idx]).sum()) for pos in CS.POSITIONS}}
    return out


def apply_frozen_data(F, D, out, n_boot=10000, sha=None, pilot_metrics=None, secs_per_question=None):
    """Score NEW states with frozen scorers and run the frozen validation analysis
    (docs/preregistration.md section 10). One fitted model per scorer, so every comparison is
    within that model; bootstrap by question with shared draws for paired differences."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for rk, models in F["models"].items():  # schema only: no feature columns are built here
        for n, m in models.items():
            want = F["feature_names"][rk].get(n)
            if want is not None and m.schema(D) != want:
                raise SystemExit(f"{rk}/{n}: output feature columns differ from training; refusing to apply")
    qs = np.array(sorted(set(D.qid)))
    qindex = {q: i for i, q in enumerate(qs)}
    W = CS.boot_weights(len(qs), n_boot, 0)
    folds = np.zeros(len(D.y), int)
    allrows = np.ones(len(D.y), bool)
    st = D.states.copy()
    states = st[["row_id", "question_id", "agent_id", "phase", "condition", "answer", "gold_display", "correct",
                 "subject", "answer_source", "stop_reason"]].copy()
    for pos in CS.POSITIONS:
        states[f"present:{pos}"] = D.present(pos).astype(int)
    states.to_csv(out / "states.csv", index=False)
    report = {"frozen_sha256": sha, "conditions": D.conditions, "n_questions": int(len(qs)), "n_boot": n_boot,
              "note_independent": "independent 'revised' states are fresh samples of the original prompt: "
                                  "its old/new transitions are resampling comparisons, not persuasion; its "
                                  "'peer_exposure' state is an original-prompt state (no peer shown).",
              "operational": _ops(D), "recipes": {}}
    for rk, models in F["models"].items():
        act = models["act"]
        z, sc = {}, pd.DataFrame({"row_id": st.row_id})
        for n, m in models.items():
            raw, zz, p = m.score(D, allrows)
            z[n] = zz
            sc[f"{n}:raw"], sc[f"{n}:z"], sc[f"{n}:p"] = np.round(raw, 6), np.round(zz, 6), np.round(p, 6)
        sc.to_csv(out / f"scores_{rk}.csv.gz", index=False)
        present = D.present(act.position)
        has_missing = bool((~present).any())
        rep = {"recipe": act.recipe, "layers": act.layers, "position": act.position, "conditions": {}}
        decs = []
        for cond in D.conditions:
            pops = {"complete_case": CS.eligible_questions(D, present, cond)}
            if has_missing and "act_fb" in z:
                pops["all_fallback"] = qs
            for pop, qset in pops.items():
                act_name = "act_fb" if pop == "all_fallback" else "act"
                names = [n for n in z if n not in (("act",) if pop == "all_fallback" else ("act_fb",))]
                ev = {n: CS.evaluate_scores(D, z[n], folds, cond, qset, W, qindex, return_vals=True) for n in names}
                diffs = {}
                for a_, b_ in DIFFS:
                    a2 = act_name if a_ == "act" else a_
                    if a2 in ev and b_ in ev:
                        diffs[f"{a2} - {b_}"] = {mm: CS.summarize(ev[a2]["vals"][mm] - ev[b_]["vals"][mm]) for mm in METRICS}
                sel = [s_ for s_ in SELECT_SCORERS if s_ in z and not (pop == "all_fallback" and s_ == "act")]
                if pop == "all_fallback":
                    sel = ["act_fb", *sel]
                zsub = {s_: np.where(np.isin(D.qid, qset), z[s_], np.nan) for s_ in sel}
                dec = CS.decisions(D, cond, zsub)
                dec = dec[dec.index.isin(qset)]
                dsum = decision_summary(dec, sel, 0) if len(dec) else {"n_questions": 0}
                calib = {}
                for n in names:
                    rows = np.isin(D.qid, qset) & (st.condition == cond).to_numpy()
                    p_ = models[n].score(D, rows)[2]
                    calib[n] = {**CS.calibration(D, p_, rows, base_rate=F.get("train_base_rate", 0.5)),
                                "mean_p": float(np.nanmean(p_[rows])),
                                "observed": float(D.y[rows].mean())} if rows.any() else None
                rep["conditions"][f"{cond}|{pop}"] = {
                    "n_questions": int(len(qset)),
                    "metrics": {n: {"metrics": e["metrics"], "counts": e["counts"]} for n, e in ev.items()},
                    "differences": diffs, "decisions": dsum, "calibration": calib}
                decs.append(dec.reset_index().assign(condition=cond, population=pop))
        pd.concat(decs).to_csv(out / f"decisions_{rk}.csv", index=False)
        # candidate source: debate vs independent on identical questions
        if {"debate", "independent"} <= set(D.conditions):
            src = {}
            for pop in (["complete_case", "all_fallback"] if has_missing and "act_fb" in z else ["complete_case"]):
                if pop == "complete_case":
                    qset = np.intersect1d(CS.eligible_questions(D, present, "debate"),
                                          CS.eligible_questions(D, present, "independent"))
                    sel = [s_ for s_ in SELECT_SCORERS if s_ in z]
                else:
                    qset, sel = qs, ["act_fb", *[s_ for s_ in SELECT_SCORERS if s_ in z and s_ != "act"]]
                per, rows_ = _source_comparison(D, "debate", "independent",
                                                {s_: np.where(np.isin(D.qid, qset), z[s_], np.nan) for s_ in sel}, qset)
                per.to_csv(out / f"source_comparison_{rk}_{pop}.csv")
                src[pop] = rows_
            rep["source_comparison"] = src
        # development readings of the primary and key comparisons (debate)
        key = "debate|complete_case"
        if key in rep["conditions"]:
            C_ = rep["conditions"][key]
            n_q = C_["n_questions"]
            pr = {}
            for p_ in C_["decisions"].get("paired", []):
                if p_["a"] == f"best_six[{'act'}]" and p_["b"] in (f"best_six[{MATCHED_OUTPUT}]", "pooled6_vote"):
                    pr[f"{p_['a']} - {p_['b']}"] = classify(p_, TARGETS["selection_accuracy"], n_q, secs_per_question)
            d = C_["differences"].get(f"act - {MATCHED_OUTPUT}", {})
            for mm in ("revision_auc", "panel_post"):
                if mm in d:
                    pr[f"act - {MATCHED_OUTPUT} | {mm}"] = classify(d[mm], TARGETS["revision_auc"], n_q, secs_per_question)
            rep["readings"] = pr
        # shift: round-0 ranking on the new questions vs the pilot's out-of-fold values
        if pilot_metrics and rk in pilot_metrics:
            r0 = rep["conditions"].get("debate|complete_case", {}).get("metrics", {})
            rep["shift"] = {n: {"state_auc_pre_new": r0.get(n, {}).get("metrics", {}).get("state_auc_pre", {}).get("value"),
                                "state_auc_pre_pilot_oof_median": pilot_metrics[rk].get(n)} for n in ("act", MATCHED_OUTPUT)}
        report["recipes"][rk] = rep
    (out / "frozen_application.json").write_text(json.dumps(_dict_nan_to_none(report), indent=1, default=float))
    return report


def apply_frozen(frozen, cache, out, n_boot=10000, pilot_dir="artifacts/pilot12b/probe_validation",
                 secs_per_question=None):
    F = pickle.load(open(frozen, "rb"))
    D = CS.StateData.from_cache(cache)
    pm = {}
    for rk in F["models"]:
        f = Path(pilot_dir) / f"metrics_{rk}.json"
        if f.exists():
            M = json.loads(f.read_text())
            pm[rk] = {n: float(np.median([M["seeds"][s]["populations"]["debate|eligible"]["scorers"][n]["metrics"]
                                          ["state_auc_pre"]["value"] for s in M["seeds"]]))
                      for n in ("act", MATCHED_OUTPUT)}
    sha = hashlib.sha256(Path(frozen).read_bytes()).hexdigest()
    apply_frozen_data(F, D, out, n_boot=n_boot, sha=sha, pilot_metrics=pm, secs_per_question=secs_per_question)
    print(f"applied {frozen} (sha256 {sha[:12]}) to {cache}: {len(set(D.qid))} questions, conditions {D.conditions}")


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
    ap.add_argument("--secs-per-question", type=float, default=None,
                    help="apply: measured GPU seconds per question, for sample-size cost projections")
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
        apply_frozen(args.frozen, args.cache, args.out, n_boot=args.n_boot, secs_per_question=args.secs_per_question)
    else:
        provenance(args.cache, args.out, sys.argv if argv is None else ["probe_validate.py", *argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
