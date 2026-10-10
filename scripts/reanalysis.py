"""Revision-gate analysis: do activations add to output-only evidence, at a given information time?

Exploratory cross-validation (repeated grouped 5-fold over questions; seeds = sensitivity checks):
    uv run python scripts/reanalysis.py --run runs/night2-debate --out artifacts/review/reanalysis_v2

Preregistered route (frozen split manifest from scripts/make_split.py; test scored once):
    uv run python scripts/reanalysis.py --run RUN --mode split --split-manifest split.json --out OUT

Every choice -- layer, regularisation, tree settings, calibration, threshold -- is made from
training questions (cross-fitted) or validation questions (split mode). The threshold is chosen
on the same calibrated score used at evaluation, among candidates that include accept-all and
reject-all, maximising PANEL accuracy under the declared tie-break (primary) or agent accuracy
(secondary). See mad.analysis.gates.

Families: output_linear, output_hgb (output-only); activation; combined (activations + output
features in one model); stacked (a cross-fitted logistic combination of the output_linear and
activation scores). Output features and activation phases are restricted to what exists at
``--time`` (mad.analysis.features).

Exports: results.json (metrics, intervals, feature names), predictions.csv (per row: raw and
calibrated probabilities, thresholds, decisions, answer identities), panel.csv (per question:
panel answer and correctness under every gate and reference policy), choices.json (every chosen
parameter, calibrator coefficients, thresholds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import gates as G  # noqa: E402
from mad.analysis.calibrate import PlattScaler, expected_calibration_error  # noqa: E402
from mad.analysis.features import ACTIVATION_PHASES, load_events, output_features  # noqa: E402
from mad.analysis.labels import apply_policy  # noqa: E402
from mad.analysis.oracles import panel_optimal_accept, policy_accept  # noqa: E402
from mad.analysis.states import StateBank  # noqa: E402
from mad.analysis.stats import (  # noqa: E402
    bootstrap_auroc,
    bootstrap_delta_auroc,
    conditional_auroc,
)

LAYER_FRACS = (0.25, 0.5, 0.65, 0.8, 0.92, 1.0)
ALL = ("output_linear", "output_hgb", "activation", "combined", "stacked")


# ----------------------------------------------------------------------------- data


def prepare(run_dir, time, position, exposure_position, families):
    """Events and per-family feature sources at one information time. Questions with any
    missing required state are dropped whole (so panel votes stay complete) and counted."""
    ev, _ = load_events(run_dir)
    Xout, names = output_features(ev, time)
    report = {"events_total": int(len(ev)), "lp_source": ev.attrs["lp_source"], "time": time,
              "output_features": names}
    need_act = any(f in ("activation", "combined", "stacked") for f in families)
    bank = StateBank(run_dir) if need_act else None
    keep = np.ones(len(ev), bool)
    if need_act:
        for ph in ACTIVATION_PHASES[time]:
            pos = exposure_position if ph == "peer_exposure" else position
            _, present = bank.rows(ev, ph, pos, missing="mask")
            keep &= present
        drop_q = set(ev.question_id[~keep])
        keep = ~ev.question_id.isin(drop_q).to_numpy()
        report["questions_dropped_missing_states"] = sorted(drop_q)
    attrs = dict(ev.attrs)
    ev = ev[keep].reset_index(drop=True)
    ev.attrs.update(attrs)
    Xout = Xout[keep]
    cache: dict[int, np.ndarray] = {}

    def act(layer):
        if layer not in cache:
            parts = []
            for ph in ACTIVATION_PHASES[time]:
                pos = exposure_position if ph == "peer_exposure" else position
                X, _ = bank.matrix(ev, ph, pos, layer)
                parts.append(X.astype(np.float32))
            cache[layer] = np.hstack(parts)
        return cache[layer]

    layers = sorted({int(bank.n_layers * f) for f in LAYER_FRACS}) if need_act else []
    feats = {fam: G.Features(fam, output=Xout, act=act, layers=layers)
             for fam in families if fam != "stacked"}
    report.update({"events_used": int(len(ev)), "questions_used": int(ev.question_id.nunique()),
                   "revised_used": int(ev.revised.sum()), "position": position,
                   "exposure_position": exposure_position if time == "exposure" else None,
                   "activation_phases": list(ACTIVATION_PHASES[time]) if need_act else [],
                   "candidate_layers": layers})
    return ev, feats, report


# ----------------------------------------------------------------------------- stacking


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Stacked:
    """Logistic combination of two base families' scores, cross-fitted on selection rows."""

    BASES = ("output_linear", "activation")

    def fit(self, base_sel: dict, y: dict, folds: np.ndarray, ev_sel, tie_break):
        self.stackers, self.calibrators, sel_cal = {}, {}, {}
        for t in ("old", "new"):
            Z = np.column_stack([_logit(base_sel[b][t]) for b in self.BASES])
            oof = np.full(len(Z), np.nan)
            for f in np.unique(folds):
                m = folds == f
                oof[m] = G.linear(1.0).fit(Z[~m], y[t][~m]).predict_proba(Z[m])[:, 1]
            sel_cal[t] = G.crossfit_calibrate(oof, y[t], folds)
            self.calibrators[t] = PlattScaler().fit(oof, y[t])
            self.stackers[t] = G.linear(1.0).fit(Z, y[t])
        self.tau = {o: G.choose_tau(ev_sel, sel_cal["new"], sel_cal["old"], o, tie_break)[0]
                    for o in G.OBJECTIVES}
        self.params = {t: {"weights": [float(w) for w in self.stackers[t][-1].coef_[0]]} for t in y}
        return self

    def score(self, base_raw: dict) -> dict:
        out = {}
        for t in ("old", "new"):
            Z = np.column_stack([_logit(base_raw[b][t]) for b in self.BASES])
            raw = self.stackers[t].predict_proba(Z)[:, 1]
            out[f"{t}_raw"], out[f"{t}_cal"] = raw, self.calibrators[t].transform(raw)
        return out


def _inner_folds(ev_sel, k=5):
    from sklearn.model_selection import GroupKFold

    fold = np.full(len(ev_sel), -1)
    g = ev_sel.question_id.to_numpy()
    for f, (_, b) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(g)), groups=g)):
        fold[b] = f
    return fold


# ----------------------------------------------------------------------------- fitting


def fit_all(ev, feats, families, sel, tie_break, mode, va=None):
    """Fit every family on selection rows ``sel`` (train; validation ``va`` in split mode)."""
    fitted = {}
    for fam in families:
        if fam != "stacked":
            fitted[fam] = (G.fit_crossfit(feats[fam], ev, sel, tie_break) if mode == "cv"
                           else G.fit_validation(feats[fam], ev, sel, va, tie_break))
    if "stacked" in families:
        rows = sel if mode == "cv" else va
        ev_s = ev[rows].reset_index(drop=True)
        key = "oof_raw" if mode == "cv" else "val_raw"
        base = {b: fitted[b][key] for b in Stacked.BASES}
        y = {"old": ev.pre_correct.to_numpy()[rows].astype(int),
             "new": ev.post_correct.to_numpy()[rows].astype(int)}
        folds = fitted["output_linear"]["oof_fold"] if mode == "cv" else _inner_folds(ev_s)
        fitted["stacked"] = Stacked().fit(base, y, folds, ev_s, tie_break)
    return fitted


def score_all(fitted, feats, rows):
    out = {fam: G.score(f, feats[fam], rows) for fam, f in fitted.items() if fam != "stacked"}
    if "stacked" in fitted:
        base = {b: {"old": out[b]["old_raw"], "new": out[b]["new_raw"]} for b in Stacked.BASES}
        out["stacked"] = fitted["stacked"].score(base)
    return out


def tau_of(f):
    return f.tau if isinstance(f, Stacked) else f["tau"]


def choices_of(f):
    params = f.params if isinstance(f, Stacked) else f["params"]
    cals = f.calibrators if isinstance(f, Stacked) else f["calibrators"]
    return {"params": params, "tau": tau_of(f),
            "calibrators": {t: G.calibrator_params(c) for t, c in cals.items()}}


# ----------------------------------------------------------------------------- evaluation


def panel_table(ev, accept, tie_break):
    return apply_policy(ev, accept, tie_break).set_index("question_id")


def paired_panel(ev, a, b, tie_break, n_boot=2000, seed=0):
    pa = panel_table(ev, a, tie_break).panel_correct
    pb = panel_table(ev, b, tie_break).panel_correct.reindex(pa.index)
    d = (pa - pb).to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "ci95": [float(x) for x in np.quantile(bs, [0.025, 0.975])],
            "n_questions": int(len(d))}


def evaluate(ev, P, tie_break, families, seed):
    """Metrics on held-out rows. ``P[fam]`` holds old/new raw/cal and accept_<objective>.
    Agent-level intervals resample QUESTIONS (agents of one question are correlated)."""
    y_old, y_new = ev.pre_correct.to_numpy().astype(int), ev.post_correct.to_numpy().astype(int)
    g = ev.question_id.to_numpy()
    rev = (ev.revised == 1).to_numpy()
    lab = ev.label.to_numpy()
    res = {"families": {}}
    for fam in families:
        p = P[fam]
        fr = {}
        for t, y in (("old", y_old), ("new", y_new)):
            s = p[f"{t}_cal"]
            c, npair, _ = conditional_auroc(y, s, g)
            ci_all = bootstrap_auroc(y, s, g, seed=seed)
            ci_rev = bootstrap_auroc(y[rev], s[rev], g[rev], seed=seed)
            fr[f"{t}_correct"] = {
                "auroc_all": ci_all.point, "auroc_all_ci95": [ci_all.lo, ci_all.hi],
                "auroc_revised": ci_rev.point, "auroc_revised_ci95": [ci_rev.lo, ci_rev.hi],
                "auroc_within_question": c, "within_question_pairs": npair,
                "brier": brier_score_loss(y, s),
                "logloss": log_loss(y, np.clip(s, 1e-6, 1 - 1e-6), labels=[0, 1]),
                "ece": expected_calibration_error(s, y), "sd": float(np.std(s)),
            }
        for o in G.OBJECTIVES:
            acc = p[f"accept_{o}"]
            fr[f"gate_{o}"] = {
                "agent_accuracy": float(np.mean(np.where(acc, y_new, y_old))),
                "panel_accuracy": float(panel_table(ev, acc, tie_break).panel_correct.mean()),
                "beneficial_kept": int((acc & (lab == "beneficial")).sum()),
                "beneficial_lost": int((~acc & (lab == "beneficial")).sum()),
                "harmful_blocked": int((~acc & (lab == "harmful")).sum()),
                "harmful_accepted": int((acc & (lab == "harmful")).sum()),
                "unhelpful_accepted": int((acc & (lab == "unhelpful")).sum()),
                "revisions_accepted": int((acc & rev).sum()), "revisions": int(rev.sum()),
            }
        res["families"][fam] = fr
    ref = {"accept_all": policy_accept(ev, "mad_standard"),
           "reject_all": policy_accept(ev, "round0_plurality"),
           "agent_optimal_not_a_panel_bound": policy_accept(ev, "oracle_full"),
           "panel_optimal": panel_optimal_accept(ev, tie_break)}
    res["reference"] = {k: {"agent_accuracy": float(np.mean(np.where(v, y_new, y_old))),
                            "panel_accuracy": float(panel_table(ev, v, tie_break).panel_correct.mean())}
                        for k, v in ref.items()}
    gates = {**{f: P[f]["accept_panel"] for f in families}, **ref}
    pairs = [(a, b) for a in ("activation", "combined", "stacked") for b in ("output_linear", "output_hgb")]
    pairs += [(f, r) for f in families for r in ("accept_all", "reject_all")]
    res["paired_panel"] = {f"{a} - {b}": paired_panel(ev, gates[a], gates[b], tie_break, seed=seed)
                           for a, b in pairs if a in gates and b in gates}
    res["auroc_differences"] = {}
    for a in ("activation", "combined", "stacked"):
        if a in P and "output_linear" in P:
            for subset, m in (("all", np.ones(len(ev), bool)), ("revised", rev)):
                for t, y in (("new", y_new), ("old", y_old)):
                    d = bootstrap_delta_auroc(y[m], P[a][f"{t}_cal"][m], P["output_linear"][f"{t}_cal"][m],
                                              g[m], seed=seed)
                    res["auroc_differences"][f"{a} - output_linear | {t}_correct | {subset}"] = {
                        "point": d.point, "ci95": [d.lo, d.hi]}
    return res, ref


ID_COLS = ["condition", "question_id", "agent_id", "pre_answer", "post_answer", "gold_display",
           "revised", "label", "pre_correct", "post_correct"]


def export_rows(ev, P, families, tag, split_col):
    out = []
    for fam in families:
        df = ev[ID_COLS].copy()
        df.insert(0, "family", fam)
        df.insert(0, "split_or_fold", split_col)
        df.insert(0, "seed", tag)
        for k in ("old_raw", "new_raw", "old_cal", "new_cal"):
            df[k] = P[fam][k]
        for o in G.OBJECTIVES:
            df[f"tau_{o}"] = P[fam][f"tau_{o}"]
            df[f"accept_{o}"] = P[fam][f"accept_{o}"]
        out.append(df)
    return pd.concat(out, ignore_index=True)


def export_panel(ev, P, ref, families, tie_break, tag):
    out = []
    gates = {f"{f}|{o}": P[f][f"accept_{o}"] for f in families for o in G.OBJECTIVES}
    gates.update({f"{k}|reference": v for k, v in ref.items()})
    for name, acc in gates.items():
        t = panel_table(ev, acc, tie_break).reset_index()
        t = t[[c for c in ("question_id", "condition", "gold_display", "panel_answer", "panel_correct")
               if c in t.columns]].copy()
        t.insert(0, "gate", name)
        t.insert(0, "seed", tag)
        out.append(t)
    return pd.concat(out, ignore_index=True)


# ----------------------------------------------------------------------------- routes


def _blank(n, families):
    P = {f: {k: np.full(n, np.nan) for k in ("old_raw", "new_raw", "old_cal", "new_cal",
                                              "tau_panel", "tau_agent")} for f in families}
    for f in families:
        for o in G.OBJECTIVES:
            P[f][f"accept_{o}"] = np.ones(n, bool)
    return P


def _fill(P, fitted, sc, mask, rev, families):
    for fam in families:
        for key, v in sc[fam].items():
            P[fam][key][mask] = v
        for o in G.OBJECTIVES:
            tau = tau_of(fitted[fam])[o]
            P[fam][f"tau_{o}"][mask] = tau
            P[fam][f"accept_{o}"][mask] = G.gate_accept(sc[fam]["new_cal"], sc[fam]["old_cal"], rev[mask], tau)


def run_cv(ev, feats, families, tie_break, seeds, k_outer=5, transfer=None):
    """``transfer``: {name: (events, feats)} of other conditions on the same questions. Gates
    fitted on this run's training questions are applied, frozen, to their held-out questions."""
    transfer = transfer or {}
    results, rows, panels, choices = [], [], [], []
    rev = (ev.revised == 1).to_numpy()
    uq = np.array(sorted(set(ev.question_id)))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        fold_of = dict(zip(rng.permutation(uq), np.arange(len(uq)) % k_outer, strict=True))
        fold = ev.question_id.map(fold_of).to_numpy()
        P = _blank(len(ev), families)
        PT = {n: _blank(len(e), families) for n, (e, _) in transfer.items()}
        for k in range(k_outer):
            tr, te = fold != k, fold == k
            fitted = fit_all(ev, feats, families, tr, tie_break, "cv")
            _fill(P, fitted, score_all(fitted, feats, te), te, rev, families)
            test_q = set(ev.question_id[te])
            for n, (e, f_t) in transfer.items():
                m = e.question_id.isin(test_q).to_numpy()
                _fill(PT[n], fitted, score_all(fitted, f_t, m), m, (e.revised == 1).to_numpy(), families)
            for fam in families:
                choices.append({"seed": seed, "fold": k, "family": fam, **choices_of(fitted[fam])})
            print(f"  seed {seed} fold {k} done", flush=True)
        res, ref = evaluate(ev, P, tie_break, families, seed)
        res["seed"] = seed
        res["transfer"] = {}
        for n, (e, _) in transfer.items():
            covered = ~np.isnan(PT[n][families[0]]["new_cal"])
            e2 = e[covered].reset_index(drop=True)
            P2 = {f: {k: v[covered] for k, v in PT[n][f].items()} for f in families}
            res["transfer"][n] = evaluate(e2, P2, tie_break, families, seed)[0]
        results.append(res)
        rows.append(export_rows(ev, P, families, seed, fold))
        panels.append(export_panel(ev, P, ref, families, tie_break, seed))
    return results, pd.concat(rows), pd.concat(panels), choices


def run_split(ev, feats, families, tie_break, manifest, transfer=None):
    split = ev.question_id.map(manifest)
    if split.isna().any():
        raise SystemExit(f"{int(split.isna().sum())} events have no split in the manifest")
    tr, va, te = ((split == s).to_numpy() for s in ("train", "validation", "test"))
    rev = (ev.revised == 1).to_numpy()
    fitted = fit_all(ev, feats, families, tr, tie_break, "split", va=va)
    sc = score_all(fitted, feats, te)  # the only time test rows are scored
    ev_te = ev[te].reset_index(drop=True)
    P = {}
    for fam in families:
        P[fam] = dict(sc[fam])
        for o in G.OBJECTIVES:
            tau = tau_of(fitted[fam])[o]
            P[fam][f"tau_{o}"] = np.full(int(te.sum()), tau)
            P[fam][f"accept_{o}"] = G.gate_accept(sc[fam]["new_cal"], sc[fam]["old_cal"], rev[te], tau)
    res, ref = evaluate(ev_te, P, tie_break, families, 0)
    res["split_sizes"] = {s: int(ev[split == s].question_id.nunique()) for s in ("train", "validation", "test")}
    res["transfer"] = {}
    for n, (e, f_t) in (transfer or {}).items():
        m = (e.question_id.map(manifest) == "test").to_numpy()
        sc_t = score_all(fitted, f_t, m)
        e2 = e[m].reset_index(drop=True)
        P2 = {}
        for fam in families:
            P2[fam] = dict(sc_t[fam])
            for o in G.OBJECTIVES:
                tau = tau_of(fitted[fam])[o]
                P2[fam][f"tau_{o}"] = np.full(int(m.sum()), tau)
                P2[fam][f"accept_{o}"] = G.gate_accept(sc_t[fam]["new_cal"], sc_t[fam]["old_cal"],
                                                       (e2.revised == 1).to_numpy(), tau)
        res["transfer"][n] = evaluate(e2, P2, tie_break, families, 0)[0]
    choices = [{"split": "train/validation", "family": f, **choices_of(fitted[f])} for f in families]
    return ([res], export_rows(ev_te, P, families, "split", "test"),
            export_panel(ev_te, P, ref, families, tie_break, "split"), choices)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/night2-debate")
    ap.add_argument("--out", default="artifacts/review/reanalysis_v2")
    ap.add_argument("--mode", choices=["cv", "split"], default="cv")
    ap.add_argument("--split-manifest", help="from scripts/make_split.py; required for --mode split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--time", choices=["pre", "exposure", "post"], default="post")
    ap.add_argument("--position", default="readout_last")
    ap.add_argument("--exposure-position", default="readout_last")
    ap.add_argument("--families", nargs="+", default=list(ALL), choices=list(ALL))
    ap.add_argument("--tie-break", help="default: the run manifest's debate.tie_break")
    ap.add_argument("--transfer-runs", nargs="*", default=[],
                    help="other conditions' runs on the same questions: gates fitted here are "
                         "applied to them, frozen, on held-out questions")
    args = ap.parse_args(argv)
    run = Path(args.run)
    fams = set(args.families)
    if "stacked" in fams:
        fams |= set(Stacked.BASES)
    fams = [f for f in ALL if f in fams]
    man = json.loads((run / "manifest.json").read_text())
    tie = args.tie_break or man.get("config", {}).get("debate", {}).get("tie_break", "round0_then_agent0")
    if tie not in ("round0_then_agent0", "round0_then_confidence", "agent0"):
        print(f"the run's tie_break {tie!r} was never implemented; the rule actually applied was "
              "round0_then_agent0, so that is used")
        tie = "round0_then_agent0"
    ev, feats, report = prepare(run, args.time, args.position, args.exposure_position, fams)
    print(f"{run}: {report['events_used']} events, {report['questions_used']} questions, "
          f"{report['revised_used']} revised; time={args.time}; families={fams}; tie-break={tie}")
    transfer = {}
    for t in args.transfer_runs:
        e_t, f_t, rep_t = prepare(Path(t), args.time, args.position, args.exposure_position, fams)
        if rep_t["output_features"] != report["output_features"]:
            raise SystemExit(f"{t}: output features differ from {run}; transfer needs the same columns")
        transfer[Path(t).name] = (e_t, f_t)
    if args.mode == "split":
        if not args.split_manifest:
            raise SystemExit("--mode split needs --split-manifest")
        manifest = json.loads(Path(args.split_manifest).read_text())["assignment"]
        results, rows, panels, choices = run_split(ev, feats, fams, tie, manifest, transfer)
    else:
        results, rows, panels, choices = run_cv(ev, feats, fams, tie, args.seeds, transfer=transfer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.update({"run": str(run), "mode": args.mode, "tie_break": tie, "families": fams,
                   "threshold_objectives": {"primary": "panel", "secondary": "agent"},
                   "results": results})
    (out / "results.json").write_text(json.dumps(report, indent=2, default=str))
    rows.to_csv(out / "predictions.csv", index=False)
    panels.to_csv(out / "panel.csv", index=False)
    (out / "choices.json").write_text(json.dumps(choices, indent=1, default=str))
    r0 = results[0]
    print(json.dumps({"reference": r0["reference"],
                      "gates_panel_objective": {f: v["gate_panel"] for f, v in r0["families"].items()},
                      "paired_panel": r0["paired_panel"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
