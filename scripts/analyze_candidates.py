"""Analyse a candidate-evaluation run.

    python scripts/analyze_candidates.py --run ROOT/main12b-candidates --out OUT [--split-manifest S]

Output-only comparator: a logistic model, fitted on the same training questions as the probes, on
the full letter distribution, the candidate's and the endorsed option's probabilities, whether
the endorsement agrees with the candidate, the context and categorical candidate positions. The
raw probability of the candidate's letter is reported too. Descriptive output-only measures:
  * within-question ranking of the correct over the incorrect candidate, per context;
  * endorsement shifts: change in log-odds of the candidate's letter under an endorsement that
    agrees / disagrees with it, relative to neutral, by candidate kind.
Probes (states at the readout, layer chosen by grouped CV on training questions only):
  * candidate-correctness AUROC on held-out questions, overall, within question, per context;
  * combined (states + output score) vs output-only, including under misleading endorsements.
All intervals resample questions. Questions carry the split recorded when they were collected
(the parent's registered split); probes and the output model train on train + validation and are
scored once on test. Without a recorded split: grouped 5-fold CV over questions (exploratory).
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
from sklearn.model_selection import GroupKFold  # noqa: E402

from mad.analysis.gates import C_GRID, cv_logloss, linear  # noqa: E402
from mad.analysis.states import StateBank  # noqa: E402
from mad.analysis.stats import bootstrap_auroc, cluster_bootstrap, conditional_auroc  # noqa: E402
from mad.candidates import ROWS  # noqa: E402


def candidate_output_features(r: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Everything observable without activations: the full letter distribution, the candidate's
    and the endorsed option's probabilities, whether the endorsement agrees with the candidate,
    which context, and categorical candidate positions. Never gold or the candidate's kind."""
    width = int(r.n_options.max())
    lp = np.full((len(r), width), -30.0)
    for i, v in enumerate(r.readout_logprobs):
        a = np.array([float(x) for x in v.split(",")])
        lp[i, : len(a)] = a
    cols = {f"lp{j}": lp[:, j] for j in range(width)}
    srt = np.sort(lp, axis=1)
    cols["margin"] = srt[:, -1] - srt[:, -2]
    cols["logit_p_candidate"] = logit(r.p_candidate.to_numpy())
    cols["p_endorsed"] = r.p_endorsed.fillna(0).to_numpy()
    cols["endorsed"] = (r.endorsed >= 0).to_numpy(float)
    cols["endorsement_agrees_with_candidate"] = (r.endorsed == r.candidate).to_numpy(float)
    cols["readout_picks_candidate"] = (r.readout_answer == r.candidate).to_numpy(float)
    for j in range(width):
        cols[f"candidate_is_{j}"] = (r.candidate == j).to_numpy(float)
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-manifest")
    args = ap.parse_args(argv)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(run / ROWS)
    r["phase"] = r.candidate_kind + "|" + r.context
    r["condition"], r["agent_id"] = "candidates", r["item"]
    g = r.question_id.to_numpy()
    rep = {"questions": int(r.question_id.nunique()), "readouts": int(len(r))}

    # ---- output-only
    piv = r.pivot_table(index="question_id", columns=["candidate_kind", "context"], values="p_candidate")
    rep["output_ranking_correct_over_incorrect"] = {
        ctx: float((piv[("correct", ctx)] > piv[("incorrect", ctx)]).mean()) for ctx in ("neutral", "endorse_correct", "endorse_incorrect")}
    shifts = {}
    for kind in ("correct", "incorrect"):
        agree = "endorse_correct" if kind == "correct" else "endorse_incorrect"
        disagree = "endorse_incorrect" if kind == "correct" else "endorse_correct"
        for name, ctx in (("agreeing", agree), ("disagreeing", disagree)):
            d = (logit(piv[(kind, ctx)]) - logit(piv[(kind, "neutral")])).to_numpy()
            iv = cluster_bootstrap(lambda x: float(np.mean(x)), np.arange(len(d)), d)
            shifts[f"{kind} candidate, {name} endorsement"] = {"mean_logodds_shift": iv.point, "ci95": [iv.lo, iv.hi]}
    rep["endorsement_shifts"] = shifts

    # ---- probes and the output-only comparator, fitted on the same training questions
    bank = StateBank(run)
    y = r.candidate_is_correct.to_numpy()
    s_out = logit(r.p_candidate.to_numpy())
    X_out, out_names = candidate_output_features(r)
    layers = sorted({int(bank.n_layers * f) for f in (0.25, 0.5, 0.65, 0.8, 0.92, 1.0)})

    rows_ix = np.concatenate([bank.rows(r[r.phase == ph], ph, "readout_last")[0]
                              for ph in r.phase.unique()])
    order = np.concatenate([np.flatnonzero((r.phase == ph).to_numpy()) for ph in r.phase.unique()])
    lookup = np.empty(len(r), int)
    lookup[order] = rows_ix

    def states(L):
        return bank.layer(L)[lookup]

    recorded = set(r.split) if "split" in r.columns else {"unassigned"}
    if args.split_manifest or "unassigned" not in recorded:
        if args.split_manifest:
            sys.path.insert(0, str(ROOT / "scripts"))
            from make_split import load_split

            # every question must be assigned; an unmapped one is an error, never silently "train"
            assign, smeta = load_split(args.split_manifest, sorted(set(r.question_id)), allow_subset=True)
            man = json.loads((run / "manifest.json").read_text())
            if man.get("split_assignment_sha256") not in (None, smeta["assignment_sha256"]):
                raise SystemExit("this manifest is not the split the candidate run recorded")
            split = r.question_id.map(assign)
        else:
            split = r.split  # the split recorded with each question at collection time
        test = split.eq("test").to_numpy()
        train = split.isin(["train", "validation"]).to_numpy()
        if not test.any() or not train.any():
            raise SystemExit("the candidate run needs both training and test questions")
        folds, route = [(train, test)], "split (train + validation fit, test scored once)"
    else:
        folds = list(GroupKFold(n_splits=5).split(np.zeros(len(r)), groups=g))
        folds = [(np.isin(np.arange(len(r)), a), np.isin(np.arange(len(r)), b)) for a, b in folds]
        route = "grouped 5-fold CV (exploratory)"
    p_probe, p_comb, p_outm = (np.full(len(r), np.nan) for _ in range(3))
    chosen = []
    for tr, te in folds:
        L = min(layers, key=lambda L: cv_logloss(lambda: linear(1e-2), states(L)[tr], y[tr], g[tr]))
        X = states(L)
        C = min(C_GRID, key=lambda C: cv_logloss(lambda C=C: linear(C), X[tr], y[tr], g[tr]))
        p_probe[te] = linear(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        Co = min(C_GRID, key=lambda C: cv_logloss(lambda C=C: linear(C), X_out[tr], y[tr], g[tr]))
        p_outm[te] = linear(Co).fit(X_out[tr], y[tr]).predict_proba(X_out[te])[:, 1]
        Xc = np.hstack([X, X_out])
        p_comb[te] = linear(C).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
        chosen.append({"layer": int(L), "C_probe": C, "C_output": Co})
    held = ~np.isnan(p_probe)
    res = {}
    for name, s in (("output_raw_p_candidate", s_out), ("output_model", p_outm), ("probe", p_probe),
                    ("combined", p_comb)):
        m = held
        ci = bootstrap_auroc(y[m], s[m], g[m])
        within, _, _ = conditional_auroc(y[m], s[m], g[m])
        per_ctx = {ctx: float(bootstrap_auroc(y[m & (r.context == ctx).to_numpy()], s[m & (r.context == ctx).to_numpy()],
                                              g[m & (r.context == ctx).to_numpy()]).point) for ctx in ("neutral", "endorse_correct", "endorse_incorrect")}
        res[name] = {"auroc": ci.point, "ci95": [ci.lo, ci.hi], "auroc_within_question": within, "by_context": per_ctx}
    rep["candidate_correctness"] = res
    rep["output_features"] = out_names
    rep["probe_choices"] = chosen
    rep["route"] = route
    (out / "candidates_report.json").write_text(json.dumps(rep, indent=2, default=str))
    r.assign(p_probe=p_probe, p_combined=p_comb).drop(columns=["readout_logprobs"]).to_csv(out / "candidate_predictions.csv", index=False)
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
