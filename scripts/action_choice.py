"""Could existing data support choosing the next action? (CPU only; exploratory.)

    python scripts/action_choice.py                  # final-panel voting (current rule)
    python scripts/action_choice.py --rule pooled    # the 2026-10-28 analysis, reproduced

200 validation questions. Actions after the three original answers, and their final answers
(``mad.analysis.actions``; docs/NEXT_PHASE_PLAN.md review decision 5):

  final_panel (default)
    stop         plurality of the 3 initial answers
    debate       plurality of the 3 revised answers
    independent  plurality of all 6 independent answers (3 initial + 3 fresh)
    Ties: uniformly random, scored exactly by expectation -> fractional outcomes throughout.

  pooled (preserved for comparison; outputs in pooled_vote_2026-10-28/)
    every available answer in one vote, ties to the round-0 plurality, then agent order. Under
    it a unanimous initial panel can never change.

Policies may use ONLY information available before the action: round-0 answers and their
visible features, then also the frozen R1 activation scores of the round-0 states. Learned
policies: one fractional-label logistic model per action, question-level cross-validation (10
folds, 5 seeds). The best action chosen after seeing outcomes is reported only as an
optimistic retrospective reference. Also writes a provisional Stage 4 sample-size re-estimate.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import actions as AC  # noqa: E402
from mad.analysis import candidate_scoring as CS  # noqa: E402

OUT = ROOT / "artifacts/val12b/action_choice"
VISIBLE = ["n_distinct", "top_count", "verbal_mean", "verbal_min", "gen_pmax_mean", "stated_p_mean", "len_mean",
           "any_capped", "logprob_mean"]
INTERNAL = ["act_mean", "act_max", "act_spread", "act_of_plurality"]


def records(rule: str) -> pd.DataFrame:
    D = CS.StateData.from_cache(ROOT / "runs/cache/val12b")
    F = pickle.load(open(ROOT / "runs/frozen/pilot12b_scorers_3857a72.pkl", "rb"))
    z = F["models"]["R1"]["act"].score(D, np.ones(len(D.y), bool))[1]
    st = D.states.reset_index(drop=True).assign(z=z)
    rows = []
    for q, g0 in st[st.phase == "pre"].groupby("question_id"):
        g0 = g0.sort_values("agent_id")
        gd = st[(st.phase == "post") & (st.condition == "debate") & (st.question_id == q)].sort_values("agent_id")
        gi = st[(st.phase == "post") & (st.condition == "independent") & (st.question_id == q)].sort_values("agent_id")
        gold = int(g0.gold_display.iloc[0])
        a0, ad, ai = list(g0.answer), list(gd.answer), list(gi.answer)
        cnt = Counter(a0)
        top = max(cnt.values())
        plural_set = [k for k, v in cnt.items() if v == top]
        rec = {"question_id": q, "subject": g0.subject.iloc[0], **AC.outcomes(a0, ad, ai, gold, rule),
               "debate_revised_only": (AC.expected_vote(ad, gold) if rule == "final_panel"
                                       else AC.pooled_vote(ad, a0, gold)),
               # visible, pre-action
               "n_distinct": len(cnt), "top_count": top,
               "verbal_mean": float(g0.of_verbalized.mean()), "verbal_min": float(g0.of_verbalized.min()),
               "gen_pmax_mean": float(g0.of_generation_last_p_max.mean()),
               "stated_p_mean": float(g0.of_stated_answer_p_answer.mean()),
               "len_mean": float(g0.of_n_completion_tokens.mean()), "any_capped": int(g0.of_capped.max()),
               "logprob_mean": float(g0.of_response_logprob_mean.mean()),
               # internal, pre-action (frozen R1 on the round-0 states)
               "act_mean": float(g0.z.mean()), "act_max": float(g0.z.max()), "act_spread": float(g0.z.max() - g0.z.min()),
               "act_of_plurality": float(g0.z[g0.answer.isin(plural_set)].mean())}
        rows.append(rec)
    return pd.DataFrame(rows).set_index("question_id")


def stage4_estimate(R: pd.DataFrame) -> dict:
    """Provisional planning numbers under this rule (docs/NEXT_PHASE_PLAN.md decision 11)."""
    A = R[list(AC.ACTIONS)].to_numpy(float)
    differs = A.max(1) != A.min(1)
    disputed = R.n_distinct.to_numpy() > 1
    out = {"questions": int(len(R)), "disputed": int(disputed.sum()),
           "informative_any_difference": int(differs.sum()),
           "informative_rate_all": float(differs.mean()),
           "informative_rate_disputed": float(differs[disputed].mean()) if disputed.any() else None,
           "informative_rate_unanimous": float(differs[~disputed].mean()) if (~disputed).any() else None}
    for name, a, b in (("independent_vs_stop", "independent", "stop"), ("debate_vs_stop", "debate", "stop"),
                       ("debate_vs_independent", "debate", "independent")):
        d = (R[a] - R[b]).to_numpy(float)
        out[name] = {"mean_diff": float(d.mean()), "questions_differing": int((d != 0).sum()),
                     "all_questions": AC.sample_size(d),
                     "disputed_only": AC.sample_size(d[disputed])}
    r = out["informative_rate_all"]
    out["provisional_85_discordant_assumption"] = {
        "note": "the 85-discordant-question figure is a provisional planning assumption (65/35 sign test)",
        "questions_needed_all": int(np.ceil(85 / r)) if r else None,
        "questions_needed_if_only_disputed_run": (int(np.ceil(85 / out["informative_rate_disputed"]))
                                                  if out["informative_rate_disputed"] else None)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["final_panel", "pooled"], default="final_panel")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = Path(a.out) if a.out else OUT / ("final_panel" if a.rule == "final_panel" else "pooled_vote_rerun")
    out_dir.mkdir(parents=True, exist_ok=True)
    R = records(a.rule)
    R.to_csv(out_dir / "question_records.csv")
    out = {"rule": a.rule, "n_questions": int(len(R)),
           "expected_correct": {k: float(R[k].sum()) for k in (*AC.ACTIONS, "debate_revised_only")}}
    A = R[list(AC.ACTIONS)].to_numpy(float)
    out["distinguishing_questions"] = {
        "any_action_differs": int((A.max(1) != A.min(1)).sum()),
        **{f"{x}_vs_{y}": {"questions_differing": int((R[x] != R[y]).sum()),
                           "expected_gain": float((R[x] - R[y]).clip(lower=0).sum()),
                           "expected_loss": float((R[y] - R[x]).clip(lower=0).sum())}
           for x, y in (("independent", "stop"), ("debate", "stop"), ("debate", "independent"))}}
    by = R.groupby("n_distinct")[list(AC.ACTIONS)].agg(["sum", "size"])
    out["by_initial_agreement"] = {int(k): {x: [round(float(by.loc[k, (x, "sum")]), 3), int(by.loc[k, (x, "size")])]
                                            for x in AC.ACTIONS} for k in by.index}
    unanimous = R.n_distinct == 1
    pols = [AC.evaluate(R, np.full(len(R), x), f"always {x}") for x in AC.ACTIONS]
    pols += [AC.evaluate(R, np.where(unanimous, "stop", "independent"), "unanimous -> stop, else independent"),
             AC.evaluate(R, np.where(unanimous, "stop", "debate"), "unanimous -> stop, else debate")]
    for thr in (80, 90, 95):
        pols.append(AC.evaluate(R, np.where(R.verbal_mean >= thr, "stop", "independent"),
                                f"mean stated confidence >= {thr} -> stop, else independent"))
    for feats, lab in ((VISIBLE, "visible"), (VISIBLE + INTERNAL, "visible+internal")):
        runs = [AC.evaluate(R, AC.learned_choice(R, feats, CS.make_folds(np.arange(len(R)), 10, s)),
                            f"learned ({lab}), seed {s}") for s in range(5)]
        pols.append({"policy": f"learned ({lab}), 5 CV seeds",
                     "expected_correct_range": [min(p["expected_correct"] for p in runs), max(p["expected_correct"] for p in runs)],
                     "minus_stop_range": [min(p["minus_stop"] for p in runs), max(p["minus_stop"] for p in runs)],
                     "runs": runs})
    pols.append(AC.evaluate(R, AC.hindsight_best(R), "retrospective best action (optimistic; uses outcomes)"))
    out["policies"] = pols
    if a.rule == "final_panel":
        out["stage4_reestimate"] = stage4_estimate(R)
    (out_dir / "action_choice.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("policies",)}, indent=1, default=float))
    for p in pols:
        if "runs" in p:
            print(f"  {p['policy']:58s} expected correct {[round(x, 2) for x in p['expected_correct_range']]}")
        else:
            print(f"  {p['policy']:58s} {p['expected_correct']:7.2f}  minus stop {p['minus_stop']:+.4f} "
                  f"[{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]  gain {p['expected_gain_vs_stop']:.2f} "
                  f"loss {p['expected_loss_vs_stop']:.2f}  extra {p['extra_answers_generated']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
