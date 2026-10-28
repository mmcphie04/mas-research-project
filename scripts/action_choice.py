"""Could existing data support choosing the next action? (CPU only; exploratory.)

    python scripts/action_choice.py

200 validation questions. Actions after the three original answers:
  stop         final answer = vote over the 3 original answers
  independent  add three independent answers; vote over all 6
  debate       add one debate round; vote over all 6 (sensitivity: vote over the 3 revised answers)
One fixed final-answer rule (plurality, preregistered tie-break: round-0 plurality, then agent
order) across actions. Policies may use ONLY information available before the action: round-0
answers and their visible features, then also the frozen R1 activation scores of the round-0
states. Learned policies are fitted and evaluated with question-level cross-validation (10 folds,
repeated over 5 seeds). The best action chosen after seeing outcomes is reported only as an
optimistic retrospective reference.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import candidate_scoring as CS  # noqa: E402
from mad.analysis.labels import panel_vote  # noqa: E402

OUT = ROOT / "artifacts/val12b/action_choice"
TIE = "round0_then_agent0"


def _vote(ans, fb):
    return panel_vote(list(ans), [0.0] * len(ans), fb, TIE)


def records():
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
        plural = _vote(a0, None)
        rec = {"question_id": q, "subject": g0.subject.iloc[0],
               "stop": int(plural == gold), "independent": int(_vote(a0 + ai, a0) == gold),
               "debate": int(_vote(a0 + ad, a0) == gold), "debate_revised_only": int(_vote(ad, a0) == gold),
               # visible, pre-action
               "n_distinct": len(cnt), "top_count": max(cnt.values()),
               "verbal_mean": float(g0.of_verbalized.mean()), "verbal_min": float(g0.of_verbalized.min()),
               "gen_pmax_mean": float(g0.of_generation_last_p_max.mean()),
               "stated_p_mean": float(g0.of_stated_answer_p_answer.mean()),
               "len_mean": float(g0.of_n_completion_tokens.mean()), "any_capped": int(g0.of_capped.max()),
               "logprob_mean": float(g0.of_response_logprob_mean.mean()),
               # internal, pre-action (frozen R1 on the round-0 states)
               "act_mean": float(g0.z.mean()), "act_max": float(g0.z.max()), "act_spread": float(g0.z.max() - g0.z.min()),
               "act_of_plurality": float(g0.z[g0.answer == plural].mean())}
        rows.append(rec)
    return pd.DataFrame(rows).set_index("question_id")


VISIBLE = ["n_distinct", "top_count", "verbal_mean", "verbal_min", "gen_pmax_mean", "stated_p_mean", "len_mean",
           "any_capped", "logprob_mean"]
INTERNAL = ["act_mean", "act_max", "act_spread", "act_of_plurality"]
ACTIONS = ("stop", "independent", "debate")


def learned_policy(R, feats, seed, k=10, C=0.3):
    """Per action, a logistic model of P(correct | pre-action features), fitted on training
    questions; choose the action with the highest predicted correctness (ties -> stop)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = R[feats].fillna(R[feats].median()).to_numpy(float)
    f = CS.make_folds(np.arange(len(R)), k, seed)
    choice = np.empty(len(R), object)
    for r in range(k):
        tr, te = f != r, f == r
        P = {}
        for a in ACTIONS:
            y = R[a].to_numpy()[tr]
            if len(np.unique(y)) < 2:
                P[a] = np.full(te.sum(), y.mean())
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y)
            P[a] = m.predict_proba(X[te])[:, 1]
        best = np.vstack([P[a] for a in ACTIONS])
        idx = np.argmax(best + np.array([[1e-9], [0], [0]]), 0)  # ties go to stop
        choice[te] = [ACTIONS[i] for i in idx]
    return choice


def evaluate(R, choice, name, n_boot=10000):
    choice = np.asarray(choice)
    got = np.array([R[c].iloc[i] for i, c in enumerate(choice)])
    stop = R["stop"].to_numpy()
    d = got - stop
    rng = np.random.default_rng(0)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"policy": name, "correct": int(got.sum()), "minus_stop": float(d.mean()),
            "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "corrections_vs_stop": int(((got == 1) & (stop == 0)).sum()), "new_errors_vs_stop": int(((got == 0) & (stop == 1)).sum()),
            "extra_answers_generated": int(3 * (choice != "stop").sum()),
            "share_stop": float(np.mean(choice == "stop")), "share_independent": float(np.mean(choice == "independent")),
            "share_debate": float(np.mean(choice == "debate"))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = records()
    R.to_csv(OUT / "question_records.csv")
    out = {"n_questions": int(len(R)), "counts": {a: int(R[a].sum()) for a in (*ACTIONS, "debate_revised_only")}}
    A = R[list(ACTIONS)].to_numpy()
    out["distinguishing_questions"] = {
        "any_action_differs": int((A.max(1) != A.min(1)).sum()),
        "independent_vs_stop": {"gain": int(((R.independent == 1) & (R.stop == 0)).sum()), "loss": int(((R.independent == 0) & (R.stop == 1)).sum())},
        "debate_vs_stop": {"gain": int(((R.debate == 1) & (R.stop == 0)).sum()), "loss": int(((R.debate == 0) & (R.stop == 1)).sum())},
        "debate_vs_independent": {"debate_only": int(((R.debate == 1) & (R.independent == 0)).sum()),
                                  "independent_only": int(((R.debate == 0) & (R.independent == 1)).sum())}}
    by_agree = R.groupby("n_distinct")[list(ACTIONS)].agg(["sum", "size"])
    out["by_initial_agreement"] = {int(k): {a: f"{int(by_agree.loc[k, (a, 'sum')])}/{int(by_agree.loc[k, (a, 'size')])}" for a in ACTIONS}
                                   for k in by_agree.index}
    unanimous = R.n_distinct == 1
    pols = [evaluate(R, np.full(len(R), a), f"always {a}") for a in ACTIONS]
    pols += [evaluate(R, np.where(unanimous, "stop", "independent"), "unanimous -> stop, else independent"),
             evaluate(R, np.where(unanimous, "stop", "debate"), "unanimous -> stop, else debate")]
    for thr in (80, 90, 95):
        pols.append(evaluate(R, np.where(R.verbal_mean >= thr, "stop", "independent"),
                             f"mean stated confidence >= {thr} -> stop, else independent"))
    for feats, lab in ((VISIBLE, "visible"), (VISIBLE + INTERNAL, "visible+internal")):
        runs = [evaluate(R, learned_policy(R, feats, s), f"learned ({lab}), seed {s}") for s in range(5)]
        pols.append({"policy": f"learned ({lab}), 5 CV seeds", "correct_range": [min(p["correct"] for p in runs), max(p["correct"] for p in runs)],
                     "minus_stop_range": [min(p["minus_stop"] for p in runs), max(p["minus_stop"] for p in runs)],
                     "runs": runs})
    best = np.array([ACTIONS[int(np.argmax(row + np.array([1e-9, 0, 0])))] for row in A])
    pols.append({**evaluate(R, best, "retrospective best action (optimistic; uses outcomes)")})
    out["policies"] = pols
    (OUT / "action_choice.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "policies"}, indent=1))
    for p in pols:
        if "runs" in p:
            print(f"  {p['policy']:58s} correct {p['correct_range']}  minus stop {[round(x, 3) for x in p['minus_stop_range']]}")
        else:
            print(f"  {p['policy']:58s} correct {p['correct']:3d}  minus stop {p['minus_stop']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}]"
                  f"  fixes {p['corrections_vs_stop']} breaks {p['new_errors_vs_stop']}  extra answers {p['extra_answers_generated']}")


if __name__ == "__main__":
    main()
