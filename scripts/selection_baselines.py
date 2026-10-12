"""Selecting among FIRST answers with predicted correctness: no further model calls (CPU only).

    python scripts/selection_baselines.py --analysis artifacts/pilot12b/analysis/pre \\
        --outcomes artifacts/pilot12b/outcomes/per_question.csv --out artifacts/pilot12b/selection

From an exploratory reanalysis run's OUT-OF-FOLD predictions (``predictions.csv``; every
prediction was made by a model that never saw that question), for each predictor family:

  most_confident_first    the first answer of the agent with the highest predicted P(correct)
  weighted_first_vote     a vote of the three first answers weighted by predicted P(correct)

compared, per question and paired, with the original vote of three, the vote of all six answers,
debate's final vote, and the run's gates (difference and protect rules, panel objective).
Intervals resample questions. With predictions from ``--time pre`` these use only round-0
information.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def boot(d: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    return [float(x) for x in np.quantile(d[idx].mean(1), [0.025, 0.975])]


def select(pred: pd.DataFrame) -> pd.DataFrame:
    """Per question: correctness of the two first-answer selection rules."""
    out = {}
    for q, g in pred.sort_values("agent_id").groupby("question_id"):
        gold = int(g.gold_display.iloc[0])
        p = g.old_cal.to_numpy(float)
        first = g.pre_answer.to_numpy(int)
        best = int(first[int(np.argmax(p))])  # ties -> lowest agent id
        w = defaultdict(float)
        for a, pa in zip(first, p, strict=True):
            w[int(a)] += float(pa)
        top = max(w.values())
        tied = [a for a in first if w[int(a)] == top]  # ties -> earliest agent's answer
        out[q] = {"most_confident_first": int(best == gold), "weighted_first_vote": int(int(tied[0]) == gold)}
    return pd.DataFrame(out).T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="a reanalysis.py output directory (cv mode)")
    ap.add_argument("--outcomes", required=True, help="per_question.csv from scripts/outcomes.py")
    ap.add_argument("--condition", default="debate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    an = Path(args.analysis)
    pred = pd.read_csv(an / "predictions.csv", dtype={"question_id": str})
    pred = pred[pred.seed.astype(str) == str(args.seed)]
    panel = pd.read_csv(an / "panel.csv", dtype={"question_id": str})
    panel = panel[panel.seed.astype(str) == str(args.seed)]
    oc = pd.read_csv(args.outcomes, dtype={"question_id": str})
    oc = oc[oc.condition == args.condition].set_index("question_id")
    res = json.loads((an / "results.json").read_text())
    table = pd.DataFrame(index=oc.index)
    table["original_vote"] = oc.original_correct
    table["pooled6_vote"] = oc.pooled6_correct
    table[f"{args.condition}_final_vote"] = oc.final_correct
    for fam in sorted(pred.family.unique()):
        s = select(pred[pred.family == fam])
        table[f"{fam}|most_confident_first"] = s.most_confident_first.reindex(table.index)
        table[f"{fam}|weighted_first_vote"] = s.weighted_first_vote.reindex(table.index)
    for gate in panel.gate.unique():
        if gate.endswith("_panel") or "|reference" in gate:
            g = panel[panel.gate == gate].set_index("question_id").panel_correct
            table[f"gate:{gate}"] = g.reindex(table.index)
    summary = {"analysis": str(an), "time": res.get("time"), "condition": args.condition,
               "n_questions": int(len(table)), "rules": {}}
    for col in table.columns:
        d = (table[col] - table.original_vote).to_numpy(float)
        summary["rules"][col] = {"correct": int(table[col].sum()),
                                 "minus_original_vote": {"mean": float(d.mean()), "ci95": boot(d)}}
    (out / f"selection_{res.get('time')}.json").write_text(json.dumps(summary, indent=2))
    table.reset_index().rename(columns={"index": "question_id"}).to_csv(out / f"selection_{res.get('time')}_per_question.csv", index=False)
    print(f"{summary['n_questions']} questions, predictions from time={summary['time']}")
    for k, v in summary["rules"].items():
        m = v["minus_original_vote"]
        print(f"  {k:45s} {v['correct']:4d}   vs original {m['mean']:+.3f} [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
