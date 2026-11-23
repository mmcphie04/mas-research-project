"""Experiment B, EXPLORATORY (requested at the review of the committed analyses, 2026-11-23; not
pre-registered): the panel-level effect of each arm, plus descriptive intervals used by the Part 2
figures. CPU only; held-out full run only (plan D5).

    python scripts/expb_panel_effect.py --root runs/expab --job expab --out artifacts/expab/analysis

* Panel plurality before / after: the plurality of the panel's three bank answers (before) and of
  its three revised answers (after), scored as exact expected accuracy with uniform tie-breaking
  (1/k when the gold is among k tied leaders: 1/2 for a two-way tie, 1/3 for a three-way tie).
  Unparsed answers do not vote (as in Experiment A); a panel with no parsed answer scores 0.
* Per mode x arm: before, after and after - before, with a question bootstrap (panels are the units,
  resampled by question; 10,000 resamples, seed 0). Reconsider-alone is reported on the shared
  panels (tag ``full``, paired with the other arms) and on all its panels (``full`` + ``fullx``).
* Paired against reconsider-alone on the shared panels: (after - before)_arm - (after - before)_alone
  per panel, resampled by question. "before" is the same bank answers in both arms, so this equals
  the difference of the after-scores.
* Descriptive, for the figures: per-agent helpful and harmful rates (share of agents with a parsed
  revision), and the thinking answer-prediction curves for changed / unchanged agents with
  question-bootstrap intervals at each cut.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from expa_analysis import vote_score  # noqa: E402
from expb_analysis import boot_groups, load  # noqa: E402

from mad.gen import tasks as T  # noqa: E402

FRACTIONS = [0, 10, 25, 50, 75, 100]


def panel_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, arm, panel), g in df.groupby(["b_mode", "b_arm", "panel"]):
        gold = int(g.meta_gold.iloc[0])
        after = [int(a) if p else -1 for a, p in zip(g.answer, g.parsed, strict=True)]
        rows.append({"b_mode": mode, "b_arm": arm, "panel": panel, "question_id": g.question_id.iloc[0],
                     "shared": bool(g._part.iloc[0].endswith("_full")),
                     "before": vote_score(g.meta_before.astype(int).tolist(), gold), "after": vote_score(after, gold)})
    p = pd.DataFrame(rows)
    p["delta"] = p.after - p.before
    return p


def panel_effect(p: pd.DataFrame) -> dict:
    out, paired = {}, {}
    for mode, gm in p.groupby("b_mode"):
        for arm, g in gm.groupby("b_arm"):
            sets = {"shared": g[g.shared], "all": g} if arm == "alone" else {"shared": g}
            for name, h in sets.items():
                out[f"{mode} {arm}" + (f" ({name} panels)" if arm == "alone" else "")] = {
                    "panels": len(h), **{m: boot_groups(h[m], h.question_id) for m in ("before", "after", "delta")}}
        alone = gm[(gm.b_arm == "alone") & gm.shared].set_index("panel")
        for arm in ("debate", "dissent", "anon"):
            a = gm[(gm.b_arm == arm) & gm.shared].set_index("panel")
            common = a.index.intersection(alone.index)
            d = a.loc[common, "delta"] - alone.loc[common, "delta"]
            paired[f"{mode}: {arm} - alone"] = boot_groups(d.to_numpy(), a.loc[common, "question_id"].to_numpy())
    return {"per_arm": out, "paired_vs_alone_shared_panels": paired}


def agent_rates(df: pd.DataFrame) -> dict:
    out = {}
    v = df[df.kind != "unparsed"]
    for (mode, arm), g in v.groupby(["b_mode", "b_arm"]):
        out[f"{mode} {arm}"] = {k: boot_groups((g.kind == k).astype(float), g.question_id) for k in ("helpful", "harmful")}
    return out


def curve_intervals(df: pd.DataFrame) -> dict:
    if "curve_logprobs" not in df:
        return {}
    d = df[df.curve_logprobs.notna() & (df.kind != "unparsed")]
    out = {}
    for arm, g in d.groupby("b_arm"):
        for name, h in (("changed", g[g.kind != "unchanged"]), ("unchanged", g[g.kind == "unchanged"])):
            p = np.array([[math.exp(lp[int(a)]) for lp in cv] for cv, a in zip(h.curve_logprobs, h.answer, strict=True)])
            out[f"qwen_t {arm} {name}"] = {"n": len(h), "fractions": FRACTIONS,
                                           "p_final_answer": [boot_groups(p[:, i], h.question_id) for i in range(p.shape[1])]}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    df, _ = load(Path(a.root), a.job)
    _, assign, _ = T.load_questions(a.plan)
    df = df[df.question_id.map(assign) == "heldout"].reset_index(drop=True)
    r = {"label": "EXPLORATORY (not pre-registered)", "split": "heldout", **panel_effect(panel_scores(df)),
         "agent_rates": agent_rates(df), "thinking_curve_intervals": curve_intervals(df)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expb_panel_effect.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r["paired_vs_alone_shared_panels"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
