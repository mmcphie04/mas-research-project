"""Committed at read, from SAVED probabilities only (CPU; exploratory; docs/NEXT_PHASE_PLAN.md A2).

    python scripts/exposure_commit.py

For every revision in the archived pilot and validation runs: does the letter distribution read
right after the agent saw the round message -- before it wrote anything (the saved
``peer_exposure`` readout: revision prompt + "\\nFinal answer: (") -- already pick the answer it
eventually stated? Also its previous answer, and the answer key. Reported by condition and by
unchanged / changed / helpful / harmful / wrong->wrong, with question-cluster bootstrap
intervals; the debate - reconsideration gap per question (paper Table 9); a peer-order check.

Anchor at 100%: ``before_stated_answer`` is the distribution at the agent's own "Answer: (",
after all of its reasoning and before the letter. Its argmax agreeing with the sampled letter
is MEASURED here, not assumed (sampling at temperature 0.7 need not pick the top letter).

The prefix differs from the standardized GPU readout ("Answer: ("), so this is a preview of the
f = 0 point, not the replication itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.analysis import commitment as CM  # noqa: E402

OUT = ROOT / "artifacts/next_phase/exposure_commit"
SOURCES = [  # (pool, view, condition label)
    ("pilot", "runs/colab/pilot12b-debate", "debate"),
    ("pilot", "runs/colab/pilot12b-self_reconsider", "self_reconsider"),
    ("pilot", "runs/colab/pilot12b-debate_answer_only", "debate_answer_only"),
    ("pilot", "runs/colab/pilot12b-misleading_one", "misleading_one"),
    ("val", "runs/val12b/val12b-debate", "debate"),
    ("val", "runs/val12b/val12b-self-self_reconsider", "self_reconsider"),
    ("val", "runs/val12b/val12b-independent", "independent"),
]
METRICS = ("exp_is_eventual", "exp_is_previous", "exp_is_gold", "exp_p_eventual", "letter_is_eventual")


def load() -> pd.DataFrame:
    df = pd.concat([CM.load_view(ROOT / v, pool, c) for pool, v, c in SOURCES], ignore_index=True)
    return df


def echo_check(el: pd.DataFrame) -> dict:
    """Peer order is randomised per agent. Among debate revisions whose two shown peers stated
    different answers, how often does the exposure argmax equal the peer shown LAST (nearest the
    readout point) vs the peer shown FIRST? A recency echo would favour the last."""
    d = el[el.condition == "debate"].copy()
    out = {}
    for pool, g in d.groupby("pool"):
        first, last, ev_first, ev_last, n = [], [], [], [], 0
        for r in g.itertuples():
            s = [int(x) for x in str(r.shown_peer_answers).split(",")] if r.shown_peer_answers else []
            if len(s) != 2 or min(s) < 0 or s[0] == s[1]:
                continue
            n += 1
            first.append(r.exp_argmax == s[0])
            last.append(r.exp_argmax == s[1])
            ev_first.append(r.answer == s[0])
            ev_last.append(r.answer == s[1])
        out[pool] = {"revisions_with_two_different_peer_answers": n,
                     "exposure_argmax_is_first_shown_peer": int(sum(first)),
                     "exposure_argmax_is_last_shown_peer": int(sum(last)),
                     "final_stated_answer_is_first_shown_peer": int(sum(ev_first)),
                     "final_stated_answer_is_last_shown_peer": int(sum(ev_last))}
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    excl = CM.exclusions(df)
    el = df[df.eligibility == "eligible"].copy()
    el = CM.add_readout(el, "lp_exposure", "exp")
    el = CM.add_readout(el, "lp_before_stated", "letter")
    rows = []
    for pool_lab, gp in [("pilot", el[el.pool == "pilot"]), ("val", el[el.pool == "val"]), ("pooled", el)]:
        for cond, gc in gp.groupby("condition"):
            for sub, gs in CM.subsets(gc):
                for m in METRICS:
                    rows.append({"pool": pool_lab, "condition": cond, "subset": sub, "metric": m,
                                 **CM.cluster_boot(gs, m, seed=1)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table.csv", index=False)
    paired = {}
    for pool_lab, gp in [("pilot", el[el.pool == "pilot"]), ("val_first100", el[el.pool == "val"]), ("pooled", el)]:
        paired[pool_lab] = {m: CM.paired_gap(gp, "debate", "self_reconsider", m, seed=2)
                            for m in ("exp_is_eventual", "exp_is_previous")}
    summary = {
        "what": "saved peer_exposure readout (prompt + '\\nFinal answer: (') vs the eventual stated answer; "
                "exploratory (pilot and validation questions were seen before)",
        "eligible_responses": {f"{p}/{c}": int(len(g)) for (p, c), g in el.groupby(["pool", "condition"])},
        "exclusions_by_condition_and_reason": excl,
        "paired_debate_minus_self_reconsider_changed_only": paired,
        "echo_check": echo_check(el),
        "anchor_letter_argmax_equals_stated": {f"{p}/{c}": CM.cluster_boot(g, "letter_is_eventual")
                                               for (p, c), g in el.groupby(["pool", "condition"])},
    }
    CM.to_json(summary, OUT / "summary.json")
    view = table[(table.metric == "exp_is_eventual") & (table.pool == "pooled")]
    print(view.pivot_table(index="subset", columns="condition", values="est").round(3).to_string())
    print(np.round(pd.DataFrame(paired).T.map(lambda x: x.get("mean_diff") if isinstance(x, dict) else x), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
