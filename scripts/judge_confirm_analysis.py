"""Judge confirmation analysis, exactly as committed in docs/REVISION_JUDGE_PLAN.md ("Confirmation
run"), reusing the committed per-judge functions of scripts/judge_analysis.py.

    python scripts/judge_confirm_analysis.py --root runs/judgec --job judgec --prev runs/judge_full \\
        --out artifacts/revision_gate/judge

Primary: the decision bar (a)-(c) on the 303 NEW helpful/harmful events
(artifacts/revision_gate/judge/confirm_events.json) -- P(revision) from the confirmation traces'
forced "1"/"2" readouts averaged over both orders; (b) the out-of-fold logistic and boosted-tree
gates; (c) the own-answer rule from the 3 NEW thinking own answers per question.
Secondary: all 391 helpful/harmful events (the 88 subsample events from the full run's thinking
judge + the 303), and the panel simulation on every question both runs judged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402
from judge_analysis import events_table, load_judge, load_own, per_judge, simulate  # noqa: E402

from mad.gen import archive as A  # noqa: E402

CONFIRM = ROOT / "artifacts/revision_gate/judge/confirm_events.json"


def confirmation_judge(root: Path, job: str) -> pd.DataFrame:
    """event_id -> p_rev, pos_bias from the confirmation readouts (both orders required)."""
    tr = A.read_rows(root, job, "judge_traces")
    ro = A.read_rows(root, job, "judge_traces_ro")
    d = tr[["request_id", "meta_event_id", "meta_order", "thinking_capped"]].merge(ro[["request_id", "p1", "p2"]], on="request_id")
    w = d.pivot_table(index="meta_event_id", columns="meta_order", values=["p1", "p2"]).dropna()
    out = pd.DataFrame({"p_rev": (w[("p2", "original_first")] + w[("p1", "revised_first")]) / 2,
                        "pos_bias": (w[("p1", "original_first")] + w[("p1", "revised_first")]) / 2 - 0.5})
    out.index.name = "event_id"
    out.attrs["thinking_capped_share"] = float(d.thinking_capped.mean())
    return out


def confirmation_own(root: Path, job: str) -> dict:
    own = A.read_rows(root, job, "judge_own")
    out: dict = {}
    for r in own.sort_values(["question_id", "agent"]).to_dict("records"):
        out.setdefault(str(r["question_id"]), []).append(int(r["answer"]))
    return out


def analyze(root: Path, job: str, prev: Path) -> dict:
    rec = json.loads(CONFIRM.read_text())
    t = events_table(ROOT)
    new = t[t.event_id.isin(set(rec["event_ids"]))]
    j_new = confirmation_judge(root, job)
    own_new = confirmation_own(root, job)
    missing = sorted(set(rec["event_ids"]) - set(j_new.index))
    primary = per_judge(new, j_new, own_new)
    j_old = load_judge(prev, "qwen3_think")
    both = pd.concat([j_old[~j_old.index.isin(j_new.index)], j_new])
    own_all = {**load_own(prev, "qwen3_think", ROOT), **own_new}  # the new 3 samples where they exist
    hh = t[t.kind.isin(["helpful", "harmful"]) & t.event_id.isin(set(both.index))]
    secondary = per_judge(hh, both, own_all)
    panels = simulate(t, ROOT, {"qwen3_think_all": both}, {"qwen3_think_all": own_all})
    return {"plan": "docs/REVISION_JUDGE_PLAN.md (Confirmation run)", "events_sha256": rec["sha256"],
            "missing_new_events": missing, "thinking_capped_share_new": j_new.attrs.get("thinking_capped_share"),
            "primary_303_new_events": primary, "secondary_all_391": secondary, "secondary_panels": panels}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--prev", default=str(ROOT / "runs/judge_full"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = analyze(Path(a.root), a.job, Path(a.prev))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "confirmation_results.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({"decision_bar_primary": r["primary_303_new_events"]["decision_bar"],
                      "accuracy_primary": r["primary_303_new_events"]["accuracy"]["pooled"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
