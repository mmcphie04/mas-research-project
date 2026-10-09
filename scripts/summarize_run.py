"""Is this model a good test bed? Instrument-quality summary for one run.

    python scripts/summarize_run.py --out-root /content/drive/MyDrive/mas_runs --name pilot12b

A pilot answers descriptive questions, not the research question: how accurate is the model,
how often is it cut off, how much do the three agents disagree, how often do they change their
answers, and how much room is there between accepting every change and a perfect choice.
Every accuracy is reported with a 95% interval from resampling questions, so a 60-question pilot
says plainly how much it can and cannot distinguish.
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

from mad.analysis.labels import apply_policy, build_events  # noqa: E402
from mad.analysis.oracles import panel_optimal_accept, policy_accept  # noqa: E402
from mad.debate import ROWS_FILE  # noqa: E402
from mad.provenance import is_complete  # noqa: E402


def boot_ci(per_question: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for a mean of per-question values, resampling questions."""
    v = per_question.to_numpy(dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarize(run_dir: Path, max_tok: int | None) -> dict:
    rows = pd.read_parquet(run_dir / ROWS_FILE)
    ev = build_events(rows)  # uses only the pre and post states
    per_q_agent = ev.groupby("question_id").pre_correct.mean()
    vote = apply_policy(ev, policy_accept(ev, "round0_plurality")).set_index("question_id").panel_correct
    after = apply_policy(ev, policy_accept(ev, "mad_standard")).set_index("question_id").panel_correct
    best = apply_policy(ev, panel_optimal_accept(ev)).set_index("question_id").panel_correct
    change = (after - vote.reindex(after.index))
    unanimous = ev.groupby("question_id").pre_answer.nunique().eq(1)

    def acc(series):
        lo, hi = boot_ci(series)
        return {"mean": round(float(series.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    out = {
        "questions": int(ev.question_id.nunique()),
        "agent_answers": int(len(ev)),
        "one_agent_accuracy": acc(per_q_agent),
        "vote_of_3": acc(vote),
        "after_round_1": acc(after),
        "change_from_round_1": acc(change),
        "best_possible": acc(best),
        "room_for_a_gate": round(float(best.mean() - after.mean()), 4),
        "first_answers_not_unanimous": round(float((~unanimous).mean()), 4),
        "revision_rate": round(float(ev.revised.mean()), 4),
        "revisions": {k: int((ev.label == k).sum()) for k in ("beneficial", "harmful", "unhelpful")},
        "chance_accuracy": round(float((1 / ev.n_options).mean()), 4),
    }
    for phase in ("pre", "post"):
        s = rows[rows.phase == phase]
        d = {"parse_ok": round(float(s.freetext_parse_ok.mean()), 4),
             "median_tokens": float(s.n_completion_tokens.median())}
        if max_tok:
            d["hit_cap"] = round(float((s.n_completion_tokens >= max_tok - 2).mean()), 4)
        out[f"generation_{phase}"] = d
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    root = Path(args.out_root)

    runs = sorted(d for d in root.glob(f"{args.name}-*")
                  if d.is_dir() and "-c0" not in d.name and is_complete(d))
    if not runs:
        print(f"no completed merged runs named {args.name}-* under {root}")
        return 1

    summary = {}
    for d in runs:
        cond = d.name[len(args.name) + 1 :]
        manifest = json.loads((d / "manifest.json").read_text())
        max_tok = manifest.get("config", {}).get("model", {}).get("max_new_tokens")
        summary[cond] = summarize(d, max_tok)
        summary[cond]["model"] = manifest.get("config", {}).get("model", {}).get("repo_id")

    first = next(iter(summary.values()))
    print(f"\n{args.name}: {first['questions']} questions, model {first['model']}, "
          f"chance = {first['chance_accuracy']:.3f}\n")
    hdr = (f"{'condition':<17}{'1 agent':>21}{'vote of 3':>21}{'after round 1':>21}"
           f"{'room for gate':>15}{'revise':>8}")
    print(hdr)
    for cond, s in summary.items():
        def f(k, s=s):
            return f"{s[k]['mean']:.3f} [{s[k]['ci95'][0]:.2f},{s[k]['ci95'][1]:.2f}]"
        print(f"{cond:<17}{f('one_agent_accuracy'):>21}{f('vote_of_3'):>21}{f('after_round_1'):>21}"
              f"{s['room_for_a_gate']:>+15.3f}{s['revision_rate']:>8.3f}")
    print(f"\nfirst answers not unanimous: {first['first_answers_not_unanimous']:.3f}")
    for cond, s in summary.items():
        g0, g1 = s["generation_pre"], s["generation_post"]
        print(f"  {cond:<16} parse ok {g0['parse_ok']:.2f} -> {g1['parse_ok']:.2f}   "
              f"hit cap {g0.get('hit_cap', float('nan')):.2f} -> {g1.get('hit_cap', float('nan')):.2f}   "
              f"median tokens {g0['median_tokens']:.0f} -> {g1['median_tokens']:.0f}")

    out = root / f"{args.name}-summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
