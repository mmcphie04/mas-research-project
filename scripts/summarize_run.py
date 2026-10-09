"""Is this model a good test bed? Instrument-quality summary for one job.

    python scripts/summarize_run.py --out-root /content/drive/MyDrive/mas_runs --name pilot12b

A pilot answers descriptive questions, not the research question: how accurate is the model,
how often is it cut off, how often does its stated answer disagree with the forced readout, how
much do the three agents disagree, how often do they change their answers, and how much room is
there between accepting every change and the best possible choice. Every accuracy and every
difference is reported with a 95% interval from resampling questions, so a 120-question pilot
says plainly how much it can and cannot distinguish.

Conditions are found through the job file (mad.job.view_dirs), never by directory-name patterns.
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
from mad.job import VIEW_FILE, view_dirs  # noqa: E402


def boot_ci(per_question: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for a mean of per-question values, resampling questions."""
    v = per_question.to_numpy(dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def stat(series: pd.Series) -> dict:
    lo, hi = boot_ci(series)
    return {"mean": round(float(series.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_questions": int(len(series))}


def panel_series(ev, accept, tie_break) -> pd.Series:
    return apply_policy(ev, accept, tie_break).set_index("question_id").panel_correct.astype(float)


def summarize(run_dir: Path, tie_break: str) -> tuple[dict, pd.Series]:
    rows = pd.read_parquet(run_dir / ROWS_FILE)
    ev = build_events(rows)  # pre and post states only; uses the canonical answer
    vote = panel_series(ev, policy_accept(ev, "round0_plurality"), tie_break)
    after = panel_series(ev, policy_accept(ev, "mad_standard"), tie_break).reindex(vote.index)
    agent_opt = panel_series(ev, policy_accept(ev, "oracle_full"), tie_break).reindex(vote.index)
    best = panel_series(ev, panel_optimal_accept(ev, tie_break), tie_break).reindex(vote.index)
    unanimous = ev.groupby("question_id").pre_answer.nunique().eq(1)
    view = json.loads((run_dir / VIEW_FILE).read_text()) if (run_dir / VIEW_FILE).exists() else {}

    revs = {}
    for k in ("beneficial", "harmful", "unhelpful"):
        e = ev[ev.label == k]
        revs[k] = {"events": int(len(e)), "distinct_questions": int(e.question_id.nunique())}

    out = {
        "questions": int(ev.question_id.nunique()),
        "agent_answers": int(len(ev)),
        "tie_break": tie_break,
        "completeness": {k: view.get(k) for k in ("complete", "chunks_complete", "chunks_total",
                                                   "questions", "questions_in_job", "hashes_verified")},
        "one_agent_accuracy": stat(ev.groupby("question_id").pre_correct.mean()),
        "one_agent_accuracy_after": stat(ev.groupby("question_id").post_correct.mean()),
        "vote_of_3": stat(vote),
        "after_round_1": stat(after),
        "agent_optimal_not_a_panel_bound": stat(agent_opt),
        "panel_optimal": stat(best),
        "after_minus_vote": stat(after - vote),
        "room_for_a_gate_panel_optimal_minus_after": stat(best - after),
        "first_answers_not_unanimous": round(float((~unanimous).mean()), 4),
        "revision_rate": round(float(ev.revised.mean()), 4),
        "revisions": revs,
        "chance_accuracy": round(float((1 / ev.n_options).mean()), 4),
        "generation": {},
    }
    for phase in ("pre", "post"):
        s = rows[rows.phase == phase]
        d = {"stated_answer_parsed": round(float(s.freetext_parse_ok.mean()), 4),
             "stated_disagrees_readout": round(float(((s.freetext_parse_ok == 1)
                                                      & (s.freetext_agrees_readout == 0)).mean()), 4),
             "median_tokens": float(s.n_completion_tokens.median())}
        if "stop_reason" in s and s.stop_reason.notna().any():
            d["hit_token_cap"] = round(float((s.stop_reason == "length").mean()), 4)
        if "answer_source" in s:
            d["answer_source"] = {k: round(float(v), 4)
                                  for k, v in s.answer_source.value_counts(normalize=True).items()}
        out["generation"][phase] = d
    return out, after


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    if not (root / f"{args.name}-job.json").exists():
        print(f"no job file {args.name}-job.json under {root}")
        return 1
    job = json.loads((root / f"{args.name}-job.json").read_text())
    tie_break = job["identity"]["config"]["debate"]["tie_break"]
    views = view_dirs(root, args.name)
    if not views:
        print(f"no merged views for job {args.name} under {root}")
        return 1

    conds, after = {}, {}
    for cond, d in views.items():
        conds[cond], after[cond] = summarize(d, tie_break)
    if "self_reconsider" in after:
        for cond in conds:
            if cond != "self_reconsider":
                a, b = after[cond], after["self_reconsider"]
                common = a.index.intersection(b.index)
                conds[cond]["after_minus_self_reconsider"] = stat(a[common] - b[common])

    summary = {"job": args.name, "model": job["identity"]["config"]["model"]["repo_id"],
               "pool": job["identity"]["pool"], "conditions": conds}
    first = next(iter(conds.values()))
    print(f"\n{args.name}: {first['questions']} questions, model {summary['model']}, "
          f"tie-break {tie_break}, chance = {first['chance_accuracy']:.3f}\n")
    print(f"{'condition':<20}{'1 agent':>21}{'vote of 3':>21}{'after round 1':>21}"
          f"{'after - vote':>23}{'room for gate':>23}{'revise':>8}")
    for cond, s in conds.items():
        def f(k, s=s, sign=""):
            m, (lo, hi) = s[k]["mean"], s[k]["ci95"]
            return f"{m:{sign}.3f} [{lo:{sign}.2f},{hi:{sign}.2f}]"
        print(f"{cond:<20}{f('one_agent_accuracy'):>21}{f('vote_of_3'):>21}{f('after_round_1'):>21}"
              f"{f('after_minus_vote', sign='+'):>23}"
              f"{f('room_for_a_gate_panel_optimal_minus_after', sign='+'):>23}"
              f"{s['revision_rate']:>8.3f}")
    print(f"\nfirst answers not unanimous: {first['first_answers_not_unanimous']:.3f}")
    for cond, s in conds.items():
        g0, g1 = s["generation"]["pre"], s["generation"]["post"]
        print(f"  {cond:<18} stated answer parsed {g0['stated_answer_parsed']:.2f} -> "
              f"{g1['stated_answer_parsed']:.2f}   hit cap {g0.get('hit_token_cap', float('nan')):.2f} -> "
              f"{g1.get('hit_token_cap', float('nan')):.2f}   stated != readout "
              f"{g0['stated_disagrees_readout']:.2f} -> {g1['stated_disagrees_readout']:.2f}   "
              f"B/H/U {'/'.join(str(s['revisions'][k]['events']) for k in ('beneficial', 'harmful', 'unhelpful'))}"
              f"   complete {s['completeness'].get('complete')}")

    out = root / f"{args.name}-summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
