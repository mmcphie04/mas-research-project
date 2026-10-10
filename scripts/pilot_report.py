"""One pilot report: instrument quality, headroom, class support, runtime, disk and power.

    python scripts/pilot_report.py --out-root /content/drive/MyDrive/mas_runs --name pilot12b \\
        --main-n 1200 1800 2400 --main-conditions 3 --out pilot12b-report

The pilot chooses instrument settings and an affordable scope. It does not test the activation
hypothesis, and nothing here should be read as evidence for or against it.

Writes <out>/report.json, <out>/report.md and <out>/transcripts.md (a fixed random sample of
transcripts plus every malformed answer, for reading by hand).
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
from scipy.stats import norm  # noqa: E402

from mad.analysis.labels import apply_policy, build_events  # noqa: E402
from mad.analysis.oracles import policy_accept  # noqa: E402
from mad.debate import ROWS_FILE, TRANSCRIPTS_FILE  # noqa: E402
from mad.job import chunk_tag, view_dirs  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from summarize_run import summarize  # noqa: E402


def dir_bytes(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def required_test_questions(effect: float, discordant: float, alpha=0.05, power=0.8) -> int:
    """Test questions needed to detect a paired panel-accuracy difference ``effect`` when a
    fraction ``discordant`` of questions are decided differently by the two gates (McNemar,
    normal approximation): among discordant questions the better gate wins with probability
    0.5 + effect / (2 * discordant)."""
    if effect >= discordant:
        return 0
    pi = 0.5 + effect / (2 * discordant)
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    n_d = ((za * 0.5 + zb * np.sqrt(pi * (1 - pi))) / (pi - 0.5)) ** 2
    return int(np.ceil(n_d / discordant))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--main-n", type=int, nargs="+", default=[1200, 1800, 2400])
    ap.add_argument("--main-conditions", type=int, default=3)
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=12, help="transcripts sampled per condition")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.out_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    job = json.loads((root / f"{args.name}-job.json").read_text())
    tie = job["identity"]["config"]["debate"]["tie_break"]
    views = view_dirs(root, args.name)
    n_q = len(job["questions"])
    rep = {"job": args.name, "questions": n_q, "conditions": {}, "caveat":
           "pilot data choose instrument settings and scope; they do not test the activation hypothesis"}

    # ---- quality, headroom, class support (per condition)
    disc_rates = {}
    for cond, d in views.items():
        s, _ = summarize(d, tie)
        rep["conditions"][cond] = s
        ev = build_events(pd.read_parquet(d / ROWS_FILE))
        a = apply_policy(ev, policy_accept(ev, "mad_standard"), tie).set_index("question_id").panel_correct
        b = apply_policy(ev, policy_accept(ev, "round0_plurality"), tie).set_index("question_id").panel_correct
        disc_rates[cond] = float((a != b.reindex(a.index)).mean())
    rep["accept_all_vs_reject_all_discordance"] = disc_rates

    # ---- runtime
    status = json.loads((root / f"{args.name}-status.json").read_text())
    t = pd.DataFrame(status["timings"])
    spq = float(t.seconds.sum() / t.questions.sum()) if len(t) else float("nan")
    stage_tot = {}
    for st in t.get("stages", []):
        for k, v in (st or {}).items():
            stage_tot[k] = stage_tot.get(k, 0.0) + float(v)
    n_cond = len(job["identity"]["conditions"])
    r0_share = stage_tot.get("round0_s", 0.0) / max(t.seconds.sum(), 1e-9) if len(t) else float("nan")
    per_cond = (1 - r0_share) / max(n_cond, 1)
    spq_main = spq * (r0_share + per_cond * args.main_conditions)
    rep["runtime"] = {"seconds_per_question_measured": round(spq, 1), "conditions_measured": n_cond,
                      "stage_seconds_total": {k: round(v, 1) for k, v in sorted(stage_tot.items())},
                      "round0_share_of_time": round(r0_share, 3),
                      "seconds_per_question_projected": round(spq_main, 1),
                      "projected_gpu_hours": {n: round(float(spq_main * n / 3600), 1) for n in args.main_n}}
    ledger = root / f"{args.name}-ledger.jsonl"
    if ledger.exists():
        ev_l = [json.loads(x) for x in ledger.read_text().splitlines()]
        ends = [e["session_seconds"] for e in ev_l if e["event"] == "session_end"]
        rep["runtime"]["session_wall_hours_total"] = round(sum(ends) / 3600, 2)

    # ---- disk
    r0 = sum(dir_bytes(root / f"{args.name}-round0-{chunk_tag(i)}") for i in range(len(job["chunks"])))
    cond_b = {c: sum(dir_bytes(root / f"{args.name}-{c}-{chunk_tag(i)}") for i in range(len(job["chunks"])))
              for c in job["identity"]["conditions"]}
    per_q_r0, per_q_cond = r0 / n_q, (np.mean(list(cond_b.values())) / n_q if cond_b else 0)
    rep["disk"] = {"round0_bytes": r0, "condition_bytes": cond_b,
                   "projected_gb": {n: round(float((per_q_r0 + per_q_cond * args.main_conditions) * n / 1e9), 1)
                                    for n in args.main_n}}

    # ---- power sensitivity
    rows = []
    for disc in (0.05, 0.10, 0.15, 0.20):
        for eff in (0.02, 0.03, 0.04):
            need = required_test_questions(eff, disc)
            rows.append({"discordant_fraction": disc, "effect_points": eff * 100,
                         "test_questions_needed": need,
                         "total_questions_at_test_fraction": int(np.ceil(need / args.test_fraction))})
    rep["power"] = {"test_fraction": args.test_fraction, "table": rows,
                    "note": "discordance between two learned gates is unknown before the main run; the "
                            "pilot's accept-all vs reject-all discordance is an upper reference"}

    # ---- transcripts for reading by hand
    rng = np.random.default_rng(0)
    lines = [f"# Transcript sample: {args.name}\n", "Fixed random sample (seed 0) plus every malformed answer.\n"]
    for cond, d in views.items():
        t_df = pd.read_json(d / TRANSCRIPTS_FILE, lines=True)
        r = pd.read_parquet(d / ROWS_FILE)
        post = t_df[t_df.phase == "post"]
        pick = post.iloc[rng.choice(len(post), size=min(args.sample, len(post)), replace=False)]
        bad_keys = r[(r.phase != "peer_exposure") & (r.stated_status.isin(["missing", "out_of_range", "permissive"]))]
        bad = t_df.merge(bad_keys[["question_id", "agent_id", "phase", "stated_status"]],
                         on=["question_id", "agent_id", "phase"])
        lines.append(f"\n## {cond}: {len(pick)} sampled, {len(bad)} malformed or permissive\n")
        for _, x in pd.concat([pick.assign(stated_status="sampled"), bad]).iterrows():
            lines.append(f"\n### {x.question_id} agent {x.agent_id} {x.phase} [{x.stated_status}] "
                         f"stop={x.get('stop_reason')}\n\n```\n{str(x.text)[:3000]}\n```\n")
    (out / "transcripts.md").write_text("".join(lines))

    (out / "report.json").write_text(json.dumps(rep, indent=2, default=str))
    md = [f"# Pilot report: {args.name}\n\n_{rep['caveat']}._\n\n",
          "| condition | 1 agent | vote of 3 | after round 1 | panel optimal | room above better baseline | "
          "parsed (pre/post) | cap hit (pre/post) | B / H / U (distinct q) |\n|---|---|---|---|---|---|---|---|---|\n"]
    for cond, s in rep["conditions"].items():
        def f(k, s=s):
            return f"{s[k]['mean']:.3f} [{s[k]['ci95'][0]:.2f}, {s[k]['ci95'][1]:.2f}]"
        g0, g1 = s["generation"]["pre"], s["generation"]["post"]
        bhu = " / ".join(f"{s['revisions'][k]['events']} ({s['revisions'][k]['distinct_questions']})"
                         for k in ("beneficial", "harmful", "unhelpful"))
        md.append(f"| {cond} | {f('one_agent_accuracy')} | {f('vote_of_3')} | {f('after_round_1')} | "
                  f"{f('panel_optimal')} | {f('room_above_better_baseline_per_question')} | "
                  f"{g0['stated_answer_parsed']:.3f} / {g1['stated_answer_parsed']:.3f} | "
                  f"{g0.get('hit_token_cap', float('nan')):.3f} / {g1.get('hit_token_cap', float('nan')):.3f} | {bhu} |\n")
    rt = rep["runtime"]
    md.append(f"\n**Runtime:** {rt['seconds_per_question_measured']} s/question measured over {n_cond} "
              f"conditions; projected {rt['seconds_per_question_projected']} s/question for "
              f"{args.main_conditions}; GPU-hours {rt['projected_gpu_hours']}.\n")
    md.append(f"\n**Disk:** projected GB {rep['disk']['projected_gb']}.\n")
    md.append("\n**Power** (test questions needed, 80% power, two-sided 5%):\n\n| discordant | effect | test q | total q |\n|---|---|---|---|\n")
    for r_ in rows:
        md.append(f"| {r_['discordant_fraction']:.2f} | {r_['effect_points']:.0f} pts | "
                  f"{r_['test_questions_needed']} | {r_['total_questions_at_test_fraction']} |\n")
    (out / "report.md").write_text("".join(md))
    print("".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
