"""Answer outcomes of a collected job, per question and paired across conditions (CPU only).

    python scripts/outcomes.py --out-root runs/colab --name pilot12b --out artifacts/pilot12b/outcomes

For every condition: the original vote of three first answers, the vote after round 1, the vote
of all six answers, and the panel-optimal ceiling (best of <= 8 accept patterns), per question,
with paired differences and 95% intervals from resampling questions. Debate against answer-only
and against self-reconsideration directly. Everything twice: on all questions (primary) and on a
sensitivity subset that excludes a question if any of its first answers hit the token cap or fell
back to the readout (round 0 is shared, so the subset is the same for every condition).

Revisions (beneficial / harmful / unhelpful) are decomposed by the first answers' agreement
pattern, whether the agent's correct first answer was a minority, whether the peers it was shown
were wrong and in the majority, whether it adopted a shown peer's answer, and whether any first
answer was correct; with subject counts and a comparison of the collection sessions (chunks).
Exports pivotal question ids and a small balanced case study. Descriptive: the pilot does not
test the activation hypothesis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.analysis.labels import apply_policy, build_events, panel_vote  # noqa: E402
from mad.analysis.oracles import panel_optimal_accept, policy_accept  # noqa: E402
from mad.debate import ROWS_FILE  # noqa: E402
from mad.job import view_dirs  # noqa: E402

LET = "ABCDEFGHIJ"


def boot(diff: np.ndarray, n_boot: int = 10000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    return [float(x) for x in np.quantile(diff[idx].mean(1), [0.025, 0.975])]


def per_question(rows: pd.DataFrame, tie: str) -> pd.DataFrame:
    ev = build_events(rows[rows.phase.isin(["pre", "post"])])
    orig = apply_policy(ev, policy_accept(ev, "round0_plurality"), tie).set_index("question_id")
    final = apply_policy(ev, policy_accept(ev, "mad_standard"), tie).set_index("question_id")
    best = apply_policy(ev, panel_optimal_accept(ev, tie), tie).set_index("question_id")
    out = pd.DataFrame(index=orig.index)
    out["original_correct"] = orig.panel_correct
    out["final_correct"] = final.panel_correct.reindex(out.index)
    out["oracle_correct"] = best.panel_correct.reindex(out.index)
    pooled = {}
    for q, g in ev.sort_values("agent_id").groupby("question_id"):
        answers = g.pre_answer.tolist() + g.post_answer.tolist()
        conf = g.pre_confidence.tolist() + g.post_confidence.tolist()
        pooled[q] = int(panel_vote(answers, conf, g.pre_answer.tolist(), tie) == g.gold_display.iloc[0])
    out["pooled6_correct"] = pd.Series(pooled)
    out["any_first_correct"] = ev.groupby("question_id").pre_correct.max()
    out["any_revised_correct"] = ev.groupby("question_id").post_correct.max()
    out["new_correct_discovery"] = ((out.any_first_correct == 0) & (out.any_revised_correct == 1)).astype(int)
    out["gain"] = ((out.original_correct == 0) & (out.final_correct == 1)).astype(int)
    out["loss"] = ((out.original_correct == 1) & (out.final_correct == 0)).astype(int)
    return out


def summary(pq: dict[str, pd.DataFrame], qset) -> dict:
    res = {"n_questions": len(qset), "conditions": {}, "paired": {}}
    base = next(iter(pq.values())).loc[qset]
    res["original_vote_correct"] = int(base.original_correct.sum())
    res["any_first_answer_correct"] = int(base.any_first_correct.sum())
    for c, t in pq.items():
        t = t.loc[qset]
        res["conditions"][c] = {
            "final_vote_correct": int(t.final_correct.sum()), "pooled6_correct": int(t.pooled6_correct.sum()),
            "oracle_correct": int(t.oracle_correct.sum()), "gains": int(t.gain.sum()), "losses": int(t.loss.sum()),
            "new_correct_discoveries": int(t.new_correct_discovery.sum()),
            "final_minus_original": {"mean": float((t.final_correct - t.original_correct).mean()),
                                     "ci95": boot((t.final_correct - t.original_correct).to_numpy(float))},
            "pooled6_minus_original": {"mean": float((t.pooled6_correct - t.original_correct).mean()),
                                       "ci95": boot((t.pooled6_correct - t.original_correct).to_numpy(float))},
        }
    pairs = [("debate", "debate_answer_only"), ("debate", "self_reconsider"),
             ("debate_answer_only", "self_reconsider"), ("debate", "misleading_one")]
    for a, b in pairs:
        if a in pq and b in pq:
            d = (pq[a].loc[qset].final_correct - pq[b].loc[qset].final_correct).to_numpy(float)
            res["paired"][f"final: {a} - {b}"] = {"mean": float(d.mean()), "ci95": boot(d),
                                                 f"{a}_only_correct": int((d > 0).sum()),
                                                 f"{b}_only_correct": int((d < 0).sum())}
    return res


def decompose(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per revision event, with the factors it is decomposed by."""
    ev = build_events(rows[rows.phase.isin(["pre", "post"])])
    post = rows[rows.phase == "post"].set_index(["question_id", "agent_id"])
    pre_tab = ev.pivot_table(index="question_id", columns="agent_id", values="pre_answer", aggfunc="first")
    out = []
    for e in ev.itertuples():
        firsts = pre_tab.loc[e.question_id].tolist()
        n_distinct = len(set(firsts))
        pattern = {1: "unanimous", 2: "2-1 split", 3: "1-1-1 split"}[n_distinct]
        others = [x for i, x in enumerate(firsts) if i != e.agent_id]
        p = post.loc[(e.question_id, e.agent_id)]
        shown = [int(x) for x in str(p.get("shown_peer_answers", "")).split(",") if x not in ("", "nan", "-1")]
        maj = int(p.get("shown_peer_majority", -1)) if pd.notna(p.get("shown_peer_majority", -1)) else -1
        out.append({
            "question_id": e.question_id, "agent_id": e.agent_id, "subject": e.subject, "label": e.label,
            "chunk": p.get("chunk", ""), "first_answers": pattern,
            "correct_minority": int(e.pre_correct == 1 and all(o != e.pre_answer for o in others)),
            "any_first_correct": int(any(f == e.gold_display for f in firsts)),
            "shown_majority_wrong": int(maj >= 0 and maj != e.gold_display),
            "adopted_shown_answer": int(e.revised == 1 and e.post_answer in shown),
        })
    return pd.DataFrame(out)


def case_study(out_root, name, rows_by_cond, pq, k=3):
    """k debate gains and k debate losses, full question / options / answers / shown peers."""
    qs = {json.loads(x)["question_id"]: json.loads(x) for x in
          (Path(out_root) / f"{name}-questions.jsonl").read_text().splitlines()}
    r = rows_by_cond["debate"]
    t = pq["debate"]
    picks = list(t[t.gain == 1].index[:k]) + list(t[t.loss == 1].index[:k])
    md = [f"# Case study: {k} debate gains and {k} debate losses (panel outcome changed)\n"]
    for q in picks:
        rq = r[r.question_id == q]
        perm = [int(x) for x in rq.permutation.iloc[0].split(",")]
        opts = [qs[q]["options"][i] for i in perm]
        gold = int(rq.gold_display.iloc[0])
        md.append(f"\n## {q} ({qs[q]['subject']}) -- {'GAIN' if t.loc[q, 'gain'] else 'LOSS'}\n\n{qs[q]['question']}\n")
        md.append("\n".join(f"- ({LET[i]}) {o}{'  **[gold]**' if i == gold else ''}" for i, o in enumerate(opts)))
        md.append("\n\n| agent | first | shown peers (order: source:answer) | revised |\n|---|---|---|---|")
        for a in sorted(rq.agent_id.unique()):
            pre = rq[(rq.agent_id == a) & (rq.phase == "pre")].iloc[0]
            post = rq[(rq.agent_id == a) & (rq.phase == "post")].iloc[0]
            shown = [x for x in str(post.shown_peer_answers).split(",") if x]
            src = str(post.shown_peer_sources).split(",")
            peers = ", ".join(f"{s}:{LET[int(x)] if int(x) >= 0 else 'none'}" for s, x in zip(src, shown, strict=False))
            md.append(f"| {a} | {LET[int(pre.answer)]} | {peers} | {LET[int(post.answer)]} |")
        md.append("")
    return picks, "\n".join(md)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.out_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    job = json.loads((root / f"{args.name}-job.json").read_text())
    tie = job["identity"]["config"]["debate"]["tie_break"]
    views = view_dirs(root, args.name)
    rows_by_cond = {c: pd.read_parquet(d / ROWS_FILE) for c, d in views.items()}
    pq = {c: per_question(r, tie) for c, r in rows_by_cond.items()}
    allq = sorted(next(iter(pq.values())).index)

    pre = next(iter(rows_by_cond.values()))
    pre = pre[pre.phase == "pre"]
    flagged = pre[(pre.stop_reason == "length") | (pre.answer_source == "readout_fallback")]
    excl = sorted(set(flagged.question_id))
    sens = [q for q in allq if q not in set(excl)]
    trunc_post = {c: sorted(set(r[(r.phase == "post") & (r.stop_reason == "length")].question_id))
                  for c, r in rows_by_cond.items()}

    report = {
        "primary_all_questions": summary(pq, allq),
        "sensitivity_excluding_truncated_or_fallback_first_answers": summary(pq, sens),
        "exclusion": {"first_answers_capped_or_fallback": int(len(flagged)), "questions_excluded": len(excl),
                      "question_ids": excl,
                      "revised_answers_capped_by_condition": {c: {"n_questions": len(v), "within_excluded_set": all(q in excl for q in v), "question_ids": v}
                                                              for c, v in trunc_post.items()}},
    }
    # decompositions
    dec = pd.concat([decompose(r).assign(condition=c) for c, r in rows_by_cond.items()], ignore_index=True)
    revs = dec[dec.label != "none"]
    tables = {}
    for factor in ("first_answers", "correct_minority", "any_first_correct", "shown_majority_wrong",
                   "adopted_shown_answer", "subject", "chunk"):
        tab = revs.groupby(["condition", factor, "label"]).size().unstack("label", fill_value=0)
        tables[factor] = {f"{c}|{v}": row.to_dict() for (c, v), row in tab.iterrows()}
    report["revision_decomposition"] = tables
    sess = []
    for c, r in rows_by_cond.items():
        ev = build_events(r[r.phase.isin(["pre", "post"])])
        ch = r[r.phase == "post"].set_index(["question_id", "agent_id"]).chunk
        ev["session"] = [("session 2 (c000-c002)" if ch.loc[(q, a)] in ("c000", "c001", "c002")
                          else "session 3 (c003-c005)") for q, a in zip(ev.question_id, ev.agent_id, strict=True)]
        for s, g in ev.groupby("session"):
            sess.append({"condition": c, "session": s, "agent_first_acc": round(g.pre_correct.mean(), 3),
                         "agent_revised_acc": round(g.post_correct.mean(), 3), "revision_rate": round(g.revised.mean(), 3),
                         **{k: int((g.label == k).sum()) for k in ("beneficial", "harmful", "unhelpful")}})
    report["sessions"] = sess
    report["pivotal_question_ids"] = {c: {"gains": sorted(t[t.gain == 1].index), "losses": sorted(t[t.loss == 1].index)}
                                      for c, t in pq.items()}
    picks, md = case_study(root, args.name, rows_by_cond, pq)
    report["case_study_question_ids"] = picks

    (out / "outcomes.json").write_text(json.dumps(report, indent=2, default=str))
    pd.concat([t.assign(condition=c) for c, t in pq.items()]).reset_index().rename(columns={"index": "question_id"}).to_csv(
        out / "per_question.csv", index=False)
    dec.to_csv(out / "revision_events.csv", index=False)
    (out / "case_study.md").write_text(md)
    p = report["primary_all_questions"]
    print(f"original vote {p['original_vote_correct']}/{p['n_questions']}; any first answer correct {p['any_first_answer_correct']}")
    for c, v in p["conditions"].items():
        print(f"  {c:20s} final {v['final_vote_correct']}  pooled6 {v['pooled6_correct']}  oracle {v['oracle_correct']}  "
              f"gains {v['gains']} losses {v['losses']}  new correct {v['new_correct_discoveries']}  "
              f"final-orig {v['final_minus_original']['mean']:+.3f} {np.round(v['final_minus_original']['ci95'], 3).tolist()}")
    for k, v in p["paired"].items():
        print(f"  {k}: {v['mean']:+.3f} {np.round(v['ci95'], 3).tolist()}")
    s = report["sensitivity_excluding_truncated_or_fallback_first_answers"]
    print(f"sensitivity: {s['n_questions']} questions; original {s['original_vote_correct']}; "
          + ", ".join(f"{c} {v['final_vote_correct']}" for c, v in s["conditions"].items()))
    print("excluded:", report["exclusion"]["first_answers_capped_or_fallback"], "first answers on",
          report["exclusion"]["questions_excluded"], "questions; revised capped:",
          {c: (v["n_questions"], v["within_excluded_set"]) for c, v in report["exclusion"]["revised_answers_capped_by_condition"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
