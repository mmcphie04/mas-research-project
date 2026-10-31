"""EXPLORATORY follow-ups to the cross-model judge run (2026-10-31 request; CPU only).

    python scripts/judge_followup.py --runs runs/judge_full --out artifacts/revision_gate/judge

1. Mixed 3-answer panels on the 149 judge questions (exact expected accuracy averaged over sample
   choices): GGG, QQQ, LLL, QQG, QGG; differences from GGG and from QQQ. Qwen3-thinking has one own
   answer per question, so no QQQ-thinking panel exists; its single-answer accuracy is reported
   next to Qwen3's single-sample accuracy on the same 44 questions. Like-for-like error
   statistics at the pair level (one sample from each side, within a model or across two):
   correlation of correctness and same-wrong-answer rate for every model pair.
2. Per judge: accuracy on helpful/harmful events split by whether the judge model's own majority
   answer equals the original answer, the revision, or neither.
3. The 44-question thinking subsample: absolute accuracy of Gemma's panels accepting every
   revision, the same panels gated by the thinking judge, and Qwen3-thinking's single own answer;
   with measured seconds per question for each (question-level means: a question's panels are
   averaged first).
Question-grouped 95% intervals, 10,000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from judge_analysis import (  # noqa: E402
    JUDGES,
    decision_score,
    events_table,
    find,
    load_judge,
    load_own,
    majority,
)
from revision_gate import cluster_mean  # noqa: E402

from mad.analysis import revision_events as R  # noqa: E402
from mad.analysis.actions import expected_vote  # noqa: E402

N_BOOT, LABEL = 10000, "EXPLORATORY"
STATUS = {"pilot12b": ("runs/colab/pilot12b-status.json", 3), "val12b": ("runs/val12b/val12b-status.json", 3),
          "pilot250": ("runs/pilot250/out/pilot250-status.json", 10), "main250": ("runs/main250/main250-status.json", 10)}


def _avg(panels, gold) -> float:
    return float(np.mean([expected_vote(list(p), gold) for p in panels]))


def mixed_panels(t: pd.DataFrame, owns: dict) -> dict:
    gold = t.drop_duplicates("question_id").set_index("question_id").gold.to_dict()
    G, Q, L, T = (owns[m] for m in ("gemma", "qwen3", "llama", "qwen3_think"))
    qs = [q for q in sorted(gold) if len(G.get(q, [])) >= 3 and len(Q.get(q, [])) >= 3 and len(L.get(q, [])) >= 3]
    sc = {k: [] for k in ("GGG", "QQQ", "LLL", "QQG", "QGG")}
    for q in qs:
        g, qq, ll, gd = G[q][:3], Q[q][:3], L[q][:3], gold[q]
        sc["GGG"].append(expected_vote(g, gd))
        sc["QQQ"].append(expected_vote(qq, gd))
        sc["LLL"].append(expected_vote(ll, gd))
        sc["QQG"].append(_avg([(*a, b) for a in itertools.combinations(qq, 2) for b in g], gd))
        sc["QGG"].append(_avg([(*a, b) for a in itertools.combinations(g, 2) for b in qq], gd))
    qa = np.array(qs)
    out = {"questions": len(qs), "accuracy": {k: cluster_mean(np.array(v), qa) for k, v in sc.items()},
           "minus_GGG": {k: cluster_mean(np.array(v) - np.array(sc["GGG"]), qa) for k, v in sc.items() if k != "GGG"},
           "minus_QQQ": {k: cluster_mean(np.array(v) - np.array(sc["QQQ"]), qa) for k, v in sc.items() if k != "QQQ"}}
    tq = [q for q in qs if T.get(q)]
    single = {"qwen3_think_single": [float(T[q][0] == gold[q]) for q in tq],
              "qwen3_single_mean_of_3": [float(np.mean([a == gold[q] for a in Q[q][:3]])) for q in tq],
              "gemma_single_mean_of_3": [float(np.mean([a == gold[q] for a in G[q][:3]])) for q in tq]}
    out["thinking_subsample_single_answers"] = {"questions": len(tq), **{k: cluster_mean(np.array(v), np.array(tq))
                                                                        for k, v in single.items()},
                                                "note": "one thinking sample per question: no QQQ-thinking panel"}
    return out


def pair_stats(t: pd.DataFrame, owns: dict) -> dict:
    """One sample from each side. Within a model: every unordered pair of distinct samples;
    across models: every (sample, sample) pair. Unparsed answers are dropped."""
    gold = t.drop_duplicates("question_id").set_index("question_id").gold.to_dict()
    nopt = t.drop_duplicates("question_id").set_index("question_id").n_options.to_dict()
    names = {"G": "gemma", "Q": "qwen3", "L": "llama"}
    out = {}
    for a, b in [("G", "G"), ("Q", "Q"), ("L", "L"), ("G", "Q"), ("G", "L"), ("Q", "L")]:
        A, B = owns[names[a]], owns[names[b]]
        per_q = []
        for q in sorted(gold):
            xa = [x for x in A.get(q, [])[:3] if x >= 0]
            xb = [x for x in B.get(q, [])[:3] if x >= 0]
            pairs = list(itertools.combinations(xa, 2)) if a == b else [(x, y) for x in xa for y in xb]
            if not pairs:
                continue
            ca = np.array([x == gold[q] for x, _ in pairs], float)
            cb = np.array([y == gold[q] for _, y in pairs], float)
            ww = [(x, y) for x, y in pairs if x != gold[q] and y != gold[q]]
            per_q.append((q, ca, cb, len(ww), sum(x == y for x, y in ww), len(ww) / (nopt[q] - 1)))
        qs = np.array([p[0] for p in per_q])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(per_q), (N_BOOT, len(per_q)))

        def corr(sel, per_q=per_q, a=a, b=b):
            ca = np.concatenate([per_q[i][1] for i in sel])
            cb = np.concatenate([per_q[i][2] for i in sel])
            if a == b:  # symmetric: both orders
                ca, cb = np.concatenate([ca, cb]), np.concatenate([cb, ca])
            return float(np.corrcoef(ca, cb)[0, 1])

        ww = np.array([p[3] for p in per_q], float)
        same = np.array([p[4] for p in per_q], float)
        exp_ = np.array([p[5] for p in per_q], float)
        cb_ = np.array([corr(r) for r in idx])
        rate_b = same[idx].sum(1) / np.maximum(ww[idx].sum(1), 1)
        out[f"{a}-{b}"] = {"questions": int(len(qs)),
                           "correlation_of_correctness": {"mean": corr(range(len(per_q))), "lo": float(np.nanquantile(cb_, .025)),
                                                          "hi": float(np.nanquantile(cb_, .975))},
                           "same_wrong_answer_rate": {"mean": float(same.sum() / ww.sum()), "lo": float(np.quantile(rate_b, .025)),
                                                      "hi": float(np.quantile(rate_b, .975)), "wrong_pairs": int(ww.sum())},
                           "uniform_expected_rate": float(exp_.sum() / ww.sum())}
    return out


def by_own_majority(t: pd.DataFrame, judges: dict, owns: dict) -> dict:
    out = {}
    for m, j in judges.items():
        x = t.join(j, on="event_id", how="inner")
        x = x[x.kind.isin(["helpful", "harmful"])].copy()
        maj = np.array([majority(owns[m].get(q, [])) for q in x.question_id])
        x["own"] = np.where(maj == x.after.to_numpy(), "revision", np.where(maj == x.before.to_numpy(), "original", "neither"))
        x["score"] = decision_score(x.p_rev.to_numpy(), (x.kind == "helpful").to_numpy(int))
        out[m] = {}
        for grp in ("original", "revision", "neither"):
            g = x[x.own == grp]
            out[m][grp] = {"events": int(len(g)), "helpful": int((g.kind == "helpful").sum()),
                           "accuracy": cluster_mean(g.score.to_numpy(), g.question_id.to_numpy()) if len(g) else {"n": 0}}
    return out


def panel_seconds() -> dict:
    """Measured Gemma seconds per question for a panel of each source condition: the round-0
    stage shared out per agent (k of the job's round-0 agents) plus the condition's own stages."""
    out = {}
    for d, (f, n0) in STATUS.items():
        s = json.loads((ROOT / f).read_text())
        nq = sum(t["questions"] for t in s["timings"])
        r0 = sum(t["stages"].get("round0_s", 0.0) for t in s["timings"]) / nq / n0  # per question per agent
        conds = {k.split("/")[0] for t in s["timings"] for k in t["stages"] if "/" in k}
        for c in conds:
            stage = sum(sum(v for k, v in t["stages"].items() if k.startswith(c + "/") and k.endswith("_s"))
                        for t in s["timings"]) / nq
            out[(d, c)] = {"round0_per_agent": r0, "condition": stage}
    return out


def thinking_subsample(t: pd.DataFrame, root: Path, judges: dict, owns: dict, runs: Path) -> dict:
    sub = t[t.thinking_subsample]
    qset = set(sub.question_id)
    ev = R.load_events(root)["events"]
    ev = ev[ev.question_id.isin(qset) & ev.apply(lambda r: (r.dataset, r.condition) in R.TRAIN_SOURCES, axis=1)]
    jp = judges["qwen3_think"].p_rev.to_dict()
    secs = panel_seconds()
    js = json.loads(next(iter(find(runs, "qwen3_think_summary.json"))).read_text())
    judge_s_per_event = js["judge"]["seconds"] / js["events"]
    own_s_per_q = js["own"]["seconds"] / js["own"]["responses"]
    rows = []
    for (d, c, q), g in ev.groupby(["dataset", "condition", "question_id"]):
        g = g.sort_values("agent_id")
        b, a, gold = g.before.tolist(), g.after.tolist(), int(g.gold.iloc[0])
        ids = [f"{d}|{c}|{q}|{x}" for x in g.agent_id]
        ch = [i for i in range(len(b)) if b[i] != a[i]]
        p = [0.0] * len(b)
        for i in ch:
            v = jp[ids[i]]
            p[i] = 1.0 if v > 0.5 else 0.0 if v < 0.5 else 0.5
        k = len(b)
        sp = secs[(d, c)]
        gem_s = k * sp["round0_per_agent"] + sp["condition"]
        rows.append({"question_id": q, "accept_all": expected_vote(a, gold), "gated": R.panel_expectation(b, a, gold, p)[1],
                     "gemma_seconds": gem_s, "gated_seconds": gem_s + len(ch) * judge_s_per_event,
                     "own": float(owns["qwen3_think"][q][0] == gold) if owns["qwen3_think"].get(q) else np.nan})
    df = pd.DataFrame(rows).groupby("question_id").mean()  # a question's panels averaged first
    qs = df.index.to_numpy()
    return {"questions": int(len(df)), "panels": len(rows),
            "accuracy": {"gemma_accept_all": cluster_mean(df.accept_all.to_numpy(), qs),
                         "gemma_gated_by_thinking_judge": cluster_mean(df.gated.to_numpy(), qs),
                         "qwen3_thinking_single_own_answer": cluster_mean(df.own.to_numpy(), qs)},
            "differences": {"gated_minus_accept_all": cluster_mean((df.gated - df.accept_all).to_numpy(), qs),
                            "own_minus_accept_all": cluster_mean((df.own - df.accept_all).to_numpy(), qs),
                            "own_minus_gated": cluster_mean((df.own - df.gated).to_numpy(), qs)},
            "seconds_per_question": {"gemma_accept_all": float(df.gemma_seconds.mean()),
                                     "gemma_gated_by_thinking_judge": float(df.gated_seconds.mean()),
                                     "qwen3_thinking_single_own_answer": own_s_per_q},
            "seconds_basis": {"gemma": "source jobs' measured stage times (round 0 shared per agent + the condition's "
                                       "stages), averaged over the question's panels",
                              "judge_seconds_per_changed_answer": judge_s_per_event,
                              "own": "Qwen3-thinking own-answer step seconds / responses (judge run)"}}


def ci(x, nd=3):
    return "—" if not x or "mean" not in x else f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"


def markdown(r) -> str:
    m = r["mixed_panels"]
    L = [f"# Cross-model judge: follow-ups ({LABEL})", "", "Question-grouped 95% intervals (10,000 resamples). "
         "Own answers at the 250-word protocol; Gemma's include pool agents 5–7 for the 42 main250/pilot250 questions.",
         "", f"## 1. Mixed 3-answer panels ({m['questions']} questions)", "",
         "| panel | accuracy | − GGG | − QQQ |", "|---|---|---|---|"]
    for k in ("GGG", "QQQ", "LLL", "QQG", "QGG"):
        L.append(f"| {k} | {ci(m['accuracy'][k])} | {ci(m['minus_GGG'].get(k))} | {ci(m['minus_QQQ'].get(k))} |")
    s = m["thinking_subsample_single_answers"]
    L += ["", f"QQQ-thinking is not possible (one thinking sample per question). Single answers on the "
          f"{s['questions']} thinking questions: Qwen3-thinking {ci(s['qwen3_think_single'])}; Qwen3 "
          f"{ci(s['qwen3_single_mean_of_3'])} (mean of its 3 samples); Gemma {ci(s['gemma_single_mean_of_3'])}.", "",
          "**Pair-level error statistics** (one sample from each side; within a model: distinct samples):", "",
          "| pair | questions | correlation of correctness | same-wrong-answer rate | wrong pairs | uniform expectation |",
          "|---|---|---|---|---|---|"]
    for k, v in r["pair_stats"].items():
        L.append(f"| {k} | {v['questions']} | {ci(v['correlation_of_correctness'])} | {ci(v['same_wrong_answer_rate'])} | "
                 f"{v['same_wrong_answer_rate']['wrong_pairs']} | {v['uniform_expected_rate']:.3f} |")
    L += ["", "## 2. Judge accuracy by the judge model's own majority answer", "",
          "| judge | own = original | own = revision | own = neither (incl. no majority) |", "|---|---|---|---|"]
    for j, v in r["by_own_majority"].items():
        L.append(f"| {j} | " + " | ".join(f"{ci(v[g]['accuracy'])} (n={v[g]['events']}, helpful {v[g]['helpful']})"
                                         for g in ("original", "revision", "neither")) + " |")
    t = r["thinking_subsample"]
    L += ["", f"## 3. Thinking subsample ({t['questions']} questions, {t['panels']} Gemma panels)", "",
          "| method | accuracy | seconds per question |", "|---|---|---|"]
    for k, lab in (("gemma_accept_all", "Gemma panels, accept every revision"),
                   ("gemma_gated_by_thinking_judge", "Gemma panels, gated by the Qwen3-thinking judge"),
                   ("qwen3_thinking_single_own_answer", "Qwen3-thinking, single own answer")):
        L.append(f"| {lab} | {ci(t['accuracy'][k])} | {t['seconds_per_question'][k]:.0f} |")
    d = t["differences"]
    L += ["", f"Differences: gated − accept-all {ci(d['gated_minus_accept_all'])}; own − accept-all "
          f"{ci(d['own_minus_accept_all'])}; own − gated {ci(d['own_minus_gated'])}. Seconds: Gemma from the source jobs' "
          f"measured stage times (round 0 shared per agent plus the panel's condition); the judge adds "
          f"{t['seconds_basis']['judge_seconds_per_changed_answer']:.1f} s per changed answer (both orders, measured); "
          "the own answer is the judge run's measured thinking own-answer time. Different jobs and protocols "
          "(150/250 words), all on an A100."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    t = events_table(ROOT)
    judges = {m: load_judge(runs, m) for m in JUDGES}
    owns = {m: load_own(runs, m, ROOT) for m in JUDGES}
    r = {"label": LABEL, "mixed_panels": mixed_panels(t, owns), "pair_stats": pair_stats(t, owns),
         "by_own_majority": by_own_majority(t, judges, owns),
         "thinking_subsample": thinking_subsample(t, ROOT, judges, owns, runs)}
    out = Path(a.out)
    (out / "followup.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    (out / "followup.md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
