"""Primary analysis of the main collection, exactly as docs/MAIN250_ANALYSIS_SPEC.md fixes it.

    python scripts/main250_primary.py --dest runs/main250 --job main250 --out artifacts/main250/analysis

All 200 questions (the train/validation/test split is NOT used here). Unit: the question; every
comparison is paired by question. Panel decision: plurality of the agents' LATEST answers, scored
as exact expected accuracy under uniformly random tie-breaking (actions.expected_vote). Answers:
canonical (stated, readout fallback where none is stated); the fallback rate is reported.

Primary (6 contrasts, no multiplicity correction, number stated with the results):
  1. debate (k agents, 2k responses) vs independent voting over 2k, k = 3, 5, ordinary and structured;
  2. structured vs ordinary debate at k = 3 and k = 5.
Descriptive: the independent-voting curve over 3..10 answers; revision events per agent and per
panel. Cost, measured: input/output tokens and generation seconds, separately from readouts.
Writes primary.json and primary.md. Nothing else is computed here; anything else is exploratory.
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

from mad.analysis.panels import (  # noqa: E402
    MATCHED,
    compare,
    debate_table,
    independent_curve,
    paired,
)
from mad.job import chunk_tag  # noqa: E402

DEBATES = {"debate3": 3, "structured3": 3, "debate5": 5, "structured5": 5}
N_BOOT, SEED = 10000, 0
COLS = ["question_id", "agent_id", "phase", "answer", "answer_source", "gold_display", "n_prompt_tokens",
        "n_completion_tokens", "stop_reason"]


def _states(dest: Path, job: str, cond: str) -> pd.DataFrame:
    s = pd.read_parquet(dest / f"{job}-{cond}" / "agent_states.parquet", columns=COLS)
    s["question_id"] = s.question_id.astype(str)
    return s


def _by_agent(s: pd.DataFrame) -> dict[str, list[int]]:
    return {q: g.sort_values("agent_id").answer.astype(int).tolist() for q, g in s.groupby("question_id")}


def load(dest: Path, job: str):
    pool_s = _states(dest, job, "pool")
    pool_pre = pool_s[pool_s.phase == "pre"]
    gold_all = [pool_pre[["question_id", "gold_display"]]]
    pool = {q: a for q, a in _by_agent(pool_pre).items() if len(a) == 10}
    states = {"pool": pool_pre}
    revised, missing = {}, {"pool": sorted(set(pool_pre.question_id) - set(pool))}
    for cond, k in DEBATES.items():
        s = _states(dest, job, cond)
        pre, post = s[s.phase == "pre"], s[s.phase == "post"]
        gold_all.append(s[["question_id", "gold_display"]])
        for q, a in _by_agent(pre).items():  # the debate's initial answers ARE the pool's first k
            if q in pool and a != pool[q][:k]:
                raise SystemExit(f"{cond} {q}: initial answers {a} != pool agents 0..{k - 1} {pool[q][:k]}")
        rev = {q: a for q, a in _by_agent(post).items() if len(a) == k}
        missing[cond] = sorted(set(pool) - set(rev))
        revised[cond] = rev
        states[cond] = post
    g = pd.concat(gold_all).drop_duplicates()
    if g.question_id.duplicated().any():
        raise SystemExit("a question's answer key differs between conditions (option order not shared)")
    gold = dict(zip(g.question_id, g.gold_display.astype(int), strict=True))
    return pool, revised, gold, states, missing


def boot_mean(x: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def cost(dest: Path, job: str, states: dict, n_chunks: int) -> dict:
    """Measured: tokens per response (exact) and stage seconds from the completion records and the
    job status. Round 0 (the pool's 10 answers) records one stage time covering generation AND
    its readouts; it is reported as such and apportioned per response only where labelled."""
    status = json.loads((dest / f"{job}-status.json").read_text())
    round0_s = sum(t["stages"].get("round0_s", 0.0) for t in status["timings"])
    out = {"round0_stage_s_total": round(round0_s, 1), "round0_note": "generation + readouts of all 10 "
           "independent answers; not separable in the records", "conditions": {}}
    p = states["pool"]
    out["conditions"]["pool"] = {"responses": len(p), "input_tokens": int(p.n_prompt_tokens.sum()),
                                 "output_tokens": int(p.n_completion_tokens.sum()),
                                 "stage_s_per_response_incl_readout": round(round0_s / len(p), 2)}
    for cond in DEBATES:
        t = {"generation_s": 0.0, "readout_s": 0.0, "exposure_readout_s": 0.0, "prompt_tokens": 0,
             "generated_tokens": 0}
        for i in range(n_chunks):
            rec = json.loads((dest / f"{job}-{cond}-{chunk_tag(i)}" / "completion.json").read_text())["timings"]
            for key in t:
                t[key] += rec.get(key, 0)
        s = states[cond]
        out["conditions"][cond] = {
            "revisions": len(s), "input_tokens": int(s.n_prompt_tokens.sum()),
            "output_tokens": int(s.n_completion_tokens.sum()),
            "generation_s": round(t["generation_s"], 1),
            "generation_s_per_revision": round(t["generation_s"] / len(s), 2),
            "extraction_s": round(t["readout_s"] + t["exposure_readout_s"], 1),
            "recorded_prompt_tokens": t["prompt_tokens"], "recorded_generated_tokens": t["generated_tokens"]}
    return out


def arm_tokens(states: dict, cond: str | None, k: int, n_independent: int | None) -> dict:
    """Tokens per question for one arm of a matched contrast: a debate arm is its k initial
    answers (pool agents 0..k-1) plus its k revisions; an independent arm is pool agents 0..n-1."""
    p = states["pool"]
    if cond is None:
        x = p[p.agent_id < n_independent]
        return {"input_tokens_per_q": float(x.groupby("question_id").n_prompt_tokens.sum().mean()),
                "output_tokens_per_q": float(x.groupby("question_id").n_completion_tokens.sum().mean())}
    init, rev = p[p.agent_id < k], states[cond]
    f = lambda c: float(init.groupby("question_id")[c].sum().mean() + rev.groupby("question_id")[c].sum().mean())  # noqa: E731
    return {"input_tokens_per_q": f("n_prompt_tokens"), "output_tokens_per_q": f("n_completion_tokens")}


def fallback(states: dict) -> dict:
    out = {}
    for cond, s in states.items():
        out[cond] = {"answers": len(s), "readout_fallback": int((s.answer_source == "readout_fallback").sum()),
                     "rate": round(float((s.answer_source == "readout_fallback").mean()), 4),
                     "scope": "the 10 independent answers" if cond == "pool" else "the revised (post-debate) answers"}
    return out


def analyze(dest: Path, job: str) -> dict:
    pool, revised, gold, states, missing = load(dest, job)
    jobrec = json.loads((dest / f"{job}-job.json").read_text())
    n_chunks = len([c for c in jobrec["chunks"] if c])
    qs = sorted(set(pool).intersection(*[set(r) for r in revised.values()]))
    pool_q = {q: pool[q] for q in qs}
    res = compare(pool_q, gold, {c: (k, {q: revised[c][q] for q in qs}) for c, k in DEBATES.items()}, n_boot=N_BOOT)
    ind = independent_curve(pool_q, gold).pivot(index="question_id", columns="k", values="acc")
    curve = {int(k): {"mean": float(ind[k].mean()), "ci95": boot_mean(ind[k].to_numpy(float))} for k in ind.columns}
    tabs = {c: debate_table(pool_q, {q: revised[c][q] for q in qs}, gold, c, k).set_index("question_id")
            for c, k in DEBATES.items()}
    primary = {}
    for c, k in DEBATES.items():
        n = MATCHED[k]
        primary[f"{c} vs independent {n}"] = {
            **paired(tabs[c].acc_after, ind[n], N_BOOT, SEED), "debate_acc": float(tabs[c].acc_after.mean()),
            "independent_acc": float(ind[n].mean()),
            "cost_debate": arm_tokens(states, c, k, None), "cost_independent": arm_tokens(states, None, k, n)}
    for k in (3, 5):
        primary[f"structured{k} vs debate{k}"] = {
            **paired(tabs[f"structured{k}"].acc_after, tabs[f"debate{k}"].acc_after, N_BOOT, SEED),
            "structured_acc": float(tabs[f"structured{k}"].acc_after.mean()),
            "debate_acc": float(tabs[f"debate{k}"].acc_after.mean()),
            "cost_structured": arm_tokens(states, f"structured{k}", k, None),
            "cost_debate": arm_tokens(states, f"debate{k}", k, None)}
    events = {}
    for c, k in DEBATES.items():
        d, t = res["debate"][c], tabs[c]
        events[c] = {"k": k, "questions": len(t), "panel_acc_before": d["acc_before"], "panel_acc_after": d["acc_after"],
                     "net_expected_accuracy_change": d["vs_initial_panel"],
                     "per_agent": {"revisions": k * len(t), "changed": d["changed"], "helpful": d["helpful"],
                                   "harmful": d["harmful"]},
                     "per_panel": {"correct_minority_recovered": d["minority_correct_recovered"],
                                   "correct_decision_preserved": d["correct_decision_preserved"],
                                   "correct_decision_lost": d["correct_decision_lost"],
                                   "correct_absent_at_start_but_found": d["absent_correct_found"],
                                   "correct_present_at_start": float(t.gold_present_initially.sum())}}
    return {"spec": "docs/MAIN250_ANALYSIS_SPEC.md", "job": job, "questions_analyzed": len(qs),
            "missing_by_condition": {c: len(v) for c, v in missing.items()},
            "bootstrap": {"resamples": N_BOOT, "seed": SEED, "interval": "95% percentile, questions resampled"},
            "primary_contrasts": primary, "n_primary_contrasts": len(primary),
            "independent_curve": curve, "revision_events": events, "fallback": fallback(states),
            "cost": cost(dest, job, states, n_chunks)}


def fmt(x, nd=3):
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def markdown(r: dict) -> str:
    L = [f"# main250 primary analysis ({r['questions_analyzed']} questions)", "",
         f"Specification: `{r['spec']}`. Score: exact expected accuracy of the panel's plurality vote "
         "under random tie-breaking, per question; differences are paired by question. Intervals: 95% "
         f"percentile, {r['bootstrap']['resamples']:,} question resamples (seed {r['bootstrap']['seed']}). "
         f"{r['n_primary_contrasts']} primary contrasts, no multiplicity correction. With 200 questions, "
         "small differences are not expected to be resolvable; an interval covering 0 is not evidence of no effect.",
         "", f"Missing questions by condition: {r['missing_by_condition']}", "",
         "## Primary contrasts", "",
         "| contrast | arm A | arm B | mean diff (A − B) | 95% interval | questions differing | expected gain | expected loss | tokens in/out per q, A | tokens in/out per q, B |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in r["primary_contrasts"].items():
        a = c.get("debate_acc") if "independent" in name else c["structured_acc"]
        b = c.get("independent_acc") if "independent" in name else c["debate_acc"]
        ca = c["cost_debate"] if "independent" in name else c["cost_structured"]
        cb = c["cost_independent"] if "independent" in name else c["cost_debate"]
        L.append(f"| {name} | {a:.3f} | {b:.3f} | {c['mean_diff']:+.3f} | [{c['lo']:+.3f}, {c['hi']:+.3f}] | "
                 f"{c['questions_differing']} | {c['expected_gain']:.2f} | {c['expected_loss']:.2f} | "
                 f"{ca['input_tokens_per_q']:,.0f} / {ca['output_tokens_per_q']:,.0f} | "
                 f"{cb['input_tokens_per_q']:,.0f} / {cb['output_tokens_per_q']:,.0f} |")
    L += ["", "## Independent-voting curve (descriptive)", "", "| answers voted | expected accuracy | 95% interval |",
          "|---|---|---|"]
    for k, v in r["independent_curve"].items():
        L.append(f"| {k} | {v['mean']:.3f} | [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}] |")
    L += ["", "## Revision events", "",
          "| condition | panel before → after | net change [95%] | revisions | changed | helpful | harmful | "
          "correct minority recovered | correct decision preserved | correct decision lost | correct absent, found |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, e in r["revision_events"].items():
        n, a, p = e["net_expected_accuracy_change"], e["per_agent"], e["per_panel"]
        L.append(f"| {c} | {e['panel_acc_before']:.3f} → {e['panel_acc_after']:.3f} | {n['mean_diff']:+.3f} "
                 f"[{n['lo']:+.3f}, {n['hi']:+.3f}] | {a['revisions']} | {a['changed']} | {a['helpful']} | "
                 f"{a['harmful']} | {p['correct_minority_recovered']:.0f} | {p['correct_decision_preserved']:.0f} | "
                 f"{p['correct_decision_lost']:.0f} | {p['correct_absent_at_start_but_found']:.0f} |")
    L += ["", "Panel event counts are questions (fractional panel scores enter as described in "
          "`mad.analysis.panels.panel_events`).", "", "## Stated-answer fallback rate", "",
          "| condition | answers | readout fallback | rate | scope |", "|---|---|---|---|---|"]
    for c, f in r["fallback"].items():
        L.append(f"| {c} | {f['answers']} | {f['readout_fallback']} | {f['rate']:.2%} | {f['scope']} |")
    co = r["cost"]
    L += ["", "## Measured cost", "",
          f"Round 0 (the pool's 10 independent answers per question): {co['round0_stage_s_total']:,.0f} s in total "
          f"({co['round0_note']}).", "",
          "| condition | responses | input tokens | output tokens | generation s | generation s / response | extraction s (readouts) |",
          "|---|---|---|---|---|---|---|"]
    p = co["conditions"]["pool"]
    L.append(f"| pool | {p['responses']} | {p['input_tokens']:,} | {p['output_tokens']:,} | (in round 0) | "
             f"{p['stage_s_per_response_incl_readout']} incl. readout | (in round 0) |")
    for c in DEBATES:
        x = co["conditions"][c]
        L.append(f"| {c} | {x['revisions']} | {x['input_tokens']:,} | {x['output_tokens']:,} | {x['generation_s']:,.0f} | "
                 f"{x['generation_s_per_revision']} | {x['extraction_s']:,.0f} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "runs/main250"))
    ap.add_argument("--job", default="main250")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = analyze(Path(a.dest), a.job)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "primary.json").write_text(json.dumps(r, indent=1) + "\n")
    (out / "primary.md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
