"""Experiment A: pre-registered analysis (docs/EXPAB_PLAN.md, section 3). CPU only.

    python scripts/expa_analysis.py --root runs/expab --job expab --out artifacts/expab/analysis [--split heldout]

Scores are exact expected accuracy: a plurality vote with uniformly random tie-breaking scores
1/k when the gold answer is among k tied leaders. Unparsed answers do not vote.

* Pool plurality: the vote over the arm's pool (``tasks.ARMS``).
* Arm decision: if the pool has fewer than 2 distinct parsed answers the chooser is not called and
  the arm IS the pool plurality; otherwise the vote over its 3 runs' picked answers (unparsed picks
  dropped; if none parsed: the pool plurality, flagged).
* Cost: per part (bank mode or chooser arm), GPU-seconds per request = the part's total measured
  call seconds / its requests; tokens per request = mean prompt + generated tokens. A question's
  arm cost = its pool's samples at their bank mode's per-sample cost + 3 chooser calls if called.
* Self-consistency (SC) of the chooser model: expected plurality accuracy of n of its bank samples,
  averaged over all subsets of size n (exact), at cost n x its per-sample cost; per question,
  interpolated linearly in cost between integer n. Beyond the curve's ends: the end value, flagged.

P1 arm - pool plurality; P2 arm - chooser SC at the arm's matched GPU-seconds (and tokens);
P3 on questions whose candidates include the gold AND a wrong answer: arm accuracy - the chooser
model's expected single-sample accuracy from its bank samples. Decomposition vs the pool vote:
headroom H, recovery R, new-correct I, damage L (R + I - L = mean difference). Also: when no
candidate is correct, how often the final pick equals the pool's plurality answer; position bias.
Headline: the arm with the largest DEVELOPMENT P1 gain, evaluated once on held-out.
Paired question bootstrap, 10,000 resamples, seed 0.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.analysis.actions import expected_vote  # noqa: E402
from mad.gen import archive as A  # noqa: E402
from mad.gen import tasks as T  # noqa: E402

N_BOOT, LABEL = 10000, "PRE-REGISTERED (docs/EXPAB_PLAN.md)"
CHOOSER_MODE = {arm: spec[0] for arm, spec in T.ARMS.items()}


def boot(x, n_boot=N_BOOT) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    b = x[np.random.default_rng(0).integers(0, len(x), (n_boot, len(x)))].mean(1)
    return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def vote_score(answers, gold) -> float:
    a = [int(x) for x in answers if int(x) >= 0]
    return expected_vote(a, gold) if a else 0.0


def sc_curve(answers: list[int], gold: int, n_max: int) -> np.ndarray:
    """Expected plurality accuracy of a uniformly random subset of size n (n = 1..n_max)."""
    out = []
    for n in range(1, n_max + 1):  # every subset (at most C(10, 5) = 252)
        out.append(np.mean([vote_score([answers[i] for i in s], gold)
                            for s in itertools.combinations(range(len(answers)), n)]))
    return np.array(out)


def costs(calls: pd.DataFrame, rows: pd.DataFrame) -> dict:
    """part -> {"seconds": per request, "tokens": per request}."""
    out = {}
    for part, c in calls.groupby("part"):
        r = rows[rows._part == part]
        n = max(len(r), 1)
        out[part] = {"seconds": float(c.seconds.sum()) / n,
                     "tokens": float((r.prompt_tokens + r.generated_tokens).mean()) if len(r) else 0.0}
    return out


def interp(curve: np.ndarray, unit: float, budget: float) -> tuple[float, str]:
    """Linear interpolation of an SC curve indexed by cost n * unit at ``budget``."""
    xs = unit * np.arange(1, len(curve) + 1)
    if budget <= xs[0]:
        return float(curve[0]), "below_curve"
    if budget >= xs[-1]:
        return float(curve[-1]), "beyond_curve"
    return float(np.interp(budget, xs, curve)), ""


def analyze(bank: pd.DataFrame, choose: pd.DataFrame, cost: dict, assign: dict, split: str) -> dict:
    bank = bank[bank.question_id.map(assign) == split]
    choose = choose[choose.question_id.map(assign) == split] if len(choose) else choose
    bidx = T.bank_index(bank)
    gold = bank.groupby("question_id").meta_gold.first().to_dict()
    qids = sorted(gold)
    picks = {(r["meta_arm"], r["question_id"]): [] for r in choose.to_dict("records")} if len(choose) else {}
    for r in choose.to_dict("records") if len(choose) else []:
        picks[(r["meta_arm"], r["question_id"])].append(r)
    out = {}
    for arm, (chooser, spec) in T.ARMS.items():
        if not len(choose) or arm not in set(choose.meta_arm):
            continue
        part = f"choose_{arm}"
        rows = []
        for q in qids:
            pool = T.pool_rows(bidx, q, spec)
            pa = [int(r["answer"]) for r in pool]
            base = vote_score(pa, gold[q])
            called = T.disagreeing(pool)
            runs = picks.get((arm, q), [])
            got = [int(r["pick_answer"]) for r in runs if int(r["pick_answer"]) >= 0]
            score = vote_score(got, gold[q]) if called and got else base
            cands = {int(r["answer"]) for r in pool if r["parsed"]}
            ch_cost = sum(cost[part]["seconds"] for _ in runs) if called else 0.0
            ch_tok = sum(cost[part]["tokens"] for _ in runs) if called else 0.0
            pool_s = sum(k * cost[f"bank_{m}"]["seconds"] for m, k in spec)
            pool_t = sum(k * cost[f"bank_{m}"]["tokens"] for m, k in spec)
            own = [int(r["answer"]) for r in bidx.get((chooser, q), [])]
            sc = sc_curve(own, gold[q], len(own))
            sc_s, flag_s = interp(sc, cost[f"bank_{chooser}"]["seconds"], pool_s + ch_cost)
            sc_t, flag_t = interp(sc, cost[f"bank_{chooser}"]["tokens"], pool_t + ch_tok)
            top = Counter(a for a in pa if a >= 0).most_common()
            plural = {a for a, c in top if c == top[0][1]} if top else set()
            fin = Counter(got).most_common()
            final = {a for a, c in fin if c == fin[0][1]} if fin else set()
            rows.append({"question_id": q, "called": called, "fallback": called and not got, "base": base, "score": score,
                         "gold_present": gold[q] in cands, "mixed": gold[q] in cands and len(cands - {gold[q]}) > 0,
                         "own_single": float(np.mean([a == gold[q] for a in own])) if own else np.nan,
                         "sc_at_seconds": sc_s, "sc_seconds_flag": flag_s, "sc_at_tokens": sc_t, "sc_tokens_flag": flag_t,
                         "cost_seconds": pool_s + ch_cost, "cost_tokens": pool_t + ch_tok,
                         "no_correct_pick_is_plurality": (len(final & plural) / len(final)) if called and got and gold[q] not in cands else np.nan,
                         "positions": [(int(r["pick_position"]), int(r["meta_m"])) for r in runs if int(r["pick_position"]) > 0]})
        d = pd.DataFrame(rows)
        diff = d.score - d.base
        mixed = d[d.mixed & d.called]
        pos = [p for ps in d.positions for p in ps]
        out[arm] = {
            "questions": len(d), "called": int(d.called.sum()), "fallback": int(d.fallback.sum()),
            "accuracy": boot(d.score), "pool_plurality": boot(d.base),
            "P1_minus_pool_plurality": boot(diff), "P1_on_called": boot(diff[d.called]),
            "P2_minus_sc_matched_seconds": boot(d.score - d.sc_at_seconds),
            "P2_minus_sc_matched_tokens": boot(d.score - d.sc_at_tokens),
            "P2_flags": {"seconds": d.sc_seconds_flag.value_counts().to_dict(), "tokens": d.sc_tokens_flag.value_counts().to_dict()},
            "P3_gap": {"questions": len(mixed), "chooser_accuracy": boot(mixed.score), "own_single_accuracy": boot(mixed.own_single),
                       "difference": boot(mixed.score - mixed.own_single)},
            "decomposition": {"H_headroom": float(((d.gold_present) & (d.base < 1)).mean()),
                              "R_recovery": float((diff.clip(lower=0) * (d.gold_present & (d.base < 1))).mean()),
                              "I_new_correct": float((diff.clip(lower=0) * ~d.gold_present).mean()),
                              "L_damage": float((-diff.clip(upper=0)).mean()), "net": float(diff.mean())},
            "no_correct_candidate": boot(d.no_correct_pick_is_plurality.dropna()),
            "position_bias": {"picks": len(pos), "share_first": float(np.mean([p == 1 for p, _ in pos])) if pos else None,
                              "expected_first": float(np.mean([1 / m for _, m in pos])) if pos else None,
                              "share_last": float(np.mean([p == m for p, m in pos])) if pos else None},
            "mean_cost": {"seconds": float(d.cost_seconds.mean()), "tokens": float(d.cost_tokens.mean())}}
    return out


def sc_table(bank: pd.DataFrame, cost: dict, assign: dict, split: str) -> dict:
    bank = bank[bank.question_id.map(assign) == split]
    out = {}
    for mode, g in bank.groupby("mode"):
        gold = g.groupby("question_id").meta_gold.first()
        curves = [sc_curve(list(x.sort_values("agent", key=lambda s: s.astype(int)).answer.astype(int)), gold[q], len(x))
                  for q, x in g.groupby("question_id")]
        m = np.mean(curves, axis=0)
        out[mode] = {"n": list(range(1, len(m) + 1)), "accuracy": [round(float(v), 4) for v in m],
                     "seconds_per_sample": cost[f"bank_{mode}"]["seconds"], "tokens_per_sample": cost[f"bank_{mode}"]["tokens"]}
    return out


def load(root: Path, job: str):
    rows, calls = [], A.read_calls(root, job)
    for part in sorted({c for c in calls.part}):
        r = A.read_rows(root, job, part)
        if len(r):
            rows.append(r.assign(_part=part))
    allr = pd.concat(rows, ignore_index=True)
    bank = allr[allr._part.str.startswith("bank_")]
    choose = allr[allr._part.str.startswith("choose_")]
    return bank, choose, costs(calls, allr)


def run(root: Path, job: str, plan: Path) -> dict:
    _, assign, manifest = T.load_questions(plan)
    bank, choose, cost = load(root, job)
    dev = analyze(bank, choose, cost, assign, "dev")
    held = analyze(bank, choose, cost, assign, "heldout")
    headline = max(dev, key=lambda a: dev[a]["P1_minus_pool_plurality"]["mean"]) if dev else None
    return {"label": LABEL, "question_ids_sha256": manifest["question_ids_sha256"], "costs": cost,
            "headline_rule": "largest development P1 gain", "headline_arm": headline,
            "development_P1": {a: v["P1_minus_pool_plurality"] for a, v in dev.items()},
            "heldout": held, "heldout_headline": held.get(headline) if headline else None,
            "self_consistency_heldout": sc_table(bank, cost, assign, "heldout")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = run(Path(a.root), a.job, Path(a.plan))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expa_results.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({"headline_arm": r["headline_arm"], "development_P1": r["development_P1"]}, default=str)[:2000])
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
