"""EXPLORATORY: selection with signals that are not the post-answer readout (which is ~one-hot;
main250 exploratory §6).

    python scripts/main250_signals.py --dest runs/main250 --job main250 --out artifacts/main250/analysis

Signals per state (fixed before computing):
  (i)   before_stated_answer: the letter distribution at the token before the agent's stated
        letter in its own response (fp32 projection of the final hidden state);
  (ii)  response_logprob_mean, entering the rules as exp(mean) = geometric-mean token probability;
  (iii) parsed verbalized confidence / 100;
  (iv)  debate conditions only: the agent's peer_exposure readout distribution (after seeing the
        peers, before revising).
A signal's VALUE for an answer: for a distribution, its probability of that answer; otherwise the
scalar. AUROC: that value against the correctness of the answer it accompanies.

Rules over a set of answers (pool's first k = 3, 5, 10; each debate condition's post answers):
  (a) soft vote, distribution signals only: argmax of the summed distributions;
  (b) plurality with each answer weighted by its signal value;
  (c) the single highest-signal answer.
Missing signals: weight 0 in (b), not eligible in (c), skipped in (a); no signal at all -> plurality.
Exact ties: expected value under random tie-breaking. All 200 questions; 95% percentile
intervals from 10,000 question resamples (seed 0).
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
from main250_exploratory import (  # noqa: E402
    B_DEFAULT,
    DEBATES,
    LABEL,
    Boot,
    _json_default,
    argmax_score,
    ci,
)

from mad.analysis.actions import expected_vote  # noqa: E402

SIGNALS = {"i_before_stated_answer": "dist", "ii_response_logprob_mean": "scalar",
           "iii_verbalized_confidence": "scalar", "iv_peer_exposure": "dist"}
COLS = ["question_id", "agent_id", "phase", "answer", "gold_display", "n_options", "pos_before_stated_answer_logprobs",
        "response_logprob_mean", "verbalized_confidence", "readout_logprobs"]


def _dist(s) -> np.ndarray | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
        return None
    return np.exp(np.array([float(v) for v in str(s).split(",")]))


def _signals(g: pd.DataFrame, exposure: pd.DataFrame | None) -> dict:
    out = {"i_before_stated_answer": [_dist(x) for x in g.pos_before_stated_answer_logprobs],
           "ii_response_logprob_mean": [None if pd.isna(x) else float(np.exp(x)) for x in g.response_logprob_mean],
           "iii_verbalized_confidence": [None if pd.isna(x) else float(x) / 100 for x in g.verbalized_confidence]}
    if exposure is not None:
        out["iv_peer_exposure"] = [_dist(x) for x in exposure.readout_logprobs]
    return out


def load(dest: Path, job: str) -> dict:
    def read(cond):
        import pyarrow.parquet as pq

        path = dest / f"{job}-{cond}" / "agent_states.parquet"
        have = set(pq.read_schema(path).names)
        s = pd.read_parquet(path, columns=[c for c in COLS if c in have])
        for c in COLS:  # a backend that records no such signal (e.g. the CPU stub): missing everywhere
            if c not in have:
                s[c] = None
        return s.assign(question_id=s.question_id.astype(str)).sort_values(["question_id", "agent_id"])

    pool = read("pool")
    pool = pool[pool.phase == "pre"]
    sets = {"pool": {}}
    gold, n_opt = {}, {}
    for q, g in pool.groupby("question_id"):
        gold[q], n_opt[q] = int(g.gold_display.iloc[0]), int(g.n_options.iloc[0])
        sets["pool"][q] = {"answers": g.answer.astype(int).to_numpy(), "signals": _signals(g, None)}
    for cond in DEBATES:
        s = read(cond)
        sets[cond] = {}
        for q, g in s.groupby("question_id"):
            post, exp_ = g[g.phase == "post"], g[g.phase == "peer_exposure"]
            if list(post.agent_id) != list(exp_.agent_id):
                raise SystemExit(f"{cond} {q}: exposure and post states do not align by agent")
            sets[cond][q] = {"answers": post.answer.astype(int).to_numpy(), "signals": _signals(post, exp_)}
    return {"qs": sorted(gold), "gold": gold, "n_opt": n_opt, "sets": sets}


def value(sig, kind, answer):
    if sig is None:
        return None
    return float(sig[int(answer)]) if kind == "dist" else float(sig)


def rule_a(answers, dists, gold):
    have = [d for d in dists if d is not None]
    return argmax_score(np.sum(have, axis=0), gold) if have else expected_vote(answers, gold)


def rule_b(answers, vals, gold, n):
    w = np.zeros(n)
    for a, v in zip(answers, vals, strict=True):
        if v is not None:
            w[int(a)] += v
    return argmax_score(w, gold) if w.sum() > 0 else expected_vote(answers, gold)


def rule_c(answers, vals, gold):
    ok = [(v, int(a)) for a, v in zip(answers, vals, strict=True) if v is not None]
    if not ok:
        return expected_vote(answers, gold)
    top = max(v for v, _ in ok)
    return float(np.mean([a == gold for v, a in ok if abs(v - top) <= 1e-12]))


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float("nan")
    r = pd.Series(scores).rank().to_numpy()  # average ranks for ties
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def describe(d, bt) -> dict:
    out = {}
    for name, sets in d["sets"].items():
        out[name] = {}
        for sig, kind in SIGNALS.items():
            per_q_s, per_q_y, top, avail = [], [], [], []
            for q in d["qs"]:
                r = sets[q]
                if sig not in r["signals"]:
                    break
                ss, yy = [], []
                for a, x in zip(r["answers"], r["signals"][sig], strict=True):
                    avail.append(x is not None)
                    if x is None:
                        continue
                    top.append(float(np.max(x)) if kind == "dist" else float(x))
                    ss.append(value(x, kind, a))
                    yy.append(int(int(a) == d["gold"][q]))
                per_q_s.append(np.array(ss))
                per_q_y.append(np.array(yy))
            if not top:
                continue
            S, Y = np.concatenate(per_q_s), np.concatenate(per_q_y)
            idx = bt.idx(len(per_q_s))
            boots = np.array([auroc(np.concatenate([per_q_s[i] for i in row]), np.concatenate([per_q_y[i] for i in row]))
                              for row in idx])
            boots = boots[np.isfinite(boots)]
            out[name][sig] = {"states": len(avail), "available_rate": float(np.mean(avail)),
                              "median_top1" if kind == "dist" else "median_value": float(np.median(top)),
                              "share_above_0.99": float((np.array(top) > 0.99).mean()),
                              "auroc_correctness": {"mean": auroc(S, Y), "lo": float(np.quantile(boots, .025)),
                                                    "hi": float(np.quantile(boots, .975)), "n": int(len(S))}}
    return out


def rules(d, bt) -> dict:
    gold, qs, out = d["gold"], d["qs"], {}
    panels = {f"pool first {k}": ("pool", k) for k in (3, 5, 10)}
    panels.update({f"{c} post-debate": (c, None) for c in DEBATES})
    for label, (name, k) in panels.items():
        sets = d["sets"][name]
        ans = {q: sets[q]["answers"][:k] if k else sets[q]["answers"] for q in qs}
        sig = {q: {s: v[:k] if k else v for s, v in sets[q]["signals"].items()} for q in qs}
        n_opt = d["n_opt"]
        plu = np.array([expected_vote(ans[q], gold[q]) for q in qs])
        cov = np.array([float(gold[q] in ans[q]) for q in qs])
        nonu = np.array([len(set(ans[q].tolist())) > 1 for q in qs])
        gap = cov - plu
        res = {"plurality": bt.mean(plu), "coverage": bt.mean(cov), "gap": bt.mean(gap),
               "non_unanimous_questions": int(nonu.sum()), "rules": {}}
        for s, kind in SIGNALS.items():
            if s not in sig[qs[0]]:
                continue
            vals = {q: [value(x, kind, a) for a, x in zip(ans[q], sig[q][s], strict=True)] for q in qs}
            variants = {"b_weighted_plurality": np.array([rule_b(ans[q], vals[q], gold[q], n_opt[q]) for q in qs]),
                        "c_highest_signal": np.array([rule_c(ans[q], vals[q], gold[q]) for q in qs])}
            if kind == "dist":
                variants = {"a_soft_vote": np.array([rule_a(ans[q], sig[q][s], gold[q]) for q in qs]), **variants}
            res["rules"][s] = {}
            for rname, sc in variants.items():
                diff = sc - plu
                sub = Boot(bt.n_boot)
                res["rules"][s][rname] = {
                    "accuracy": bt.mean(sc), "minus_plurality": bt.mean(diff),
                    "questions_differing": int((diff != 0).sum()),
                    "gap_closed_share": bt.ratio(diff, gap),
                    "non_unanimous": {"minus_plurality": sub.mean(diff[nonu]), "gap": sub.mean(gap[nonu]),
                                      "gap_closed_share": sub.ratio(diff[nonu], gap[nonu])}}
        out[label] = res
    return out


def analyze(dest: Path, job: str, n_boot: int = B_DEFAULT) -> dict:
    d = load(dest, job)
    bt = Boot(n_boot)
    return {"label": LABEL, "note": "exploratory; signals, value definitions and rules fixed before computing",
            "questions": len(d["qs"]), "bootstrap": {"resamples": n_boot, "seed": 0},
            "describe": describe(d, bt), "selection": rules(d, bt)}


NAMES = {"i_before_stated_answer": "(i) before-stated-answer distribution",
         "ii_response_logprob_mean": "(ii) response mean log-prob (as exp)",
         "iii_verbalized_confidence": "(iii) verbalized confidence", "iv_peer_exposure": "(iv) peer-exposure readout"}


def markdown(r: dict) -> str:
    L = [f"# main250 — selection signals ({LABEL})", "",
         f"**Exploratory.** {r['questions']} questions; brackets are 95% percentile intervals from "
         f"{r['bootstrap']['resamples']:,} question resamples (seed 0). Signal definitions, values and rules were "
         "fixed before computing (see `scripts/main250_signals.py`). Scores: exact expected accuracy under random "
         "tie-breaking.", "", "## Signals per state", "",
         "| states | signal | available | median top-1 / value | share > 0.99 | AUROC for correctness |",
         "|---|---|---|---|---|---|"]
    for name, sigs in r["describe"].items():
        for s, v in sigs.items():
            med = v.get("median_top1", v.get("median_value"))
            L.append(f"| {name} {'(pre)' if name == 'pool' else '(post)'} | {NAMES[s]} | {100 * v['available_rate']:.1f}% "
                     f"| {med:.4f} | {100 * v['share_above_0.99']:.1f}% | {ci(v['auroc_correctness'])} |")
    L += ["", "Verbalized-confidence parse rate = its 'available' column. AUROC scores each signal's value for the "
          "answer it accompanies (for (iv): the exposure probability of the agent's post answer) against that "
          "answer's correctness.", "", "## Rules against plurality", ""]
    for label, x in r["selection"].items():
        L += [f"### {label}", "",
              f"Plurality {ci(x['plurality'])}; coverage {ci(x['coverage'])}; gap {ci(x['gap'])}; "
              f"non-unanimous questions: {x['non_unanimous_questions']}.", "",
              "| signal | rule | accuracy | − plurality | questions differing | share of gap closed | non-unanimous: − plurality | non-unanimous: share of gap closed |",
              "|---|---|---|---|---|---|---|---|"]
        for s, rs in x["rules"].items():
            for rn, v in rs.items():
                L.append(f"| {NAMES[s]} | {rn} | {ci(v['accuracy'])} | {ci(v['minus_plurality'])} | {v['questions_differing']} | "
                         f"{ci(v['gap_closed_share'], 2)} | {ci(v['non_unanimous']['minus_plurality'])} | "
                         f"{ci(v['non_unanimous']['gap_closed_share'], 2)} |")
        L.append("")
    L += ["Rules (b) and (c) choose among the answers given: on unanimous answer sets the gap is zero and they "
          "equal plurality, so their share of the gap closed is the same overall and on non-unanimous questions by "
          "construction (only the per-question difference grows). A soft vote (a) can choose an option no agent "
          "gave, so it can differ even on unanimous sets (e.g. (iv) on structured3)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "runs/main250"))
    ap.add_argument("--job", default="main250")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=B_DEFAULT)
    a = ap.parse_args(argv)
    r = analyze(Path(a.dest), a.job, a.n_boot)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection_signals.json").write_text(json.dumps(r, indent=1, default=_json_default) + "\n")
    (out / "selection_signals.md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
