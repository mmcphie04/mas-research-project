"""EXPLORATORY analyses of the main collection (not in docs/MAIN250_ANALYSIS_SPEC.md's primary set).

    python scripts/main250_exploratory.py --dest runs/main250 --job main250 --out artifacts/main250/analysis

Every number here is exploratory. All 200 questions unless a subgroup is named. Intervals: 95%
percentile, 10,000 question resamples (seed 0); ratios resample questions, not answers. Scores
are exact expected accuracy under uniformly random tie-breaking, as in the primary analysis.

Sections (fixed by the 2026-10-30 request; the selection rules 6a-c were fixed before computing):
 1 coverage against selection    2 martingale check (Choi et al. 2025)    3 correlated errors
 4 consensus effects of debate    5 headroom by agreement pattern          6 fixed selection rules
 7 power                          8 by subject group
Readout probabilities are the stored letter distributions (normalized over the options; fp32
head, readout batch size 1 -- recorded from the job).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.analysis.actions import expected_vote  # noqa: E402
from mad.analysis.panels import MATCHED, panel_events  # noqa: E402

DEBATES = {"debate3": 3, "structured3": 3, "debate5": 5, "structured5": 5}
GROUPS = {"math/physics/engineering/chemistry": {"math", "physics", "engineering", "chemistry"},
          "law/history/philosophy/psychology": {"law", "history", "philosophy", "psychology"}}
REST = "other subjects"
Z_A, Z_B = 1.959964, 0.841621  # two-sided 5%, 80% power
B_DEFAULT = 10000
LABEL = "EXPLORATORY"


# ----------------------------------------------------------------------------- data


def _probs(s: str) -> np.ndarray:
    return np.exp(np.array([float(v) for v in s.split(",")]))


def load(dest: Path, job: str) -> dict:
    cols = ["question_id", "agent_id", "phase", "answer", "gold_display", "n_options", "readout_logprobs", "subject"]
    pool = pd.read_parquet(dest / f"{job}-pool" / "agent_states.parquet", columns=cols)
    pool = pool[pool.phase == "pre"].assign(question_id=lambda d: d.question_id.astype(str)).sort_values(
        ["question_id", "agent_id"])
    qs = sorted(pool.question_id.unique())
    Q = {}
    for q, g in pool.groupby("question_id"):
        Q[q] = {"gold": int(g.gold_display.iloc[0]), "m": int(g.n_options.iloc[0]), "subject": str(g.subject.iloc[0]),
                "pool": g.answer.astype(int).to_numpy(), "pool_p": np.stack([_probs(x) for x in g.readout_logprobs])}
    D = {}
    for cond, k in DEBATES.items():
        s = pd.read_parquet(dest / f"{job}-{cond}" / "agent_states.parquet", columns=cols)
        s = s.assign(question_id=s.question_id.astype(str)).sort_values(["question_id", "agent_id"])
        D[cond] = {}
        for q, g in s.groupby("question_id"):
            pre, post = g[g.phase == "pre"], g[g.phase == "post"]
            if int(pre.gold_display.iloc[0]) != Q[q]["gold"]:
                raise SystemExit(f"{cond} {q}: answer key differs from the pool's")
            a0 = pre.answer.astype(int).to_numpy()
            if not np.array_equal(a0, Q[q]["pool"][:k]):
                raise SystemExit(f"{cond} {q}: initial answers are not the pool's first {k}")
            D[cond][q] = {"pre": a0, "post": post.answer.astype(int).to_numpy(),
                          "pre_p": np.stack([_probs(x) for x in pre.readout_logprobs]),
                          "post_p": np.stack([_probs(x) for x in post.readout_logprobs])}
    jobrec = json.loads((dest / f"{job}-job.json").read_text())
    ex = next((h.get("execution", {}) for h in jobrec.get("history", []) if h.get("execution")), {})
    readout = {"head": jobrec["identity"]["config"]["model"].get("readout_head"),
               "readout_batch_size": ex.get("readout_batch_size")}
    return {"qs": qs, "Q": Q, "D": D, "readout": readout}


# ----------------------------------------------------------------------------- bootstrap


class Boot:
    def __init__(self, n_boot=B_DEFAULT, seed=0):
        self.n_boot, self.seed, self._idx = n_boot, seed, {}

    def idx(self, n):
        if n not in self._idx:
            self._idx[n] = np.random.default_rng(self.seed).integers(0, n, (self.n_boot, n))
        return self._idx[n]

    def mean(self, x) -> dict:
        x = np.asarray(x, float)
        if len(x) == 0:
            return {"n": 0}
        b = x[self.idx(len(x))].mean(1)
        return {"n": int(len(x)), "mean": float(x.mean()), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}

    def ratio(self, num, den) -> dict:
        """sum(num)/sum(den) over questions; questions resampled."""
        num, den = np.asarray(num, float), np.asarray(den, float)
        if den.sum() == 0:
            return {"n": 0, "den": 0}
        i = self.idx(len(num))
        with np.errstate(invalid="ignore", divide="ignore"):
            b = num[i].sum(1) / den[i].sum(1)
        b = b[np.isfinite(b)]
        return {"n": int((den > 0).sum()), "den": float(den.sum()), "mean": float(num.sum() / den.sum()),
                "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}

    def stat(self, fn, *arrays) -> dict:
        """Any statistic of per-question arrays (first axis = question)."""
        point = fn(*arrays)
        n = len(arrays[0])
        b = np.array([fn(*[a[r] for a in arrays]) for r in self.idx(n)])
        b = b[np.isfinite(b)]
        return {"n": n, "mean": float(point), "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


# ----------------------------------------------------------------------------- helpers


def pattern(answers) -> str:
    top = Counter(int(a) for a in answers).most_common(1)[0][1]
    k = len(answers)
    return "unanimous" if top == k else ("majority" if top > k / 2 else "no majority")


def argmax_score(scores: np.ndarray, gold: int) -> float:
    top = scores.max()
    tied = np.flatnonzero(np.isclose(scores, top, rtol=0, atol=1e-12))
    return (1.0 / len(tied)) if gold in tied else 0.0


def soft_vote(probs: np.ndarray, gold: int) -> float:  # rule 6a
    return argmax_score(probs.sum(0), gold)


def weighted_plurality(answers, probs: np.ndarray, gold: int) -> float:  # rule 6b
    w = np.zeros(probs.shape[1])
    for a, p in zip(answers, probs, strict=True):
        w[int(a)] += p[int(a)]
    return argmax_score(w, gold)


def most_confident(answers, probs: np.ndarray, gold: int) -> float:  # rule 6c
    conf = np.array([p[int(a)] for a, p in zip(answers, probs, strict=True)])
    top = np.flatnonzero(np.isclose(conf, conf.max(), rtol=0, atol=1e-12))
    return float(np.mean([int(answers[i]) == gold for i in top]))


def subject_group(s: str) -> str:
    return next((g for g, members in GROUPS.items() if s in members), REST)


@cache
def wrong_win_table(m: int, k: int) -> np.ndarray:
    """h[C]: probability the correct option wins (ties broken uniformly) when C of k votes are
    correct and the k-C wrong votes fall independently and uniformly on the m wrong options."""
    h = np.zeros(k + 1)
    for c in range(1, k + 1):
        w = k - c
        if w == 0:
            h[c] = 1.0
            continue
        tot = 0.0
        for counts in _compositions(w, m):
            p = math.factorial(w) / math.prod(math.factorial(x) for x in counts) / m ** w
            mx = max(counts)
            t = sum(1 for x in counts if x == mx)
            tot += p * (1.0 if c > mx else (1.0 / (t + 1) if c == mx else 0.0))
        h[c] = tot
    return h


def _compositions(w: int, m: int):
    if m == 1:
        yield (w,)
        return
    for first in range(w + 1):
        for rest in _compositions(w - first, m - 1):
            yield (first, *rest)


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    dist = np.array([1.0])
    for pi in p:
        dist = np.convolve(dist, [1 - pi, pi])
    return dist


# ----------------------------------------------------------------------------- sections


def coverage(d, bt):
    Q, qs, out = d["Q"], d["qs"], {"pool": {}, "debate": {}}
    for k in range(3, 11):
        cov = np.array([float(Q[q]["gold"] in Q[q]["pool"][:k]) for q in qs])
        plu = np.array([expected_vote(Q[q]["pool"][:k], Q[q]["gold"]) for q in qs])
        out["pool"][k] = {"coverage": bt.mean(cov), "plurality": bt.mean(plu), "gap": bt.mean(cov - plu)}
    for c in DEBATES:
        b = np.array([float(Q[q]["gold"] in d["D"][c][q]["pre"]) for q in qs])
        a = np.array([float(Q[q]["gold"] in d["D"][c][q]["post"]) for q in qs])
        pa = np.array([expected_vote(d["D"][c][q]["post"], Q[q]["gold"]) for q in qs])
        out["debate"][c] = {"coverage_before": bt.mean(b), "coverage_after": bt.mean(a), "change": bt.mean(a - b),
                            "plurality_after": bt.mean(pa), "gap_after": bt.mean(a - pa)}
    return out


def martingale(d, bt):
    Q, qs, out = d["Q"], d["qs"], {}
    for c, k in DEBATES.items():
        dc = np.zeros((len(qs), k))
        dp = np.zeros((len(qs), k))
        c0 = np.zeros((len(qs), k))
        for i, q in enumerate(qs):
            r, g = d["D"][c][q], Q[q]["gold"]
            c0[i] = r["pre"] == g
            dc[i] = (r["post"] == g).astype(float) - c0[i]
            dp[i] = r["post_p"][:, g] - r["pre_p"][:, g]

        def block(sel, dc=dc, dp=dp):
            return {"n_answers": int(sel.sum()),
                    "change_in_correctness": bt.ratio((dc * sel).sum(1), sel.sum(1)),
                    "change_in_p_correct": bt.ratio((dp * sel).sum(1), sel.sum(1))}
        ones = np.ones_like(c0)
        out[c] = {"all_agents": {"all": block(ones), "initially_correct": block(c0), "initially_wrong": block(1 - c0)},
                  "per_agent": {a: {"all": block(np.eye(k)[a] * ones), "initially_correct": block(np.eye(k)[a] * c0),
                                    "initially_wrong": block(np.eye(k)[a] * (1 - c0))} for a in range(k)}}
    return out


def icc1(M: np.ndarray) -> float:
    """One-way ICC(1) for an n x k matrix (rows = questions)."""
    n, k = M.shape
    gm = M.mean()
    rm = M.mean(1)
    msb = k * ((rm - gm) ** 2).sum() / (n - 1)
    msw = ((M - rm[:, None]) ** 2).sum() / (n * (k - 1))
    den = msb + (k - 1) * msw
    return float((msb - msw) / den) if den > 0 else float("nan")


def correlated(d, bt):
    Q, qs = d["Q"], d["qs"]
    M = np.array([(Q[q]["pool"] == Q[q]["gold"]).astype(float) for q in qs])
    rho = bt.stat(icc1, M)
    neff = {k: {"n_eff": k / (1 + (k - 1) * rho["mean"]), "lo": k / (1 + (k - 1) * rho["hi"]),
                "hi": k / (1 + (k - 1) * rho["lo"])} for k in range(3, 11)}
    pairs, same, expect = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
    for i, q in enumerate(qs):
        a, g, m = Q[q]["pool"], Q[q]["gold"], Q[q]["m"]
        wrong = [x for x in a if x != g]
        n = len(wrong)
        pairs[i] = n * (n - 1) / 2
        same[i] = sum(v * (v - 1) / 2 for v in Counter(wrong).values())
        expect[i] = pairs[i] / (m - 1)
    obs = bt.ratio(same, pairs)
    exp_ = bt.ratio(expect, pairs)
    ratio = bt.stat(lambda s, e, p: s.sum() / e.sum() if e.sum() else np.nan, same, expect, pairs)
    # independent-voter prediction from per-agent accuracy (wrong answers uniform over wrong options)
    ms = np.array([Q[q]["m"] - 1 for q in qs])
    obs_curve = {k: np.array([expected_vote(Q[q]["pool"][:k], Q[q]["gold"]) for q in qs]) for k in range(3, 11)}

    def predicted(Mx, Tx, k):
        return float((Tx @ poisson_binomial(Mx[:, :k].mean(0))).mean())

    curve = {}
    for k in range(3, 11):
        T = np.stack([wrong_win_table(int(m), k) for m in ms])  # per question: P(win | C correct)
        curve[k] = {"observed": bt.mean(obs_curve[k]),
                    "predicted_independent": bt.stat(lambda Mx, Tx, k=k: predicted(Mx, Tx, k), M, T),
                    "observed_minus_predicted": bt.stat(lambda Mx, Tx, o, k=k: o.mean() - predicted(Mx, Tx, k),
                                                        M, T, obs_curve[k])}
    return {"icc_correctness": rho, "n_eff": neff,
            "same_wrong_answer": {"observed_rate": obs, "uniform_expected_rate": exp_, "observed_over_expected": ratio,
                                  "wrong_pairs": float(pairs.sum())},
            "voting_curve_vs_independent": curve,
            "per_agent_accuracy": M.mean(0).round(4).tolist()}


def consensus(d, bt):
    Q, qs, out = d["Q"], d["qs"], {}
    for c, k in DEBATES.items():
        wu0, wu1 = np.zeros(len(qs)), np.zeros(len(qs))
        rows = []
        for i, q in enumerate(qs):
            r, g = d["D"][c][q], Q[q]["gold"]
            wu0[i] = float(len(set(r["pre"])) == 1 and r["pre"][0] != g)
            wu1[i] = float(len(set(r["post"])) == 1 and r["post"][0] != g)
            maj, top = Counter(r["pre"].tolist()).most_common(1)[0]
            if k / 2 < top < k:
                unan = len(set(r["post"])) == 1
                rows.append((i, maj == g, unan, unan and r["post"][0] == maj))
        res = {"wrong_unanimity_before": bt.mean(wu0), "wrong_unanimity_after": bt.mean(wu1),
               "change": bt.mean(wu1 - wu0), "majority_splits": len(rows)}
        for label, sel in (("all", lambda x: True), ("initial_majority_correct", lambda x: x[1]),
                           ("initial_majority_wrong", lambda x: not x[1])):
            num, den = np.zeros(len(qs)), np.zeros(len(qs))
            spl = np.zeros(len(qs))
            for i, mc, unan, onmaj in rows:
                if sel((i, mc)):
                    spl[i] = 1
                    den[i] = float(unan)
                    num[i] = float(onmaj)
            res[label] = {"splits": int(spl.sum()), "became_unanimous": int(den.sum()),
                          "became_unanimous_rate": bt.ratio(den, spl),
                          "share_ending_on_initial_plurality": bt.ratio(num, den)}
        out[c] = res
    return out


def headroom(d, bt):
    Q, qs = d["Q"], d["qs"]
    ind = {k: np.array([expected_vote(Q[q]["pool"][:k], Q[q]["gold"]) for q in qs]) for k in range(3, 11)}
    post = {c: np.array([expected_vote(d["D"][c][q]["post"], Q[q]["gold"]) for q in qs]) for c in DEBATES}
    pat = {k: np.array([pattern(Q[q]["pool"][:k]) for q in qs]) for k in (3, 5)}
    contrasts = {}
    for c, k in DEBATES.items():
        contrasts[f"{c} vs independent {MATCHED[k]}"] = (k, post[c] - ind[MATCHED[k]])
    for k in (3, 5):
        contrasts[f"structured{k} vs debate{k}"] = (k, post[f"structured{k}"] - post[f"debate{k}"])
    by_pattern = {}
    for name, (k, diff) in contrasts.items():
        by_pattern[name] = {p: Boot(bt.n_boot).mean(diff[pat[k] == p]) for p in ("unanimous", "majority", "no majority")}
    events = {}
    for c, k in DEBATES.items():
        ev = [panel_events(list(d["D"][c][q]["pre"]), list(d["D"][c][q]["post"]), Q[q]["gold"]) for q in qs]
        events[c] = {}
        for p in ("unanimous", "majority", "no majority"):
            sel = [e for e, pp in zip(ev, pat[k], strict=True) if pp == p]
            events[c][p] = {"questions": len(sel), "net_change": Boot(bt.n_boot).mean([e["net"] for e in sel]),
                            **{key: float(sum(e[key] for e in sel)) for key in
                               ("changed", "helpful", "harmful", "minority_correct_recovered",
                                "correct_decision_preserved", "correct_decision_lost", "absent_correct_found")}}
    ceiling = {}
    for k in (3, 5):
        stop, add, deb, st = ind[k], ind[MATCHED[k]], post[f"debate{k}"], post[f"structured{k}"]
        c3 = np.maximum.reduce([stop, deb, add])
        c4 = np.maximum.reduce([stop, deb, st, add])
        ceiling[k] = {"stop": bt.mean(stop), "debate": bt.mean(deb), "structured": bt.mean(st),
                      f"add_independent_to_{MATCHED[k]}": bt.mean(add),
                      "ceiling_stop_debate_add": bt.mean(c3), "ceiling_including_structured": bt.mean(c4),
                      "by_pattern": {p: {"questions": int((pat[k] == p).sum()),
                                         "ceiling_stop_debate_add": Boot(bt.n_boot).mean(c3[pat[k] == p]),
                                         "stop": Boot(bt.n_boot).mean(stop[pat[k] == p])}
                                     for p in ("unanimous", "majority", "no majority")}}
    odd = {k: bt.mean(ind[k]) for k in (3, 5, 7, 9)}
    odd_steps = {f"{a}->{b}": bt.mean(ind[b] - ind[a]) for a, b in ((3, 5), (5, 7), (7, 9), (3, 9))}
    counts = {k: {p: int((pat[k] == p).sum()) for p in ("unanimous", "majority", "no majority")} for k in (3, 5)}
    return {"pattern_counts": counts, "contrasts_by_pattern": by_pattern, "revision_events_by_pattern": events,
            "ceiling": ceiling, "odd_only_curve": odd, "odd_only_steps": odd_steps}


def selection_rules(d, bt):
    Q, qs, out = d["Q"], d["qs"], {"pool": {}, "post_debate": {}}
    for k in (3, 5, 10):
        plu = np.array([expected_vote(Q[q]["pool"][:k], Q[q]["gold"]) for q in qs])
        rules = {"a_soft_vote": [soft_vote(Q[q]["pool_p"][:k], Q[q]["gold"]) for q in qs],
                 "b_confidence_weighted_plurality": [weighted_plurality(Q[q]["pool"][:k], Q[q]["pool_p"][:k], Q[q]["gold"]) for q in qs],
                 "c_most_confident_agent": [most_confident(Q[q]["pool"][:k], Q[q]["pool_p"][:k], Q[q]["gold"]) for q in qs]}
        out["pool"][k] = {"plurality": bt.mean(plu),
                          **{r: {"accuracy": bt.mean(v), "minus_plurality": bt.mean(np.array(v) - plu),
                                 "questions_differing": int((np.array(v) != plu).sum())} for r, v in rules.items()}}
    for c in DEBATES:
        plu = np.array([expected_vote(d["D"][c][q]["post"], Q[q]["gold"]) for q in qs])
        rules = {"a_soft_vote": [soft_vote(d["D"][c][q]["post_p"], Q[q]["gold"]) for q in qs],
                 "b_confidence_weighted_plurality": [weighted_plurality(d["D"][c][q]["post"], d["D"][c][q]["post_p"],
                                                                        Q[q]["gold"]) for q in qs]}
        out["post_debate"][c] = {"plurality": bt.mean(plu),
                                 **{r: {"accuracy": bt.mean(v), "minus_plurality": bt.mean(np.array(v) - plu),
                                        "questions_differing": int((np.array(v) != plu).sum())} for r, v in rules.items()}}
    return out


def power(d, bt):
    Q, qs = d["Q"], d["qs"]
    ind = {k: np.array([expected_vote(Q[q]["pool"][:k], Q[q]["gold"]) for q in qs]) for k in (6, 10)}
    post = {c: np.array([expected_vote(d["D"][c][q]["post"], Q[q]["gold"]) for q in qs]) for c in DEBATES}
    diffs = {f"{c} vs independent {MATCHED[k]}": post[c] - ind[MATCHED[k]] for c, k in DEBATES.items()}
    diffs.update({f"structured{k} vs debate{k}": post[f"structured{k}"] - post[f"debate{k}"] for k in (3, 5)})
    n0, z = len(qs), Z_A + Z_B
    out = {}
    for name, x in diffs.items():
        sd = bt.stat(lambda v: v.std(ddof=1), x)
        mde = {key: z * sd[key] / math.sqrt(n0) for key in ("mean", "lo", "hi")}
        need = {f"{p}_point": {key: math.ceil((z * sd[key] / (p / 100)) ** 2) for key in ("mean", "lo", "hi")}
                for p in (1, 2)}
        out[name] = {"paired_sd": sd, "mde_80pct_power_n200": mde, "questions_needed": need,
                     "note": "normal approximation, two-sided 5%; 'lo'/'hi' carry the SD's interval through"}
    return out


def by_subject(d, bt):
    Q, qs = d["Q"], d["qs"]
    grp = np.array([subject_group(Q[q]["subject"]) for q in qs])
    gain = np.array([expected_vote(Q[q]["pool"][:9], Q[q]["gold"]) - expected_vote(Q[q]["pool"][:3], Q[q]["gold"])
                     for q in qs])
    net = {c: np.array([expected_vote(d["D"][c][q]["post"], Q[q]["gold"]) - expected_vote(d["D"][c][q]["pre"], Q[q]["gold"])
                        for q in qs]) for c in DEBATES}
    out = {}
    for g in [*GROUPS, REST]:
        sel = grp == g
        out[g] = {"questions": int(sel.sum()), "voting_gain_3_to_9": Boot(bt.n_boot).mean(gain[sel]),
                  "debate_net_effect": {c: Boot(bt.n_boot).mean(v[sel]) for c, v in net.items()}}
    return out


def analyze(dest: Path, job: str, n_boot: int = B_DEFAULT) -> dict:
    d = load(dest, job)
    bt = Boot(n_boot)
    P = np.concatenate([d["Q"][q]["pool_p"].max(1) for q in d["qs"]])
    own = np.concatenate([d["Q"][q]["pool_p"][np.arange(10), d["Q"][q]["pool"]] for q in d["qs"]])
    conc = {"pool_states": int(len(P)), "share_max_p_above_0.99": float((P > 0.99).mean()),
            "share_p_own_answer_above_0.99": float((own > 0.99).mean()), "median_p_own_answer": float(np.median(own)),
            "note": "the readout follows the agent's own response, so it largely restates the stated answer"}
    return {"label": LABEL, "note": "exploratory; not in the primary specification; rules 6a-c fixed before computing",
            "questions": len(d["qs"]), "bootstrap": {"resamples": n_boot, "seed": 0}, "readout": d["readout"],
            "readout_concentration": conc,
            "1_coverage_vs_selection": coverage(d, bt), "2_martingale": martingale(d, bt),
            "3_correlated_errors": correlated(d, bt), "4_consensus": consensus(d, bt),
            "5_headroom_by_agreement": headroom(d, bt), "6_fixed_selection_rules": selection_rules(d, bt),
            "7_power": power(d, bt), "8_by_subject_group": by_subject(d, bt)}


# ----------------------------------------------------------------------------- markdown


def ci(x, nd=3, pct=False):
    if not x or x.get("n", 1) == 0 or "mean" not in x:
        return "—"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f"{f(x['mean'])} [{f(x['lo'])}, {f(x['hi'])}]"


def markdown(r: dict) -> str:
    L = [f"# main250 — {LABEL} analyses ({r['questions']} questions)", "",
         f"**Everything in this file is exploratory** (outside `docs/MAIN250_ANALYSIS_SPEC.md`). Brackets: 95% "
         f"percentile intervals, {r['bootstrap']['resamples']:,} question resamples (seed 0); ratios resample "
         "questions. Scores: exact expected accuracy under random tie-breaking. Readout probabilities: the stored "
         f"letter distributions, {r['readout']['head']} head, readout batch size {r['readout']['readout_batch_size']}.", "",
         f"**Measured property of the readouts that limits sections 2 and 6:** in "
         f"{100 * r['readout_concentration']['share_max_p_above_0.99']:.1f}% of the {r['readout_concentration']['pool_states']:,} "
         f"pool states the readout puts more than 0.99 on one option, and in "
         f"{100 * r['readout_concentration']['share_p_own_answer_above_0.99']:.1f}% that option is the agent's own stated "
         "answer. The readout is taken after the response, so it largely restates the answer: Δp(correct) nearly "
         "equals Δ correctness, rules a and b nearly reproduce plurality, and rule c's 'most confident' agent is "
         "mostly a near-tie at ≈ 1 among agents.", ""]
    c = r["1_coverage_vs_selection"]
    L += ["## 1. Coverage against selection (exploratory)", "",
          "| k | coverage (≥ 1 correct among agents 0..k−1) | plurality accuracy | gap |", "|---|---|---|---|"]
    for k, v in c["pool"].items():
        L.append(f"| {k} | {ci(v['coverage'])} | {ci(v['plurality'])} | {ci(v['gap'])} |")
    L += ["", "| condition | coverage before | coverage after | change | plurality after | gap after |",
          "|---|---|---|---|---|---|"]
    for k, v in c["debate"].items():
        L.append(f"| {k} | {ci(v['coverage_before'])} | {ci(v['coverage_after'])} | {ci(v['change'])} | "
                 f"{ci(v['plurality_after'])} | {ci(v['gap_after'])} |")
    m = r["2_martingale"]
    L += ["", "## 2. Martingale check (Choi et al. 2025) (exploratory)", "",
          "Change from the pre to the post state per answer: correctness (0/1) and the readout probability of the "
          "correct option.", "",
          "| condition | agent | answers | Δ correctness | Δ p(correct) | initially correct: Δ correctness / Δ p | initially wrong: Δ correctness / Δ p |",
          "|---|---|---|---|---|---|---|"]
    for cond, v in m.items():
        rows = [("all", v["all_agents"])] + [(str(a), x) for a, x in v["per_agent"].items()]
        for a, x in rows:
            L.append(f"| {cond} | {a} | {x['all']['n_answers']} | {ci(x['all']['change_in_correctness'])} | "
                     f"{ci(x['all']['change_in_p_correct'])} | {ci(x['initially_correct']['change_in_correctness'])} / "
                     f"{ci(x['initially_correct']['change_in_p_correct'])} (n={x['initially_correct']['n_answers']}) | "
                     f"{ci(x['initially_wrong']['change_in_correctness'])} / {ci(x['initially_wrong']['change_in_p_correct'])} "
                     f"(n={x['initially_wrong']['n_answers']}) |")
    e = r["3_correlated_errors"]
    L += ["", "## 3. Correlated errors among the 10 independent answers (exploratory)", "",
          f"- Intra-class correlation of correctness (ICC(1)): {ci(e['icc_correctness'])}.",
          "- Effective number of independent voters, n_eff = k / (1 + (k−1)ρ): "
          + ", ".join(f"k={k}: {v['n_eff']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]" for k, v in e["n_eff"].items()) + ".",
          f"- Same wrong answer when two answers are both wrong ({e['same_wrong_answer']['wrong_pairs']:,.0f} pairs): "
          f"observed {ci(e['same_wrong_answer']['observed_rate'])}; expected if wrong answers were spread evenly "
          f"over the wrong options {ci(e['same_wrong_answer']['uniform_expected_rate'])}; observed / expected "
          f"{ci(e['same_wrong_answer']['observed_over_expected'], 2)}.",
          f"- Per-agent accuracy (agents 0–9): {e['per_agent_accuracy']}.", "",
          "| k | observed plurality accuracy | predicted if independent (per-agent accuracy; wrong answers uniform) | observed − predicted |",
          "|---|---|---|---|"]
    for k, v in e["voting_curve_vs_independent"].items():
        L.append(f"| {k} | {ci(v['observed'])} | {ci(v['predicted_independent'])} | {ci(v['observed_minus_predicted'])} |")
    s = r["4_consensus"]
    L += ["", "## 4. Consensus effects of debate (exploratory)", "",
          "| condition | wrong unanimity before | after | change | majority splits | became unanimous | ending on the initial plurality: all / plurality correct / plurality wrong |",
          "|---|---|---|---|---|---|---|"]
    for cond, v in s.items():
        L.append(f"| {cond} | {ci(v['wrong_unanimity_before'])} | {ci(v['wrong_unanimity_after'])} | {ci(v['change'])} | "
                 f"{v['majority_splits']} | {v['all']['became_unanimous']} | "
                 f"{ci(v['all']['share_ending_on_initial_plurality'])} / "
                 f"{ci(v['initial_majority_correct']['share_ending_on_initial_plurality'])} "
                 f"(n={v['initial_majority_correct']['became_unanimous']}) / "
                 f"{ci(v['initial_majority_wrong']['share_ending_on_initial_plurality'])} "
                 f"(n={v['initial_majority_wrong']['became_unanimous']}) |")
    h = r["5_headroom_by_agreement"]
    L += ["", "## 5. Headroom by initial agreement pattern (exploratory)", "",
          f"Questions per pattern (panel's initial answers): k=3 {h['pattern_counts'][3]}, k=5 {h['pattern_counts'][5]}.", "",
          "| contrast | unanimous | majority | no majority |", "|---|---|---|---|"]
    for name, v in h["contrasts_by_pattern"].items():
        L.append(f"| {name} | {ci(v['unanimous'])} (n={v['unanimous'].get('n', 0)}) | {ci(v['majority'])} "
                 f"(n={v['majority'].get('n', 0)}) | {ci(v['no majority'])} (n={v['no majority'].get('n', 0)}) |")
    L += ["", "| condition | pattern | questions | net change | changed | helpful | harmful | minority recovered | decision preserved | decision lost | absent found |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for cond, v in h["revision_events_by_pattern"].items():
        for p, x in v.items():
            L.append(f"| {cond} | {p} | {x['questions']} | {ci(x['net_change'])} | {x['changed']:.0f} | {x['helpful']:.0f} | "
                     f"{x['harmful']:.0f} | {x['minority_correct_recovered']:.0f} | {x['correct_decision_preserved']:.0f} | "
                     f"{x['correct_decision_lost']:.0f} | {x['absent_correct_found']:.0f} |")
    L += ["", "Per-question ceiling (best of the actions, chosen with hindsight per question):", "",
          "| k | stop | debate | structured | add independent | ceiling (stop, debate, add) | ceiling incl. structured |",
          "|---|---|---|---|---|---|---|"]
    for k, v in h["ceiling"].items():
        add = next(x for key, x in v.items() if key.startswith("add_independent"))
        L.append(f"| {k} | {ci(v['stop'])} | {ci(v['debate'])} | {ci(v['structured'])} | {ci(add)} | "
                 f"{ci(v['ceiling_stop_debate_add'])} | {ci(v['ceiling_including_structured'])} |")
    for k, v in h["ceiling"].items():
        L.append(f"- k={k} by pattern (stop → ceiling): " + "; ".join(
            f"{p} (n={x['questions']}): {ci(x['stop'])} → {ci(x['ceiling_stop_debate_add'])}" for p, x in v["by_pattern"].items()))
    L += ["", "Odd-only independent curve: " + ", ".join(f"{k}: {ci(v)}" for k, v in h["odd_only_curve"].items())
          + ". Steps: " + ", ".join(f"{k}: {ci(v)}" for k, v in h["odd_only_steps"].items()) + "."]
    sr = r["6_fixed_selection_rules"]
    L += ["", "## 6. Fixed (non-learned) selection rules (exploratory; fixed before computing)", "",
          "a) soft vote: argmax of the summed readout probabilities. b) confidence-weighted plurality: each answer "
          "weighted by its agent's readout probability for it. c) the single most confident agent's answer "
          "(highest readout probability for its own answer). Exact ties: expected value.", "",
          "| answers | plurality | a soft vote (− plurality) | b weighted (− plurality) | c most confident (− plurality) |",
          "|---|---|---|---|---|"]
    for k, v in sr["pool"].items():
        L.append(f"| pool, first {k} | {ci(v['plurality'])} | " + " | ".join(
            f"{ci(v[x]['accuracy'])} ({ci(v[x]['minus_plurality'])})"
            for x in ("a_soft_vote", "b_confidence_weighted_plurality", "c_most_confident_agent")) + " |")
    for cond, v in sr["post_debate"].items():
        L.append(f"| {cond}, post-debate | {ci(v['plurality'])} | " + " | ".join(
            f"{ci(v[x]['accuracy'])} ({ci(v[x]['minus_plurality'])})"
            for x in ("a_soft_vote", "b_confidence_weighted_plurality")) + " | — |")
    pw = r["7_power"]
    L += ["", "## 7. Power for the primary contrasts (exploratory)", "",
          "| contrast | paired SD | smallest effect detectable (80% power, n = 200) | questions for 1 point | questions for 2 points |",
          "|---|---|---|---|---|"]
    for name, v in pw.items():
        md, n1, n2 = v["mde_80pct_power_n200"], v["questions_needed"]["1_point"], v["questions_needed"]["2_point"]
        L.append(f"| {name} | {ci(v['paired_sd'])} | {md['mean']:.3f} [{md['lo']:.3f}, {md['hi']:.3f}] | "
                 f"{n1['mean']:,} [{n1['lo']:,}, {n1['hi']:,}] | {n2['mean']:,} [{n2['lo']:,}, {n2['hi']:,}] |")
    L += ["", "Normal approximation, two-sided 5%; intervals carry the SD's bootstrap interval through."]
    sg = r["8_by_subject_group"]
    L += ["", "## 8. By MMLU-Pro subject group (exploratory)", "",
          "| group | questions | voting gain 3 → 9 | " + " | ".join(f"{c} net effect" for c in DEBATES) + " |",
          "|---|---|---|" + "---|" * len(DEBATES)]
    for g, v in sg.items():
        L.append(f"| {g} | {v['questions']} | {ci(v['voting_gain_3_to_9'])} | "
                 + " | ".join(ci(v["debate_net_effect"][c]) for c in DEBATES) + " |")
    return "\n".join(L) + "\n"


def _json_default(o):
    if isinstance(o, np.integer | np.floating):
        return o.item()
    raise TypeError(type(o))


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
    (out / "exploratory.json").write_text(json.dumps(r, indent=1, default=_json_default) + "\n")
    (out / "exploratory.md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
