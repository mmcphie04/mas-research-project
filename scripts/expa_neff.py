"""Experiment A, EXPLORATORY: the correlated-errors voting model n_eff = n / (1 + (n-1) rho), with
separate within-family and between-family rho, fitted on DEVELOPMENT questions only, then checked
against held-out voting curves for single-family and mixed pools. CPU only.

    python scripts/expa_neff.py --root runs/expab --job expab --out artifacts/expab/analysis

Fit (development): per family (gemma, qwen_nt) the per-sample accuracy p and the within-family
correlation of correctness rho_w (pairs of distinct samples of the same question, pooled); the
between-family rho_b (Gemma x Qwen pairs). A pool's average pairwise correlation rho_bar weights
its within- and between-family pairs; n_eff = n / (1 + (n - 1) rho_bar).
Prediction: the independent-voter model (correct with probability p, wrong answers uniform over the
question's wrong options; ties broken at random) at n_eff, interpolated linearly between the
integers around n_eff. Observed (held-out): exact expected plurality accuracy averaged over sample
subsets (single family: all subsets of size n; mixed k + k: up to 200 seeded subset pairs).
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from expa_analysis import load, vote_score  # noqa: E402

from mad.gen import tasks as T  # noqa: E402

FAMILIES = ("gemma", "qwen_nt")


def correctness(bank: pd.DataFrame, mode: str) -> dict[str, np.ndarray]:
    g = bank[bank["mode"] == mode]
    return {q: (x.sort_values("agent", key=lambda s: s.astype(int)).answer.astype(int).to_numpy() == x.meta_gold.iloc[0]).astype(float)
            for q, x in g.groupby("question_id")}


def pair_corr(a: dict, b: dict | None) -> float:
    xs, ys = [], []
    for q, ca in a.items():
        if b is None:
            for i, j in itertools.combinations(range(len(ca)), 2):
                xs += [ca[i], ca[j]]
                ys += [ca[j], ca[i]]
        elif q in b:
            for x, y in itertools.product(ca, b[q]):
                xs.append(x)
                ys.append(y)
    return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else float("nan")


@cache
def _win(m: int, k: int) -> tuple:
    """P(correct wins | C correct of k, wrong votes uniform over m wrong options), C = 0..k."""
    def comps(w, parts):
        if parts == 1:
            yield (w,)
            return
        for f in range(w + 1):
            for rest in comps(w - f, parts - 1):
                yield (f, *rest)
    h = [0.0] * (k + 1)
    for c in range(1, k + 1):
        w = k - c
        if w == 0:
            h[c] = 1.0
            continue
        tot = 0.0
        for counts in comps(w, m):
            p = math.factorial(w) / math.prod(math.factorial(x) for x in counts) / m ** w
            mx = max(counts)
            t = sum(1 for x in counts if x == mx)
            tot += p * (1.0 if c > mx else (1.0 / (t + 1) if c == mx else 0.0))
        h[c] = tot
    return tuple(h)


def indep(p: float, k: int, m: int) -> float:
    return sum(math.comb(k, c) * p ** c * (1 - p) ** (k - c) * _win(m, k)[c] for c in range(k + 1))


def predict(p: float, n_eff: float, m: int) -> float:
    lo = max(1, int(math.floor(n_eff)))
    hi = max(1, int(math.ceil(n_eff)))
    a, b = indep(p, lo, m), indep(p, hi, m)
    return a if hi == lo else a + (b - a) * (n_eff - lo)


def observed(bank, qids, spec, seed=0, max_pairs=200) -> float:
    idx = T.bank_index(bank)
    rng = np.random.default_rng(seed)
    vals = []
    for q in qids:
        parts = [[int(r["answer"]) for r in idx.get((m, q), [])] for m, _ in spec]
        gold = int(bank[bank.question_id == q].meta_gold.iloc[0])
        choices = [list(itertools.combinations(range(len(p)), k)) for p, (_, k) in zip(parts, spec, strict=True)]
        combos = list(itertools.product(*choices))
        if len(combos) > max_pairs:
            combos = [combos[i] for i in rng.choice(len(combos), max_pairs, replace=False)]
        vals.append(np.mean([vote_score([parts[f][i] for f, sel in enumerate(c) for i in sel], gold) for c in combos]))
    return float(np.mean(vals))


def run(root: Path, job: str, plan: Path) -> dict:
    _, assign, _ = T.load_questions(plan)
    bank, _, _ = load(root, job)
    dev = bank[bank.question_id.map(assign) == "dev"]
    held = bank[bank.question_id.map(assign) == "heldout"]
    c = {f: correctness(dev, f) for f in FAMILIES}
    fit = {"p": {f: float(np.mean([v.mean() for v in c[f].values()])) for f in FAMILIES},
           "rho_within": {f: pair_corr(c[f], None) for f in FAMILIES}, "rho_between": pair_corr(c["gemma"], c["qwen_nt"])}
    m_mean = int(round(held.groupby("question_id").n_options.first().mean())) - 1
    hq = sorted(set(held.question_id))
    out = {"fit_development": fit, "wrong_options_used": m_mean, "curves_heldout": {}}
    for f in FAMILIES:
        for n in range(1, 11):
            ne = n / (1 + (n - 1) * fit["rho_within"][f])
            out["curves_heldout"][f"{f} x{n}"] = {"n_eff": ne, "predicted": predict(fit["p"][f], ne, m_mean),
                                                  "observed": observed(held, hq, [(f, n)])}
    for k in (1, 2, 3, 4, 5):
        n = 2 * k
        within_pairs = 2 * math.comb(k, 2)
        between_pairs = k * k
        rho_bar = ((math.comb(k, 2) * (fit["rho_within"]["gemma"] + fit["rho_within"]["qwen_nt"]) + between_pairs * fit["rho_between"])
                   / (within_pairs + between_pairs))
        ne = n / (1 + (n - 1) * rho_bar)
        p = (fit["p"]["gemma"] + fit["p"]["qwen_nt"]) / 2
        out["curves_heldout"][f"gemma x{k} + qwen_nt x{k}"] = {"n_eff": ne, "rho_bar": rho_bar, "predicted": predict(p, ne, m_mean),
                                                               "observed": observed(held, hq, [("gemma", k), ("qwen_nt", k)])}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = {"label": "EXPLORATORY", **run(Path(a.root), a.job, Path(a.plan))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expa_neff.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["fit_development"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
