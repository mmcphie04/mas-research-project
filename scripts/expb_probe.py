"""Experiment B probes -- EXPLORATORY (docs/EXPAB_PLAN.md, section 4). CPU only.

    python scripts/expb_probe.py --root runs/expab --job expab --out artifacts/expab/analysis

1. Positive control first (``posctl_<mode>``: ~50 contexts whose prompt names the letter to
   answer): the commitment readout's argmax must recover the instructed letter, and a probe on the
   activations at position A (multinomial logistic, 5-fold CV) must recover it at some layer.
   Passed iff readout recovery >= 0.9 and the best layer's probe accuracy >= 0.8.
2. Probes on Experiment B rows (activations at A and B, layers 14/22/31): targets "will change"
   (all agents in peer arms, from A) and "helpful vs harmful" (changed agents, from A and from B).
   L2 logistic. The layer x position is chosen on DEVELOPMENT questions by grouped-CV
   within-question AUROC; the probe trained on all development rows is evaluated once on held-out.
   Primary measure: within-question AUROC (every positive-negative pair from the same question;
   ties 1/2), with a question bootstrap; ordinary AUROC beside it.
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

from mad.gen import archive as A  # noqa: E402
from mad.gen import readouts as RO  # noqa: E402
from mad.gen import tasks as T  # noqa: E402
from mad.provenance import is_complete  # noqa: E402

N_BOOT = 10000


def within_q_auroc(scores, labels, groups) -> tuple[float, float]:
    """(numerator, pairs): pairs (positive, negative) within the same group; ties 1/2."""
    num = den = 0.0
    df = pd.DataFrame({"s": scores, "y": labels, "g": groups})
    for _, h in df.groupby("g"):
        p, n = h.s[h.y == 1].to_numpy(), h.s[h.y == 0].to_numpy()
        if len(p) and len(n):
            d = p[:, None] - n[None, :]
            num += float((d > 0).sum() + 0.5 * (d == 0).sum())
            den += float(d.size)
    return num, den


def wq_boot(scores, labels, groups) -> dict:
    df = pd.DataFrame({"s": scores, "y": labels, "g": groups})
    per = []
    for _, h in df.groupby("g"):
        n, d = within_q_auroc(h.s, h.y, h.g)
        if d:
            per.append((n, d))
    if not per:
        return {"pairs": 0}
    num, den = np.array([x for x, _ in per]), np.array([y for _, y in per])
    idx = np.random.default_rng(0).integers(0, len(per), (N_BOOT, len(per)))
    b = num[idx].sum(1) / den[idx].sum(1)
    return {"questions": len(per), "pairs": int(den.sum()), "mean": float(num.sum() / den.sum()),
            "lo": float(np.quantile(b, .025)), "hi": float(np.quantile(b, .975))}


def load_acts(root: Path, job: str, part: str) -> tuple[pd.DataFrame, np.ndarray]:
    frames, arrs = [], []
    for d in sorted(root.glob(f"{job}-{part}-c*")):
        if d.is_dir() and is_complete(d) and (d / "acts.npy").exists():
            frames.append(pd.read_parquet(d / "rows.parquet"))
            arrs.append(np.load(d / "acts.npy").astype(np.float32))
    return (pd.concat(frames, ignore_index=True), np.concatenate(arrs)) if frames else (pd.DataFrame(), np.zeros((0,)))


def positive_control(root, job, part_suffix: str = "") -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    out = {}
    for mode in ("qwen_t", "qwen_nt"):
        rows, acts = load_acts(root, job, f"posctl_{mode}{part_suffix}")
        if not len(rows):
            continue
        y = rows.instructed.to_numpy()
        readout = float(np.mean([int(np.argmax(lp)) == t for lp, t in zip(rows.commit_logprobs, y, strict=True)]))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        by_pos = {}
        for pi, pname in [(0, "A"), (2, "R")][: acts.shape[1] - 1 if acts.shape[1] == 3 else 1]:
            probe = {}
            for li, layer in enumerate(RO.LAYERS):
                X = acts[:, pi, li, :]
                try:
                    pred = cross_val_predict(LogisticRegression(max_iter=2000), X, y, cv=folds)
                    probe[layer] = float(np.mean(pred == y))
                except ValueError:  # too few examples per letter for CV
                    probe[layer] = None
            by_pos[pname] = probe
        best = {p: max((v for v in d.values() if v is not None), default=None) for p, d in by_pos.items()}
        out[mode] = {"contexts": len(rows), "readout_recovery": readout, "probe_accuracy_by_position_layer": by_pos,
                     "passed_by_position": {p: bool(readout >= 0.9 and b is not None and b >= 0.8) for p, b in best.items()}}
    return out


def probes(root, job, assign) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    rows, acts = [], []
    calls = A.read_calls(root, job)
    for part in sorted(p for p in set(calls.part) if p.startswith("b_") and p.endswith("_ro")):
        r, x = load_acts(root, job, part)
        if not len(r):
            continue
        gen = A.read_rows(root, job, part[:-3])[["request_id", "answer", "parsed", "meta_before", "meta_gold", "meta_arm", "question_id", "mode"]]
        r = r.merge(gen, on="request_id", how="left")
        rows.append(r)
        acts.append(x)
    if not rows:
        return {}
    df, X = pd.concat(rows, ignore_index=True), np.concatenate(acts)
    df["split"] = df.question_id.map(assign)
    df["changed"] = (df.parsed & (df.answer != df.meta_before)).astype(int)
    df["helpful"] = (df.answer == df.meta_gold).astype(int)
    out = {}
    targets = {"will_change_from_A": (df.meta_arm != "alone") & df.parsed, "helpful_vs_harmful": df.changed.astype(bool) &
               ((df.answer == df.meta_gold) | (df.meta_before == df.meta_gold))}
    for mode in sorted(set(df["mode"])):
        for tname, sel in targets.items():
            d = df[sel & (df["mode"] == mode)]
            y = (d.changed if tname == "will_change_from_A" else d.helpful).to_numpy()
            positions = ([0] if tname == "will_change_from_A" else [0, 1]) + ([2] if X.shape[1] == 3 else [])  # + R
            dev, held = d.split == "dev", d.split == "heldout"
            best, scores = None, {}
            for pos in positions:
                for li, layer in enumerate(RO.LAYERS):
                    Xs = X[d.index][:, pos, li, :]
                    ok = ~np.isnan(Xs).any(1)
                    m = dev.to_numpy() & ok
                    if m.sum() < 20 or len(set(y[m])) < 2:
                        continue
                    pred = np.zeros(m.sum())
                    g = d.question_id.to_numpy()[m]
                    for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(Xs[m], y[m], g):
                        if len(set(y[m][tr])) < 2:
                            continue
                        pred[te] = LogisticRegression(max_iter=2000, C=0.1).fit(Xs[m][tr], y[m][tr]).predict_proba(Xs[m][te])[:, 1]
                    n, den = within_q_auroc(pred, y[m], g)
                    scores[f"pos{'ABR'[pos]}_L{layer}"] = n / den if den else None
                    if den and (best is None or n / den > best[0]):
                        best = (n / den, pos, li, layer)
            res = {"dev_within_question_auroc": scores, "chosen": None, "heldout": None}
            if best:
                _, pos, li, layer = best
                Xs = X[d.index][:, pos, li, :]
                ok = ~np.isnan(Xs).any(1)
                m_dev, m_held = dev.to_numpy() & ok, held.to_numpy() & ok
                res["chosen"] = f"pos{'ABR'[pos]}_L{layer}"
                if m_held.sum() and len(set(y[m_dev])) == 2:
                    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xs[m_dev], y[m_dev])
                    s = clf.predict_proba(Xs[m_held])[:, 1]
                    res["heldout"] = {"within_question_auroc": wq_boot(s, y[m_held], d.question_id.to_numpy()[m_held]),
                                      "rows": int(m_held.sum())}
            out[f"{mode} {tname}"] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _, assign, _ = T.load_questions(a.plan)
    r = {"label": "EXPLORATORY", "positive_control": positive_control(Path(a.root), a.job),
         "probes": probes(Path(a.root), a.job, assign)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expb_probes.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["positive_control"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
