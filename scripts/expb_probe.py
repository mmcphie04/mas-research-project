"""Experiment B probes -- EXPLORATORY (docs/EXPAB_PLAN.md, section 4). CPU only.

    python scripts/expb_probe.py --root runs/expab --job expab --out artifacts/expab/analysis

1. Positive control first (``posctl_<mode>``: ~50 contexts whose prompt names the letter to
   answer): the commitment readout's argmax must recover the instructed letter, and a probe on the
   activations at position A (multinomial logistic, 5-fold CV) must recover it at some layer.
   Passed iff readout recovery >= 0.9 and the best layer's probe accuracy >= 0.8.
2. Probes on Experiment B rows (layers 14/22/31): targets "will change" (all agents in peer arms)
   and "helpful vs harmful" (changed agents). Amendment of 2026-11-21: position A failed its
   positive control and is excluded; the probes use R (the readout position, which passed) and B.
   R exists only in launch-5 (held-out) rows, so where the layer must be chosen on development
   it is reported as not computable as committed.
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


POS = {"A": 0, "B": 1, "R": 2}
EXCLUDED = {"A": "failed its positive control (200 development contexts; amendment of 2026-11-21)"}
TARGET_POSITIONS = {"will_change": ["R"], "helpful_vs_harmful": ["B", "R"]}  # A excluded (EXCLUDED)


class Acts:
    """Activations of every B readout chunk, memory-mapped; one (position, layer) slice at a time.
    Pilot chunks hold [A, B]; launch-5 chunks hold [A, B, R]. A missing position reads as NaN."""

    def __init__(self, files: list[tuple[Path, int]]):
        self.files = files

    def slice(self, pos: int, li: int) -> np.ndarray:
        out = []
        for f, n in self.files:
            a = np.load(f, mmap_mode="r")
            out.append(np.asarray(a[:, pos, li, :], np.float32) if a.shape[1] > pos else np.full((n, a.shape[-1]), np.nan, np.float32))
        return np.concatenate(out)


def load_b(root: Path, job: str) -> tuple[pd.DataFrame, Acts]:
    rows, files = [], []
    calls = A.read_calls(root, job)
    for part in sorted(p for p in set(calls.part) if p.startswith("b_") and p.endswith("_ro")):
        gen = A.read_rows(root, job, part[:-3])[["request_id", "answer", "parsed", "meta_before", "meta_gold", "meta_arm", "question_id", "mode"]]
        for d in sorted(root.glob(f"{job}-{part}-c*")):
            if d.is_dir() and is_complete(d) and (d / "acts.npy").exists():
                r = pd.read_parquet(d / "rows.parquet")[["request_id"]].merge(gen, on="request_id", how="left")
                rows.append(r.assign(_part=part))
                files.append((d / "acts.npy", len(r)))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), Acts(files)


def probes(root, job, assign) -> dict:
    """Layer x position chosen on DEVELOPMENT by grouped-CV within-question AUROC; the probe trained on
    all development rows is evaluated once on held-out. Positions: B, and R where development rows
    have it; A is excluded (failed its positive control)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    df, acts = load_b(Path(root), job)
    if not len(df):
        return {}
    df["split"] = df.question_id.map(assign)
    df["changed"] = (df.parsed & (df.answer != df.meta_before)).astype(int)
    df["helpful"] = (df.answer == df.meta_gold).astype(int)
    sel = {"will_change": (df.meta_arm != "alone") & df.parsed,
           "helpful_vs_harmful": df.changed.astype(bool) & ((df.answer == df.meta_gold) | (df.meta_before == df.meta_gold))}
    cache: dict = {}

    def X(pos, li):
        if (pos, li) not in cache:
            cache.clear()  # one slice in memory at a time
            cache[(pos, li)] = acts.slice(pos, li)
        return cache[(pos, li)]

    out = {"excluded_positions": EXCLUDED}
    for mode in sorted(set(df["mode"])):
        for tname, mask in sel.items():
            d = df[mask & (df["mode"] == mode)]
            y = (d.changed if tname == "will_change" else d.helpful).to_numpy()
            dev, held = (d.split == "dev").to_numpy(), (d.split == "heldout").to_numpy()
            g_all = d.question_id.to_numpy()
            best, scores, notes = None, {}, {}
            for pname in TARGET_POSITIONS[tname]:
                pos = POS[pname]
                for li, layer in enumerate(RO.LAYERS):
                    Xs = X(pos, li)[d.index]
                    ok = ~np.isnan(Xs).any(1)
                    m = dev & ok
                    if not m.any():
                        notes[pname] = ("not computable as committed: no development row has this position "
                                        "(the development pilot's readouts predate R)")
                        break
                    if m.sum() < 20 or len(set(y[m])) < 2:
                        notes[pname] = f"too few development rows ({int(m.sum())}) or one class"
                        continue
                    pred, g = np.zeros(m.sum()), g_all[m]
                    for tr, te in GroupKFold(n_splits=min(5, len(set(g)))).split(Xs[m], y[m], g):
                        if len(set(y[m][tr])) < 2:
                            continue
                        pred[te] = LogisticRegression(max_iter=2000, C=0.1).fit(Xs[m][tr], y[m][tr]).predict_proba(Xs[m][te])[:, 1]
                    n, den = within_q_auroc(pred, y[m], g)
                    scores[f"pos{pname}_L{layer}"] = n / den if den else None
                    if den and (best is None or n / den > best[0]):
                        best = (n / den, pos, pname, li, layer)
            res = {"positions_considered": TARGET_POSITIONS[tname], "position_notes": notes,
                   "dev_rows": int(dev.sum()), "dev_within_question_auroc": scores, "chosen": None, "heldout": None}
            if best:
                _, pos, pname, li, layer = best
                Xs = X(pos, li)[d.index]
                ok = ~np.isnan(Xs).any(1)
                m_dev, m_held = dev & ok, held & ok
                res["chosen"] = f"pos{pname}_L{layer}"
                if m_held.sum() and len(set(y[m_dev])) == 2:
                    clf = LogisticRegression(max_iter=2000, C=0.1).fit(Xs[m_dev], y[m_dev])
                    s_ = clf.predict_proba(Xs[m_held])[:, 1]
                    res["heldout"] = {"within_question_auroc": wq_boot(s_, y[m_held], g_all[m_held]),
                                      "ordinary_auroc": float(roc_auc_score(y[m_held], s_)) if len(set(y[m_held])) == 2 else None,
                                      "rows": int(m_held.sum()), "positives": int(y[m_held].sum())}
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
    r = {"label": "EXPLORATORY",
         "positive_control": {"decision_rerun200": positive_control(Path(a.root), a.job, "__rerun200"),
                              "original_50": positive_control(Path(a.root), a.job)},
         "probes": probes(Path(a.root), a.job, assign)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expb_probes.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["positive_control"], default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
