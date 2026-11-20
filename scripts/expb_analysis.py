"""Experiment B: pilot report and measures (docs/EXPAB_PLAN.md, section 4). CPU only.

    python scripts/expb_analysis.py --root runs/expab --job expab --out artifacts/expab/analysis [--pilot]

Per agent: before = its bank answer (all three panel answers parsed, panel not unanimous), after =
its revised stated answer (the last ``Answer: (X)``). Kinds: unchanged, helpful (wrong -> right),
harmful (right -> wrong), wrong_wrong; an unparsed revision is counted separately ("unparsed"),
never as a change.

Pilot (arms debate and alone): change rate, helpful/harmful counts and ratio per mode x arm, and
the sizing of the full run: panels needed so each mode x arm has >= 150 helpful + harmful events
(= ceil(150 / (helpful+harmful per panel))), with the measured GPU-seconds per panel.
Measures (full run): the same per mode x arm, paired arm contrasts on the same panels (panel
bootstrap), the order effect (changed agents whose two shown peers differ and whose new answer is
one of them: last-shown vs first-shown adopted), the commitment readout (probability on the
agent's own previous answer right after reading the peers), and the thinking answer-prediction
curves (probability of the final answer at 0..100% of the trace, by kind).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.gen import archive as A  # noqa: E402

TARGET_HH = 150
N_BOOT = 10000


def boot_groups(values, groups, n_boot=N_BOOT) -> dict:
    v, g = np.asarray(values, float), np.asarray(groups)
    if len(v) == 0:
        return {"n": 0}
    u, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, v, len(u)), np.bincount(inv, None, len(u))
    idx = np.random.default_rng(0).integers(0, len(u), (n_boot, len(u)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return {"n": int(len(v)), "groups": int(len(u)), "mean": float(v.mean()), "lo": float(np.quantile(b, .025)),
            "hi": float(np.quantile(b, .975))}


def kinds(df: pd.DataFrame) -> pd.Series:
    b, a, g = df.meta_before.astype(int), df.answer.astype(int), df.meta_gold.astype(int)
    k = np.where(~df.parsed, "unparsed", np.where(a == b, "unchanged", np.where(a == g, "helpful",
                 np.where(b == g, "harmful", "wrong_wrong"))))
    return pd.Series(k, index=df.index)


def load(root: Path, job: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    calls = A.read_calls(root, job)
    parts = sorted(p for p in set(calls.part) if p.startswith("b_") and not p.endswith("_ro")) if len(calls) else []
    frames = []
    for p in parts:
        r = A.read_rows(root, job, p)
        mode = "qwen_t" if p.startswith("b_qwen_t_") else "qwen_nt"  # part = b_<mode>_<arm>
        arm = p[len(f"b_{mode}_"):]
        ro = A.read_rows(root, job, f"{p}_ro")
        if len(ro):
            r = r.merge(ro, on="request_id", how="left")
        frames.append(r.assign(b_mode=mode, b_arm=arm, _part=p))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(df):
        df["kind"] = kinds(df)
        df["panel"] = df.question_id + "|" + df.b_mode
    return df, calls


def summary(df: pd.DataFrame, calls: pd.DataFrame, units_per_hour: float) -> dict:
    out = {}
    for (mode, arm), g in df.groupby(["b_mode", "b_arm"]):
        k = g.kind.value_counts()
        panels = g.panel.nunique()
        hh = int(k.get("helpful", 0) + k.get("harmful", 0))
        secs = float(calls[calls.part == g._part.iloc[0]].seconds.sum()) if len(calls) else 0.0
        valid = g[g.kind != "unparsed"]
        per_panel_hh = hh / panels if panels else 0.0
        need = math.ceil(TARGET_HH / per_panel_hh) if per_panel_hh else None
        out[f"{mode} {arm}"] = {
            "panels": panels, "agents": len(g), "unparsed": int(k.get("unparsed", 0)),
            "change_rate": boot_groups((valid.kind != "unchanged").astype(float), valid.question_id),
            "helpful": int(k.get("helpful", 0)), "harmful": int(k.get("harmful", 0)), "wrong_wrong": int(k.get("wrong_wrong", 0)),
            "helpful_harmful_ratio": (int(k.get("helpful", 0)) / int(k.get("harmful", 0))) if k.get("harmful", 0) else None,
            "helpful_harmful_per_panel": per_panel_hh, "panels_needed_for_150": need,
            "gpu_seconds_per_panel": secs / panels if panels else None,
            "units_for_150": (need * secs / panels / 3600 * units_per_hour) if need and panels else None,
            "thinking_capped_rate": float(g.thinking_capped.mean()) if "thinking_capped" in g else None}
    return out


def contrasts(df: pd.DataFrame) -> dict:
    out = {}
    for mode, g in df.groupby("b_mode"):
        piv = {}
        for arm, h in g.groupby("b_arm"):
            h = h[h.kind != "unparsed"]
            piv[arm] = h.assign(changed=(h.kind != "unchanged").astype(float), helpful=(h.kind == "helpful").astype(float),
                                harmful=(h.kind == "harmful").astype(float)).groupby("panel")[["changed", "helpful", "harmful"]].mean()
        for a, b in (("debate", "alone"), ("dissent", "debate"), ("anon", "debate")):
            if a in piv and b in piv:
                common = piv[a].index.intersection(piv[b].index)
                d = piv[a].loc[common] - piv[b].loc[common]
                out[f"{mode}: {a} - {b}"] = {m: boot_groups(d[m].to_numpy(), common.to_numpy()) for m in ("changed", "helpful", "harmful")}
    return out


def order_effect(df: pd.DataFrame) -> dict:
    out = {}
    for (mode, arm), g in df[df.b_arm != "alone"].groupby(["b_mode", "b_arm"]):
        rows = []
        for r in g[g.kind.isin(["helpful", "harmful", "wrong_wrong"])].to_dict("records"):
            shown = json.loads(r["meta_shown_answers"])
            if len(shown) == 2 and shown[0] != shown[1] and int(r["answer"]) in shown:
                rows.append((r["question_id"], float(int(r["answer"]) == shown[-1])))
        out[f"{mode} {arm}"] = boot_groups([x for _, x in rows], [q for q, _ in rows]) if rows else {"n": 0}
    return out


def commitment(df: pd.DataFrame) -> dict:
    if "commit_logprobs" not in df:
        return {}
    out = {}
    d = df[df.commit_logprobs.notna() & (df.kind != "unparsed")].copy()
    d["p_before"] = [math.exp(lp[b]) for lp, b in zip(d.commit_logprobs, d.meta_before.astype(int), strict=True)]
    for (mode, arm), g in d.groupby(["b_mode", "b_arm"]):
        out[f"{mode} {arm}"] = {k: boot_groups(h.p_before, h.question_id) for k, h in
                                (("unchanged", g[g.kind == "unchanged"]), ("changed", g[g.kind != "unchanged"]))}
    return out


def curves(df: pd.DataFrame) -> dict:
    if "curve_logprobs" not in df:
        return {}
    d = df[df.curve_logprobs.notna() & (df.kind != "unparsed")]
    out = {}
    for (mode, arm), g in d.groupby(["b_mode", "b_arm"]):
        res = {}
        for k in ("unchanged", "helpful", "harmful", "wrong_wrong"):
            h = g[g.kind == k]
            if not len(h):
                continue
            p = np.array([[math.exp(lp[int(a)]) for lp in cv] for cv, a in zip(h.curve_logprobs, h.answer, strict=True)])
            m = np.array([list(x) for x in h.curve_letter_mentioned], float)
            res[k] = {"n": len(h), "p_final_answer": p.mean(0).round(4).tolist(), "letter_mentioned": m.mean(0).round(4).tolist()}
        out[f"{mode} {arm}"] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--units-per-hour", type=float, default=6.73)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args(argv)
    df, calls = load(Path(a.root), a.job)
    r = {"summary": summary(df, calls, a.units_per_hour)}
    if not a.pilot:
        r.update({"contrasts": contrasts(df), "order_effect": order_effect(df), "commitment": commitment(df), "curves": curves(df)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("expb_pilot.json" if a.pilot else "expb_results.json")).write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r["summary"], indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
