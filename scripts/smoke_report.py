"""GPU smoke-test report and re-pricing (docs/EXPAB_PLAN.md, section 6). CPU only.

    python scripts/smoke_report.py --root runs/expab_smoke --job smoke --out artifacts/expab/smoke

Per part (model x mode x task): requests, throughput (generated tokens / call seconds, overall; and
per sequence: each request's generated tokens / the seconds of its own call(s), averaged), parse
rate, mean prompt / thinking / answer tokens, thinking-cap rate, refused (context) requests; the
session's peak GPU memory per step and vLLM's KV-cache lines. Sanity (parts ``sanity_<mode>``):
vLLM against the judge run's HF-generated own answers on the same questions and protocol -- parse
rate, mean generated tokens, accuracy (paired question bootstrap of the difference).

Re-pricing: measured GPU-seconds per request by part type x the full plan's request counts (the
chooser call rate -- questions whose pool disagrees -- and B's helpful+harmful per panel also from
the smoke), at --units-per-hour, plus per-session overhead. Smoke calls are smaller than the full
run's (fewer concurrent requests), so per-request seconds from the smoke are, if anything, high.
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
from mad.gen import tasks as T  # noqa: E402

N_BOOT = 10000
# full-plan request counts (docs/EXPAB_PLAN.md); chooser and B counts depend on measured rates
FULL = {"bank_gemma": 1500 * 10, "bank_qwen_nt": 1500 * 10, "bank_qwen_t": 1500 * 5,
        "judge_traces": 606, "judge_own": 447}
SESSION_OVERHEAD_H = 0.25  # setup + vLLM environment + model loads + final backup, per session (estimate)


def base(part: str) -> str:
    return part.split("__")[0]


def per_part(rows: pd.DataFrame, calls: pd.DataFrame) -> dict:
    out = {}
    for part, c in calls.groupby("part"):
        r = rows[rows._part == part]
        sec = dict(zip(c.call_id, c.seconds, strict=True))
        own = []
        for x in r.to_dict("records"):
            s = sum(sec.get(k, 0.0) for k in (x.get("think_call"), x.get("answer_call")) if k)
            if s > 0:
                own.append(x["generated_tokens"] / s)
        gen = c.generated_tokens.sum() if "generated_tokens" in c else np.nan
        out[part] = {
            "requests": int(len(r)), "seconds": float(c.seconds.sum()),
            "seconds_per_request": float(c.seconds.sum() / max(len(r), 1)),
            "tokens_per_second_overall": float(gen / c.seconds.sum()) if c.seconds.sum() else None,
            "tokens_per_second_per_sequence": float(np.mean(own)) if own else None,
            "parse_rate": float(r.parsed.mean()) if "parsed" in r and len(r) and r.answer_tokens.sum() else None,
            "mean_prompt_tokens": float(r.prompt_tokens.mean()) if len(r) else None,
            "mean_thinking_tokens": float(r.thinking_tokens.mean()) if len(r) else None,
            "mean_answer_tokens": float(r.answer_tokens.mean()) if len(r) else None,
            "thinking_cap_rate": float(r.thinking_capped.mean()) if len(r) and r.thinking_tokens.sum() else None,
            "answer_cap_rate": float((r.answer_finish == "length").mean()) if len(r) else None,
            "refused_context": int(r.skipped.notna().sum()) if "skipped" in r else 0}
    return out


def sanity(rows: pd.DataFrame, prev: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from judge_analysis import events_table, load_own  # noqa: PLC0415
    gold = events_table(ROOT).drop_duplicates("question_id").set_index("question_id").gold.to_dict()
    hf_gen = {"qwen_nt": prev / "judge" / "qwen3", "gemma": prev / "judge" / "gemma_own"}
    out = {}
    for mode in ("gemma", "qwen_nt"):
        r = rows[rows._part.map(base) == f"sanity_{mode}"]
        if not len(r):
            continue
        hf_own = load_own(prev, "gemma" if mode == "gemma" else "qwen3", ROOT)
        files = sorted(hf_gen[mode].glob("*_own.jsonl"))
        hf_rows = pd.DataFrame([json.loads(x) for f in files for x in open(f)]) if files else pd.DataFrame()
        qs = sorted(set(r.question_id) & set(hf_own))
        v = r[r.question_id.isin(qs)]
        acc_v = v.groupby("question_id").apply(lambda g: float((g.answer == gold[g.name]).mean()), include_groups=False)
        acc_h = pd.Series({q: float(np.mean([a == gold[q] for a in hf_own[q]])) for q in qs})
        d = (acc_v - acc_h.loc[acc_v.index]).to_numpy()
        b = d[np.random.default_rng(0).integers(0, len(d), (N_BOOT, len(d)))].mean(1) if len(d) else np.array([np.nan])
        h = hf_rows[hf_rows.question_id.astype(str).isin(qs)] if len(hf_rows) else hf_rows
        out[mode] = {"questions": len(qs),
                     "vllm": {"parse_rate": float(v.parsed.mean()), "mean_generated_tokens": float(v.generated_tokens.mean()),
                              "accuracy": float(acc_v.mean())},
                     "hf": {"parse_rate": float(h.parsed.mean()) if len(h) else None,
                            "mean_generated_tokens": float(h.generated_tokens.mean()) if len(h) else None,
                            "accuracy": float(acc_h.mean())},
                     "accuracy_difference": {"mean": float(d.mean()) if len(d) else None, "lo": float(np.quantile(b, .025)),
                                             "hi": float(np.quantile(b, .975))}}
    return out


def disagreement_rates(rows: pd.DataFrame) -> dict:
    bank = rows[rows._part.str.startswith("bank_")]
    idx = T.bank_index(bank)
    qs = sorted(set(bank.question_id))
    present = set(bank["mode"])
    return {arm: float(np.mean([T.disagreeing(T.pool_rows(idx, q, spec)) for q in qs]))
            for arm, (_, spec) in T.ARMS.items() if {m for m, _ in spec} <= present}


def price(pp: dict, rows: pd.DataFrame, units_per_hour: float, b_panels_full: int, b_pilot_panels: int) -> dict:
    sec = {}
    for part, v in pp.items():
        sec.setdefault(base(part), []).append((v["seconds"], v["requests"]))
    spr = {k: sum(s for s, _ in v) / max(sum(n for _, n in v), 1) for k, v in sec.items()}
    lines = {}
    for k, n in FULL.items():
        if k in spr:
            lines[k] = {"requests": n, "gpu_hours": n * spr[k] / 3600}
    rates = disagreement_rates(rows)
    for arm, rate in rates.items():
        k = f"choose_{arm}"
        if k in spr:
            n = int(round(1500 * rate)) * 3
            lines[k] = {"requests": n, "gpu_hours": n * spr[k] / 3600, "call_rate": rate}
    for mode in ("qwen_nt", "qwen_t"):
        for arm in T.B_ARMS:
            k = f"b_{mode}_{arm}"
            src = spr.get(k) or spr.get(f"b_{mode}_debate")
            if src:
                lines[f"B pilot {mode} {arm}"] = ({"requests": b_pilot_panels * 3, "gpu_hours": b_pilot_panels * 3 * src / 3600}
                                                  if arm in ("debate", "alone") else {"requests": 0, "gpu_hours": 0.0})
                lines[f"B full {mode} {arm}"] = {"requests": b_panels_full * 3, "gpu_hours": b_panels_full * 3 * src / 3600}
    for k in list(spr):
        if k.endswith("_ro") and k.startswith("b_"):
            lines.setdefault("B readouts (per 1,000 rows)", {"requests": 1000, "gpu_hours": 1000 * spr[k] / 3600})
    total = sum(v["gpu_hours"] for v in lines.values())
    return {"seconds_per_request": spr, "lines": lines, "gpu_hours": total,
            "units": total * units_per_hour, "units_per_hour": units_per_hour,
            "assumptions": {"b_panels_full_per_mode_arm": b_panels_full, "b_pilot_panels_per_mode": b_pilot_panels,
                            "session_overhead_hours_each": SESSION_OVERHEAD_H}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--prev", default=str(ROOT / "runs/judge_full"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--units-per-hour", type=float, default=6.73)
    ap.add_argument("--b-panels-full", type=int, default=300)
    ap.add_argument("--b-pilot-panels", type=int, default=100)
    a = ap.parse_args(argv)
    root = Path(a.root)
    calls = A.read_calls(root, a.job)
    frames = [A.read_rows(root, a.job, p).assign(_part=p) for p in sorted(set(calls.part)) if not p.endswith("_ro")]
    rows = pd.concat([f for f in frames if len(f)], ignore_index=True)
    pp = per_part(rows, calls[~calls.part.str.endswith("_ro")])
    ro = {p: {"seconds_per_row": float(c.seconds.sum() / max(c.requests.sum(), 1))} for p, c in calls[calls.part.str.endswith("_ro")].groupby("part")}
    sess = root / f"{a.job}-session-summary.json"
    r = {"parts": pp, "readouts": ro, "session": json.loads(sess.read_text()) if sess.exists() else None,
         "sanity": sanity(rows, Path(a.prev)) if (Path(a.prev) / "judge").exists() else None,
         "pricing": price({**pp, **{p: {"seconds": float(calls[calls.part == p].seconds.sum()),
                                         "requests": int(calls[calls.part == p].requests.sum())} for p in ro}},
                          rows, a.units_per_hour, a.b_panels_full, a.b_pilot_panels)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_report.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({"gpu_hours": r["pricing"]["gpu_hours"], "units": r["pricing"]["units"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

