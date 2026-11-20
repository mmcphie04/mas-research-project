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


def position_bias(rows: pd.DataFrame) -> dict:
    out = {}
    ch = rows[rows._part.map(base).str.startswith("choose_")]
    for part, g in ch.groupby(ch._part.map(base)):
        ok = g[g.pick_position > 0]
        out[part] = {"runs": len(g), "parsed_picks": len(ok),
                     "share_first": float((ok.pick_position == 1).mean()) if len(ok) else None,
                     "expected_first": float((1 / ok.meta_m).mean()) if len(ok) else None,
                     "share_last": float((ok.pick_position == ok.meta_m).mean()) if len(ok) else None}
    return out


def think_location(rows: pd.DataFrame) -> dict:
    t = rows[rows.thinking_tokens > 0] if "thinking_tokens" in rows else rows.iloc[0:0]
    return {p: {"rows": len(g), "think_in_prompt": float(g.think_in_prompt.mean()),
                "think_in_generation": float(g.think_in_generation.mean())} for p, g in t.groupby("_part")}


def seconds_per_request(pp: dict) -> dict:
    sec = {}
    for part, v in pp.items():
        sec.setdefault(base(part), []).append((v["seconds"], v["requests"]))
    return {k: sum(s for s, _ in v) / max(sum(n for _, n in v), 1) for k, v in sec.items()}


def p2_coverage(spr: dict, rates: dict) -> dict:
    """Each arm's measured cost per question in units of one qwen_t bank sample (and of its own
    chooser's samples): its pool's samples + 3 chooser calls x the call rate."""
    s_qt = spr.get("bank_qwen_t")
    out = {}
    for arm, (chooser, spec) in T.ARMS.items():
        if f"choose_{arm}" not in spr or arm not in rates or not s_qt:
            continue
        pool = sum(k * spr.get(f"bank_{m}", 0.0) for m, k in spec)
        ch = rates[arm] * 3 * spr[f"choose_{arm}"]
        out[arm] = {"chooser": chooser, "call_rate": rates[arm], "seconds_per_question": pool + ch,
                    "in_qwen_t_samples": (pool + ch) / s_qt,
                    "in_own_chooser_samples": (pool + ch) / spr[f"bank_{chooser}"] if spr.get(f"bank_{chooser}") else None}
    return out


def price(pp: dict, rows: pd.DataFrame, units_per_hour: float, b_panels_full: int, b_pilot_panels: int,
          ro: dict) -> dict:
    """GPU-hours and units per LAUNCH (1: bank + judge confirmation; 2: B pilot; 3: choosers; 4: B
    full), from measured seconds per request; plus SESSION_OVERHEAD_H per launch."""
    spr = seconds_per_request(pp)
    rates = disagreement_rates(rows)
    s_ro = {base(k[:-3]) if k.endswith("_ro") else base(k): v["seconds_per_row"] for k, v in ro.items()}
    s_ro_b = np.mean([v for k, v in s_ro.items() if k.startswith("b_")]) if any(k.startswith("b_") for k in s_ro) else 0.0
    launches = {"1 bank + judge confirmation": {}, "2 B pilot": {}, "3 choosers": {}, "4 B full": {}}
    for k, n in FULL.items():
        if k in spr:
            launches["1 bank + judge confirmation"][k] = n * spr[k] / 3600
    for k in ("judge_traces", "judge_own"):
        if k in s_ro:
            launches["1 bank + judge confirmation"][f"{k} readouts (HF)"] = FULL[k] * s_ro[k] / 3600
    for arm, rate in rates.items():
        if f"choose_{arm}" in spr:
            launches["3 choosers"][f"choose_{arm}"] = int(round(1500 * rate)) * 3 * spr[f"choose_{arm}"] / 3600
    for mode in ("qwen_nt", "qwen_t"):
        src = {arm: spr.get(f"b_{mode}_{arm}") or spr.get(f"b_{mode}_debate") for arm in T.B_ARMS}
        for arm in ("debate", "alone"):
            if src[arm]:
                launches["2 B pilot"][f"b_{mode}_{arm}"] = b_pilot_panels * 3 * src[arm] / 3600
        if s_ro_b:
            launches["2 B pilot"][f"b_{mode} readouts (HF)"] = b_pilot_panels * 3 * 2 * s_ro_b / 3600
        for arm in T.B_ARMS:
            if src[arm]:
                launches["4 B full"][f"b_{mode}_{arm}"] = b_panels_full * 3 * src[arm] / 3600
        if s_ro_b:
            launches["4 B full"][f"b_{mode} readouts (HF)"] = b_panels_full * 3 * len(T.B_ARMS) * s_ro_b / 3600
    out = {}
    for name, items in launches.items():
        h = sum(items.values()) + SESSION_OVERHEAD_H
        out[name] = {"items_gpu_hours": {k: round(v, 3) for k, v in items.items()}, "gpu_hours": round(h, 2),
                     "units": round(h * units_per_hour, 1)}
    cov = p2_coverage(spr, rates)
    extra = None
    if any(v["in_qwen_t_samples"] > 4 for v in cov.values()) and spr.get("bank_qwen_t"):
        h = 1500 * 3 * spr["bank_qwen_t"] / 3600
        extra = {"proposal": "raise the qwen_t bank from 5 to 8 samples (an arm costs more than ~4 qwen_t samples, so "
                             "its matched-seconds self-consistency point would lie beyond a 5-sample curve)",
                 "extra_requests": 4500, "gpu_hours": round(h, 2), "units": round(h * units_per_hour, 1)}
    return {"seconds_per_request": spr, "call_rates": rates, "launches": out,
            "total_units": round(sum(v["units"] for v in out.values()), 1), "p2_coverage": cov, "qwen_t_8_samples": extra,
            "units_per_hour": units_per_hour,
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
         "position_bias": position_bias(rows), "think_location": think_location(rows),
         "pricing": price(pp, rows, a.units_per_hour, a.b_panels_full, a.b_pilot_panels, ro)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_report.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps({"launches": {k: v["units"] for k, v in r["pricing"]["launches"].items()},
                      "total_units": r["pricing"]["total_units"]}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

