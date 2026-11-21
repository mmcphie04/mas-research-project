"""Report after a bank launch (docs/EXPAB_PLAN.md, review of Stop 2, item D). CPU only.

    python scripts/expab_launch_report.py --root runs/expab --job expab --tag orig --out artifacts/expab/launch1

* Per bank mode: rows, parse rate, mean prompt / thinking / answer / generated tokens, thinking-cap
  and answer-cap rates, context refusals; measured aggregate tokens/s and GPU-seconds per request
  at full scale.
* Gemma's overall bank accuracy (all its samples on the launch's questions) next to main250's
  single-sample accuracy (HF, the same protocol): the vLLM-vs-HF check asked for after the smoke
  test. No other accuracy is computed (no choosing outcome is looked at).
* Disagreement rate per chooser pool and disputed-panel supply per mode (dev / held-out; panels
  per question) -- answers and parse status only, no gold.
* Re-pricing of later launches from this launch's measured throughput.
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

from mad.gen import archive as A  # noqa: E402
from mad.gen import tasks as T  # noqa: E402


def bank_parts(root: Path, job: str, tag: str) -> list[str]:
    calls = A.read_calls(root, job)
    return sorted(p for p in set(calls.part) if p.startswith("bank_") and p.endswith(f"_{tag}")) if len(calls) else []


def per_mode(root: Path, job: str, tag: str) -> tuple[dict, pd.DataFrame]:
    calls = A.read_calls(root, job)
    out, frames = {}, []
    for part in bank_parts(root, job, tag):
        r = A.read_rows(root, job, part)
        c = calls[calls.part == part]
        mode = r["mode"].iloc[0]
        frames.append(r)
        out[mode] = {"rows": len(r), "questions": int(r.question_id.nunique()), "parse_rate": float(r.parsed.mean()),
                     "mean_prompt_tokens": float(r.prompt_tokens.mean()), "mean_thinking_tokens": float(r.thinking_tokens.mean()),
                     "mean_answer_tokens": float(r.answer_tokens.mean()), "mean_generated_tokens": float(r.generated_tokens.mean()),
                     "thinking_cap_rate": float(r.thinking_capped.mean()) if r.thinking_tokens.sum() else None,
                     "thinking_eos_rate": float(r.thinking_eos.mean()) if r.thinking_tokens.sum() else None,
                     "answer_cap_rate": float((r.answer_finish == "length").mean()),
                     "refused_context": int(r.skipped.notna().sum()),
                     "gpu_seconds": float(c.seconds.sum()), "gpu_seconds_per_request": float(c.seconds.sum() / len(r)),
                     "tokens_per_second": float(c.generated_tokens.sum() / c.seconds.sum())}
    return out, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gemma_accuracy(bank: pd.DataFrame) -> dict:
    g = bank[bank["mode"] == "gemma"]
    if not len(g):
        return {}
    acc = (g.answer == g.meta_gold).astype(float)
    per_q = acc.groupby(g.question_id).mean().to_numpy()
    b = per_q[np.random.default_rng(0).integers(0, len(per_q), (10000, len(per_q)))].mean(1)
    ref = None
    p = ROOT / "runs/main250/main250-pool/agent_states.parquet"
    if p.exists():
        m = pd.read_parquet(p, columns=["phase", "correct"])
        ref = float(m[m.phase == "pre"].correct.mean())
    return {"vllm_bank_accuracy": float(acc.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
            "questions": len(per_q), "samples": len(g), "main250_hf_single_sample_accuracy": ref,
            "note": "different questions (main250 positions 321-520 vs these); same protocol"}


def main250_like_for_like(root: Path, job: str) -> dict:
    """Gemma's vLLM bank protocol on main250's own 200 questions (part sanity_main250_gemma) against
    main250's HF pool (10 samples per question, phase pre). Paired by question; 95% bootstrap CI of
    the difference (vLLM - HF), resampling questions. Rule committed before the data were seen
    (docs/EXPAB_PLAN.md): a real gap = that CI excludes zero."""
    v = A.read_rows(root, job, "sanity_main250_gemma")
    p = ROOT / "runs/main250/main250-pool/agent_states.parquet"
    if not len(v) or not p.exists():
        return {}
    m = pd.read_parquet(p, columns=["phase", "question_id", "correct"])
    hf = m[m.phase == "pre"].groupby("question_id").correct.mean()
    vq = (v.answer == v.meta_gold).astype(float).groupby(v.question_id).mean()
    qs = sorted(set(vq.index) & set(hf.index))
    d = (vq.loc[qs] - hf.loc[qs]).to_numpy()
    b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    lo, hi = float(np.quantile(b, .025)), float(np.quantile(b, .975))
    return {"questions": len(qs), "vllm_samples": len(v), "vllm_accuracy": float(vq.loc[qs].mean()),
            "hf_accuracy": float(hf.loc[qs].mean()), "difference_vllm_minus_hf": float(d.mean()), "ci95": [lo, hi],
            "real_gap": bool(lo > 0 or hi < 0), "vllm_parse_rate": float(v.parsed.mean()),
            "vllm_answer_cap_rate": float((v.answer_finish == "length").mean()),
            "vllm_mean_generated_tokens": float(v.generated_tokens.mean())}


def disagreement(bank: pd.DataFrame, assign: dict) -> dict:
    idx = T.bank_index(bank)
    qs = sorted(set(bank.question_id))
    present = set(bank["mode"])
    out = {}
    for arm, (_, spec) in T.ARMS.items():
        if {m for m, _ in spec} <= present:
            out[arm] = {"pool": spec, "rate": float(np.mean([T.disagreeing(T.pool_rows(idx, q, spec)) for q in qs]))}
    supply = {}
    for mode in ("qwen_nt", "qwen_t"):
        if mode not in present:
            continue
        for split in ("dev", "heldout"):
            sq = [q for q in qs if assign.get(q) == split]
            per_q = [sum(1 for t in T.TRIPLES if T.panel(idx, mode, q, t)) for q in sq]
            supply[f"{mode} {split}"] = {"questions": len(sq), "disputed_panels": int(sum(per_q)),
                                         "panels_per_question": {int(k): int(v) for k, v in pd.Series(per_q).value_counts().sort_index().items()},
                                         "first_triple_only": int(sum(1 for q in sq if T.panel(idx, mode, q, 0)))}
    return {"chooser_pools": out, "disputed_panel_supply": supply}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--tag", required=True, help="the bank launch's part tag (orig / ext)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _, assign, _ = T.load_questions(ROOT / "artifacts/expab/plan")
    modes, bank = per_mode(Path(a.root), a.job, a.tag)
    r = {"bank": modes, "gemma_accuracy_check": gemma_accuracy(bank),
         "gemma_main250_like_for_like": main250_like_for_like(Path(a.root), a.job), **disagreement(bank, assign)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "launch_report.json").write_text(json.dumps(r, indent=1, default=str))
    print(json.dumps(r, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
