"""Operational report after the chooser launch (launch 4). CPU only. OUTCOME-BLIND: no gold answer
and no correctness is read, so no choosing outcome (development or held-out) is looked at here;
those are analysed once, as committed (scripts/expa_analysis.py).

    python scripts/expab_choose_report.py --root runs/expab --job expab --out artifacts/expab/launch4

Per arm: completion (complete / partial k of n chunks / not run); questions with a disagreeing pool (chooser invoked), requests, pick-parse rate (a valid
"Best candidate: N"), thinking-cap and answer-cap rates, context refusals, mean generated tokens,
GPU-seconds and tokens/s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.gen import archive as A  # noqa: E402

BLIND = {"meta_gold", "gold", "correct", "gold_display"}


def report(root: Path, job: str, tag: str | None = None) -> dict:
    calls = A.read_calls(root, job)
    parts = sorted(p for p in set(calls.part) if p.startswith("choose_") and (tag is None or p.endswith(f"_{tag}"))
                   and not (tag is None and "__" in p))
    out = {}
    for part in parts:
        r = A.read_rows(root, job, part)
        r = r[[c for c in r.columns if c not in BLIND]]  # outcome-blind by construction
        c = calls[calls.part == part]
        out[part.split("__")[0][len("choose_"):]] = {
            "questions": int(r.question_id.nunique()), "requests": len(r),
            "pick_parse_rate": float((r.pick_position > 0).mean()),
            "thinking_cap_rate": float(r.thinking_capped.mean()) if r.thinking_tokens.sum() else None,
            "answer_cap_rate": float((r.answer_finish == "length").mean()),
            "refused_context": int(r.skipped.notna().sum()),
            "mean_prompt_tokens": float(r.prompt_tokens.mean()), "mean_generated_tokens": float(r.generated_tokens.mean()),
            "gpu_seconds": float(c.seconds.sum()), "tokens_per_second": float(c.generated_tokens.sum() / c.seconds.sum())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "scripts"))
    from expa_analysis import arm_completion  # noqa: PLC0415 -- reads the job record only

    r = {"arm_completion": arm_completion(Path(a.root), a.job), "arms": report(Path(a.root), a.job, a.tag)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "choose_report.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
