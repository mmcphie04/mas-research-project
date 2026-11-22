"""Operational report after the Experiment B full run (launch 5). CPU only. OUTCOME-BLIND: no gold
answer, no correctness, no before/after comparison (change rates and helpful/harmful kinds are B
outcomes, analysed once in scripts/expb_analysis.py).

    python scripts/expb_ops_report.py --root runs/expab --job expab --out artifacts/expab/launch5

Per B part (mode x arm x tag): completion (chunks done / expected, from the job record), panels,
agents (rows), revision parse rate, thinking-cap and answer-cap rates, context refusals, mean
prompt / generated tokens, GPU-seconds; readout completion (rows with readouts / rows) and readout
seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.gen import archive as A  # noqa: E402
from mad.provenance import is_complete  # noqa: E402

BLIND = {"meta_gold", "gold", "correct", "gold_display"}


def report(root: Path, job: str, tag: str = "full") -> dict:
    root = Path(root)
    rec = json.loads((root / f"{job}-job.json").read_text())
    calls = A.read_calls(root, job)
    out = {}
    for part, n in sorted(rec["parts"].items()):
        if not part.startswith("b_") or part.endswith("_ro") or not (part.endswith(f"_{tag}") or part.endswith(f"_{tag}x")):
            continue
        done = sum(is_complete(A.chunk_dir(root, job, part, i)) for i in range(n))
        e = {"chunks_done": done, "chunks_expected": n}
        r = A.read_rows(root, job, part)
        if len(r):
            r = r[[c for c in r.columns if c not in BLIND]]  # outcome-blind by construction
            c = calls[calls.part == part]
            trip = r["meta_triple"].astype(str) if "meta_triple" in r else "0"
            e.update({"panels": int((r.question_id + "|" + trip).nunique()), "questions": int(r.question_id.nunique()),
                      "agents": len(r), "revision_parse_rate": float(r.parsed.mean()),
                      "thinking_cap_rate": float(r.thinking_capped.mean()) if r.thinking_tokens.sum() else None,
                      "answer_cap_rate": float((r.answer_finish == "length").mean()),
                      "refused_context": int(r.skipped.notna().sum()),
                      "mean_prompt_tokens": float(r.prompt_tokens.mean()), "mean_generated_tokens": float(r.generated_tokens.mean()),
                      "gpu_seconds": float(c.seconds.sum())})
            ro = A.read_rows(root, job, f"{part}_ro")
            rc = calls[calls.part == f"{part}_ro"]
            e["readouts"] = {"rows": len(ro), "share_of_agents": len(ro) / len(r), "seconds": float(rc.seconds.sum())}
        out[part] = e
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--tag", default="full", help="the run's part tag; its extension <tag>x is included")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = report(Path(a.root), a.job, a.tag)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "b_ops_report.json").write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
