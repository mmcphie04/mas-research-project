"""G4 morning report: operations only -- no accuracy, no correctness column is ever read.

    python scripts/g4_report.py --dest runs/main250 --job main250 --out artifacts/main250/run

Per chunk (from the job's status record): seconds, peak allocated and reserved GPU memory.
Per condition (from the complete local chunk directories): responses, share capped at the
token limit, stated-answer rate, and generation seconds per response. The pool's responses are
generated in round 0, whose stage time also covers its readout; that is labelled.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mad.job import chunk_tag  # noqa: E402
from mad.provenance import is_complete  # noqa: E402

COLUMNS = ["phase", "n_completion_tokens", "stop_reason", "answer_source"]  # never correctness


def report(dest: Path, job: str) -> dict:
    rec = json.loads((dest / f"{job}-job.json").read_text())
    status = json.loads((dest / f"{job}-status.json").read_text())
    conds = list(rec["identity"]["conditions"])
    cap = rec["identity"]["config"]["model"]["max_new_tokens"]
    chunks = [{k: t.get(k) for k in ("chunk", "questions", "seconds", "peak_allocated_gb", "peak_reserved_gb")}
              for t in status.get("timings", [])]
    by_chunk = {t["chunk"]: t for t in status.get("timings", [])}
    per = {}
    for c in conds:
        frames, gen_s, n_chunks = [], 0.0, 0
        for i, members in enumerate(rec["chunks"]):
            d = dest / f"{job}-{c}-{chunk_tag(i)}"
            if not members or not is_complete(d):
                continue
            s = pd.read_parquet(d / "agent_states.parquet", columns=COLUMNS)
            frames.append(s[s.phase == ("pre" if c == "pool" else "post")])
            st = by_chunk.get(chunk_tag(i), {}).get("stages", {})
            gen_s += st.get("round0_s", 0.0) if c == "pool" else st.get(f"{c}/generation_s", 0.0)
            n_chunks += 1
        if not frames:
            per[c] = {"chunks": 0}
            continue
        x = pd.concat(frames)
        per[c] = {"chunks": n_chunks, "responses": len(x),
                  "capped_share": round(float((x.stop_reason == "length").mean()), 4),
                  "capped": int((x.stop_reason == "length").sum()),
                  "at_cap_tokens": int((x.n_completion_tokens >= cap).sum()),
                  "stated_answer_rate": round(float((x.answer_source == "stated").mean()), 4),
                  "generation_s_per_response": round(gen_s / len(x), 2),
                  "timing_note": "round-0 stage (generation + readout of all 10 agents)" if c == "pool" else "generation stage"}
    return {"job": job, "state": status.get("state"), "chunks_done": status.get("chunks_done"),
            "chunks_total": status.get("chunks_total"), "elapsed_hours_total": status.get("elapsed_hours_total"),
            "failures": status.get("failures"), "token_cap": cap, "chunks": chunks, "conditions": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(ROOT / "runs/main250"))
    ap.add_argument("--job", default="main250")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = report(Path(a.dest), a.job)
    text = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / "g4_operations.json").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
