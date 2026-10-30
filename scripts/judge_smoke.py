"""Judge smoke test (revision gate, section 5): each judge in its own process, in sequence, on 10
events (both orders) and 10 questions' own answers, to measure tokens, seconds and memory.

    python scripts/judge_smoke.py --out /content/judge_smoke [--models gemma qwen3 qwen3_think llama]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", default=["gemma", "qwen3", "qwen3_think", "llama"])
    ap.add_argument("--events-limit", type=int, default=10)
    ap.add_argument("--questions-limit", type=int, default=10)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = []
    for m in a.models:
        t = time.time()
        cmd = [sys.executable, str(ROOT / "scripts/judge_run.py"), "--model", m, "--task", "judge", "own",
               "--events-limit", str(a.events_limit), "--questions-limit", str(a.questions_limit), "--out", str(out)]
        with open(out / f"{m}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        seq.append({"model": m, "exit": rc, "seconds": round(time.time() - t, 1)})
        (out / "sequence.json").write_text(json.dumps(seq, indent=1))
        print(json.dumps(seq[-1]), flush=True)
    return 0 if all(s["exit"] == 0 for s in seq) else 1


if __name__ == "__main__":
    raise SystemExit(main())
