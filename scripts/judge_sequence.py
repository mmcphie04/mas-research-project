"""Full cross-model judge run (docs/REVISION_JUDGE_PLAN.md): each step in its own process, in the
committed order, so a stop at the cap loses the least informative work last.

    python scripts/judge_sequence.py --out /content/judge
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REUSE = "artifacts/revision_gate/judge/gemma_reuse.json"
STEPS = [  # (label, judge_run arguments)
    ("gemma_judge", ["--model", "gemma", "--task", "judge"]),
    ("qwen3", ["--model", "qwen3", "--task", "judge", "own"]),
    ("llama", ["--model", "llama", "--task", "judge", "own"]),
    ("qwen3_think", ["--model", "qwen3_think", "--task", "judge", "own"]),
    ("gemma_own", ["--model", "gemma", "--task", "own", "--skip-questions", REUSE]),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = []
    for label, args in STEPS:
        step_out = out / label  # each step's files in its own directory (gemma runs twice)
        step_out.mkdir(exist_ok=True)
        t = time.time()
        with open(out / f"{label}.log", "w") as log:
            rc = subprocess.run([sys.executable, str(ROOT / "scripts/judge_run.py"), *args, "--out", str(step_out)],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        seq.append({"step": label, "exit": rc, "seconds": round(time.time() - t, 1)})
        (out / "sequence.json").write_text(json.dumps(seq, indent=1))
        (step_out / "DONE").write_text(json.dumps(seq[-1]))
        print(json.dumps(seq[-1]), flush=True)
    return 0 if all(s["exit"] == 0 for s in seq) else 1


if __name__ == "__main__":
    raise SystemExit(main())
