"""Run several gen_run.py or hf_readouts.py steps in ONE process, so the model is loaded once (the
engine / HF model is cached per process). Different vLLM models still need separate processes
(vLLM does not reliably release GPU memory in-process), so a session groups steps by model.

    python scripts/multi_step.py --module gen_run --steps '[["--task","bank","--mode","qwen_nt",...], ...]' \\
        --out-root /content/expab --job smoke

--out-root and --job are appended to every step. Stops at the first step that fails.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, choices=["gen_run", "hf_readouts"])
    ap.add_argument("--steps", required=True, help="JSON list of argument lists")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    spec = importlib.util.spec_from_file_location(a.module, ROOT / "scripts" / f"{a.module}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[a.module] = mod
    spec.loader.exec_module(mod)
    for i, args in enumerate(json.loads(a.steps)):
        t0 = time.time()
        rc = mod.main([*args, "--out-root", a.out_root, "--job", a.job])
        print(json.dumps({"substep": i, "args": args, "exit": rc, "seconds": round(time.time() - t0, 1)}), flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
