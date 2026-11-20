"""Run a session's steps in order on the runtime, each in its own process with its environment's
Python: generation steps with the vLLM environment, readout steps with the system (HF) one. vLLM and
HF never hold the GPU at the same time. GPU memory is sampled every few seconds; each step's peak,
seconds and exit are recorded. The first failing step ends the session (the supervisor then backs
everything up and stops the runtime).

    python scripts/gen_session.py --plan session.json --out-root /content/expab --job expab

session.json: {"steps": [{"name": "...", "env": "vllm" | "hf", "script": "gen_run.py" | "hf_readouts.py",
                          "args": [...]}]}
--out-root and --job are appended to every step. Writes <out-root>/<job>-session-summary.json (a
small file the backup syncs) after every step: exit, seconds, peak GPU memory, vLLM's KV-cache
lines and the log tail.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = {"vllm": "/content/vllm_env/bin/python", "hf": sys.executable}


def gpu_used_mb() -> float | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return float(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Sampler(threading.Thread):
    def __init__(self, every=3.0):
        super().__init__(daemon=True)
        self.every, self.peak, self.stop_flag = every, None, False

    def run(self):
        while not self.stop_flag:
            v = gpu_used_mb()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            time.sleep(self.every)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--python-vllm", default=PYTHONS["vllm"])
    a = ap.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    pythons = {"vllm": a.python_vllm, "hf": sys.executable}
    record = {"plan": plan, "steps": []}
    for step in plan["steps"]:
        cmd = [pythons[step["env"]], str(ROOT / "scripts" / step["script"]), *step["args"],
               "--out-root", str(out), "--job", a.job]
        s = Sampler()
        s.start()
        t0 = time.time()
        # vLLM manages its own GPU memory: the HF stack's allocator setting is not passed to it
        drop = {"PYTHONPATH"} | ({"PYTORCH_CUDA_ALLOC_CONF"} if step["env"] == "vllm" else set())
        with open(out / f"{a.job}-{step['name']}.steplog", "w") as log:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**{k: v for k, v in os.environ.items() if k not in drop},
                                     "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}).returncode
        s.stop_flag = True
        lines = (out / f"{a.job}-{step['name']}.steplog").read_text(errors="replace").splitlines()
        record["steps"].append({"name": step["name"], "env": step["env"], "exit": rc, "seconds": round(time.time() - t0, 1),
                                "peak_gpu_mb": s.peak, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "vllm_memory_lines": [x[-300:] for x in lines if "KV cache" in x or "Maximum concurrency" in x][:6],
                                "log_tail": [x[-300:] for x in lines[-15:]]})
        tmp = out / f"{a.job}-session-summary.json.tmp"  # "-summary.json": a small file the backup syncs
        tmp.write_text(json.dumps(record, indent=1))
        tmp.replace(out / f"{a.job}-session-summary.json")
        print(json.dumps(record["steps"][-1]), flush=True)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
