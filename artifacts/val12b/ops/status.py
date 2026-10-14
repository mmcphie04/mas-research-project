"""Runtime: operational status only (no accuracy): job state, chunks, elapsed, per-chunk timing,
failures, parse/cap rates, disk."""
import json
import os
import subprocess
import sys

out, name = os.environ.get("OUT", "/content/mas_runs"), os.environ["NAME"]
p = f"{out}/{name}-status.json"
s = json.load(open(p)) if os.path.exists(p) else {}
pid = open(f"{out}/{name}.pid").read().strip() if os.path.exists(f"{out}/{name}.pid") else None
alive = bool(pid) and os.path.exists(f"/proc/{pid}")
t = [{k: c.get(k) for k in ("chunk", "questions", "seconds", "seconds_per_question")} for c in s.get("timings", [])]
print(json.dumps({"alive": alive, "state": s.get("state"), "chunks_done": s.get("chunks_done"),
                  "chunks_total": s.get("chunks_total"), "elapsed_hours_total": s.get("elapsed_hours_total"),
                  "timings": t, "failures": len(s.get("failures", [])),
                  "parse_cap_last_chunk": s.get("diagnostics_last_chunk")}))
log = f"{out}/{name}.log"
if os.path.exists(log):
    print("".join(open(log).readlines()[-4:])[-1200:])
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1])
sys.stdout.flush()
