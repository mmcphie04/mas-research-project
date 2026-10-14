"""Runtime: start one driver job detached, log to <out>/<name>.log, pid to <out>/<name>.pid.
Arguments come from the environment (JOB_ARGS as a JSON list; NAME)."""
import json
import os
import subprocess
import sys

code, out = "/content/mas-research-project", os.environ.get("OUT", "/content/mas_runs")
name = os.environ["NAME"]
os.makedirs(out, exist_ok=True)
cmd = [sys.executable, "scripts/run_experiment.py", *json.loads(os.environ["JOB_ARGS"])]
log = open(f"{out}/{name}.log", "a")
p = subprocess.Popen(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "PYTHONPATH": f"{code}/src", "PYTHONUNBUFFERED": "1"})
open(f"{out}/{name}.pid", "w").write(str(p.pid))
print("launched", name, "pid", p.pid, "cmd", " ".join(cmd[1:]))
