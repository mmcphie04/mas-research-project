"""Runtime: operational status of a commit_curves command (NAME = smoke|core): alive, state,
expected/done/missing per mode, smoke verdict if written, last log lines, GPU memory, disk.
Prints no curve results."""
import json as _json
import os as _os

# parameters: /content/curves_env.json (uploaded with the session files), else the environment
if _os.path.exists("/content/curves_env.json"):
    _os.environ.update({k: (v if isinstance(v, str) else _json.dumps(v))
                        for k, v in _json.load(open("/content/curves_env.json")).items()})

import json
import os
import subprocess

out, name = "/content/next_commit", os.environ["NAME"]
d = {"smoke": f"{out}/smoke", "core": f"{out}/rows_core"}[name]
pid = open(f"{out}/{name}.pid").read().strip() if os.path.exists(f"{out}/{name}.pid") else None
st = json.load(open(f"{d}/run_status.json")) if os.path.exists(f"{d}/run_status.json") else {}
rep = json.load(open(f"{d}/smoke_report.json")) if os.path.exists(f"{d}/smoke_report.json") else {}
print(json.dumps({"alive": bool(pid) and os.path.exists(f"/proc/{pid}"), "state": st.get("state"),
                  "modes": {m: {k: v[k] for k in ("expected", "done", "missing")} for m, v in st.get("modes", {}).items()},
                  "seconds": st.get("seconds"), "smoke_overall": rep.get("overall")}))
if os.path.exists(f"{out}/{name}.log"):
    print("".join(open(f"{out}/{name}.log").readlines()[-4:])[-1200:])
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1])
