"""Runtime: operational status of a detached command (NAME from /content/session_env.json):
alive, last log lines, GPU memory, disk, and a job status file if the command writes one.
Prints no results."""
import json
import os
import subprocess

env = json.load(open("/content/session_env.json"))
name = env["NAME"]
pidfile, logfile = env.get("PIDFILE", f"/content/{name}.pid"), env.get("LOGFILE", f"/content/{name}.log")
pid = open(pidfile).read().strip() if os.path.exists(pidfile) else None
print(json.dumps({"alive": bool(pid) and os.path.exists(f"/proc/{pid}")}))
for p in env.get("STATUS_FILES", []):
    if os.path.exists(p):
        s = json.load(open(p))
        print(json.dumps({k: s.get(k) for k in ("state", "chunks_done", "chunks_total", "elapsed_hours_total", "rows", "complete")}))
if os.path.exists(logfile):
    print("".join(open(logfile).readlines()[-5:])[-1500:])
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout.splitlines()[-1])
