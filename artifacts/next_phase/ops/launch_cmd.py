"""Runtime: start one command detached in the pinned checkout (log -> /content/<NAME>.log,
pid -> /content/<NAME>.pid). Parameters from /content/session_env.json: NAME, CMD (JSON list,
run with this Python from the checkout), optional REQUIRE {path: sha256} checked first, optional
EXTRACT {tarball: destination directory} unpacked after the checks."""
import hashlib
import json
import os
import subprocess
import sys
import tarfile

env = json.load(open("/content/session_env.json"))
code, name = "/content/mas-research-project", env["NAME"]
for path, want in env.get("REQUIRE", {}).items():
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != {want}; refusing to start")
for tgz, dest in env.get("EXTRACT", {}).items():
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tgz) as t:
        t.extractall(dest, filter="data")
log = open(f"/content/{name}.log", "a")
p = subprocess.Popen([sys.executable, *env["CMD"]], cwd=code, stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True, env={**os.environ, "PYTHONPATH": f"{code}/src", "PYTHONUNBUFFERED": "1"})
open(f"/content/{name}.pid", "w").write(str(p.pid))
print("launched", name, "pid", p.pid, " ".join(env["CMD"]))
