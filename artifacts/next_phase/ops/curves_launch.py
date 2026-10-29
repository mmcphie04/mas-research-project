"""Runtime: verify and unpack the uploaded input build, then start one commit_curves command
detached (log -> <out>/<name>.log, pid -> <out>/<name>.pid). Environment: NAME (smoke|core),
BUILD_TGZ, BUILD_SHA256, CURVE_ARGS (JSON list, appended after the subcommand's fixed args)."""
import json as _json
import os as _os

# parameters: /content/curves_env.json (uploaded with the session files), else the environment
if _os.path.exists("/content/curves_env.json"):
    _os.environ.update({k: (v if isinstance(v, str) else _json.dumps(v))
                        for k, v in _json.load(open("/content/curves_env.json")).items()})

import hashlib
import json
import os
import subprocess
import sys
import tarfile

code, out = "/content/mas-research-project", "/content/next_commit"
name = os.environ["NAME"]
os.makedirs(out, exist_ok=True)
tgz = os.environ["BUILD_TGZ"]
h = hashlib.sha256(open(tgz, "rb").read()).hexdigest()
if h != os.environ["BUILD_SHA256"]:
    raise SystemExit(f"build tarball hash {h} != expected; refusing to start")
if not os.path.isdir(f"{out}/build_core"):
    with tarfile.open(tgz) as t:
        t.extractall(out, filter="data")
fixed = {"smoke": ["smoke", "--build", f"{out}/build_core", "--out", f"{out}/smoke"],
         "core": ["run", "--build", f"{out}/build_core", "--expect-sources", "core", "--out", f"{out}/rows_core"]}[name]
cmd = [sys.executable, "scripts/commit_curves.py", *fixed, *json.loads(os.environ.get("CURVE_ARGS", "[]"))]
log = open(f"{out}/{name}.log", "a")
p = subprocess.Popen(cmd, cwd=code, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ, "PYTHONPATH": f"{code}/src", "PYTHONUNBUFFERED": "1"})
open(f"{out}/{name}.pid", "w").write(str(p.pid))
print("launched", name, "pid", p.pid, "cmd", " ".join(cmd[1:]))
