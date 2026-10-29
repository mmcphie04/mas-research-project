"""Runtime (copied from artifacts/val12b/ops/setup.py): code at the pinned commit (from an uploaded git bundle), HF token into its standard
location (never printed), packages, GPU and disk report. COMMIT comes from the environment."""
import json as _json
import os as _os

# parameters: /content/curves_env.json (uploaded with the session files), else the environment
if _os.path.exists("/content/curves_env.json"):
    _os.environ.update({k: (v if isinstance(v, str) else _json.dumps(v))
                        for k, v in _json.load(open("/content/curves_env.json")).items()})

import os
import subprocess
import sys

COMMIT = os.environ["COMMIT"]


def sh(*a, cwd=None):
    r = subprocess.run(list(a), cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{a[0]} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()


os.makedirs("/root/.cache/huggingface", exist_ok=True)
if os.path.exists("/content/hf_token.tmp"):
    os.replace("/content/hf_token.tmp", "/root/.cache/huggingface/token")
    os.chmod("/root/.cache/huggingface/token", 0o600)
code = "/content/mas-research-project"
if not os.path.isdir(code):
    sh("git", "clone", "--quiet", "/content/repo.bundle", code)
sh("git", "fetch", "--quiet", "/content/repo.bundle", "review-9e45160-fixes", cwd=code)
sh("git", "checkout", "--quiet", "--detach", COMMIT, cwd=code)
assert sh("git", "rev-parse", "HEAD", cwd=code) == COMMIT
assert not sh("git", "status", "--porcelain", cwd=code), "checkout has local changes"
print("code at", sh("git", "log", "-1", "--format=%h %s", cwd=code))
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{code}/requirements-colab.txt"],
                   capture_output=True, text=True)
print("pip exit", r.returncode, r.stderr[-1500:] if r.returncode else "")
import importlib.metadata as md  # noqa: E402

for p in ("torch", "transformers", "accelerate", "datasets", "numpy", "pandas", "scikit-learn", "pyarrow"):
    try:
        print(f"{p:<14}", md.version(p))
    except Exception as e:  # noqa: BLE001
        print(p, "MISSING", e)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout)
print("token present:", os.path.exists("/root/.cache/huggingface/token"))
