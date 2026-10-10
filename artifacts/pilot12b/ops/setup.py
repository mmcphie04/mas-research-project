import os, subprocess, sys
COMMIT = "5ff52b38e332546106c18463317f07a21b8b0ae1"
def sh(*a, cwd=None):
    r = subprocess.run(list(a), cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"{a[0]} failed: {r.stderr[-2000:]}")
    return r.stdout.strip()
# Hugging Face token: into the standard location, never printed
os.makedirs("/root/.cache/huggingface", exist_ok=True)
os.replace("/content/hf_token.tmp", "/root/.cache/huggingface/token")
os.chmod("/root/.cache/huggingface/token", 0o600)
code = "/content/mas-research-project"
if not os.path.isdir(code):
    sh("git", "clone", "--quiet", "/content/repo.bundle", code)
sh("git", "fetch", "--quiet", "/content/repo.bundle", "review-9e45160-fixes", cwd=code)
sh("git", "checkout", "--quiet", "--detach", COMMIT, cwd=code)
head = sh("git", "rev-parse", "HEAD", cwd=code)
assert head == COMMIT, head
assert not sh("git", "status", "--porcelain", cwd=code), "checkout has local changes"
print("code at", sh("git", "log", "-1", "--format=%h %s", cwd=code))
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{code}/requirements-colab.txt"],
                   capture_output=True, text=True)
print("pip exit", r.returncode, r.stderr[-1500:] if r.returncode else "")
import importlib.metadata as md
for p in ("torch", "transformers", "accelerate", "datasets", "numpy", "pandas", "scikit-learn", "pyarrow"):
    try:
        print(f"{p:<14}", md.version(p))
    except Exception as e:
        print(p, "MISSING", e)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                     capture_output=True, text=True).stdout)
print(subprocess.run(["df", "-h", "/content"], capture_output=True, text=True).stdout)
