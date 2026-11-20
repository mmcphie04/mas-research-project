"""Runtime: create the SEPARATE vLLM environment (vLLM pins its own torch; the HF stack keeps the
system torch for readouts and activations). Run after setup.py (code + token in place).

    python artifacts/expab/ops/setup_vllm.py      # prints and writes /content/vllm_env_versions.json

Pinned: vllm 0.33.0 (released 2026-11-02; torch 2.14.0, transformers >=5.16.1,<5.21); falls back to
0.32.0 (torch 2.13.0) if the first does not import with CUDA. Never touches the system environment.
"""
import json
import os
import shutil
import subprocess
import sys

ENV = "/content/vllm_env"
PY = f"{ENV}/bin/python"
VERSIONS = ["0.33.0", "0.32.0"]


def sh(*a, timeout=1800):
    return subprocess.run(list(a), capture_output=True, text=True, timeout=timeout)


# uv creates the environment and installs into it (Colab's Python has no ensurepip, so `python -m
# venv` cannot bootstrap pip there)
r = sh(sys.executable, "-m", "pip", "install", "-q", "uv")
UV = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
if not os.path.exists(PY):
    r = sh(UV, "venv", ENV, "--python", sys.executable)
    if r.returncode:
        raise SystemExit(f"venv failed: {(r.stderr or r.stdout)[-1000:]}")
probe = ("import json, torch, vllm, transformers, pandas, pyarrow; print(json.dumps({'vllm': vllm.__version__, "
         "'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), "
         "'transformers': transformers.__version__, 'pandas': pandas.__version__}))")
info = None
for v in VERSIONS:
    r = sh(UV, "pip", "install", "--python", PY, "-q", f"vllm=={v}", "pandas", "pyarrow")
    if r.returncode:
        print(f"vllm {v}: install failed: {r.stderr[-800:]}")
        continue
    p = sh(PY, "-c", probe, timeout=600)
    if p.returncode == 0 and '"cuda_available": true' in p.stdout:
        info = json.loads(p.stdout.strip().splitlines()[-1])
        break
    print(f"vllm {v}: import/CUDA check failed: {(p.stderr or p.stdout)[-800:]}")
if info is None:
    raise SystemExit("no pinned vLLM version imports with CUDA; nothing launched")
sysinfo = sh(sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)").stdout.strip()
info["system_env_torch_transformers"] = sysinfo
json.dump(info, open("/content/vllm_env_versions.json", "w"), indent=1)
print(json.dumps(info))
