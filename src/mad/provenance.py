"""Run provenance: manifest written before a run, completion written after.

The before/after split is deliberate — a run directory with a manifest and no completion record
is a crashed run, and must never be silently analysed as if it were complete.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mad.config import RunConfig, config_hash

MANIFEST = "manifest.json"
COMPLETION = "completion.json"


def _run(cmd: list[str]) -> str | None:
    try:
        return (
            subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
        )
    except Exception:
        return None


def git_state() -> dict[str, Any]:
    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_dirty": bool(status),
        "git_dirty_files": (status or "").splitlines(),
    }


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "transformers",
        "datasets",
        "vllm",
    ):
        try:
            from importlib.metadata import version

            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class Manifest:
    payload: dict[str, Any]
    path: Path

    @property
    def run_dir(self) -> Path:
        return self.path.parent


def write_manifest(cfg: RunConfig, extra: dict[str, Any] | None = None) -> Manifest:
    """Write manifest.json BEFORE the run starts."""
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": cfg.run_id,
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "packages": package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        **git_state(),
    }
    if extra:
        payload.update(extra)
    path = run_dir / MANIFEST
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return Manifest(payload=payload, path=path)


def write_completion(
    run_dir: Path,
    row_counts: dict[str, int],
    shard_hashes: dict[str, str] | None = None,
    errors: list[str] | None = None,
) -> Path:
    """Write completion.json AFTER the run finishes."""
    payload = {
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "row_counts": row_counts,
        "shard_sha256": shard_hashes or {},
        "errors": errors or [],
    }
    path = Path(run_dir) / COMPLETION
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def is_complete(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / MANIFEST).exists() and (run_dir / COMPLETION).exists()


def require_complete(run_dir: str | Path) -> dict[str, Any]:
    """Load a run's manifest, refusing to proceed if the run crashed mid-flight."""
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST).exists():
        raise FileNotFoundError(f"no manifest in {run_dir}; this is not a run directory")
    if not (run_dir / COMPLETION).exists():
        raise RuntimeError(
            f"{run_dir} has a manifest but no completion record: the run crashed. "
            "Re-run it or explicitly analyse it with is_complete() checks disabled."
        )
    return json.loads((run_dir / MANIFEST).read_text())


def prompt_hashes(templates: dict[str, str]) -> dict[str, str]:
    return {k: config_hash(v)[:16] for k, v in sorted(templates.items())}
