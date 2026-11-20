"""Chunked, hash-recorded outputs for generation jobs (Experiments A/B, judge confirmation).

Layout under ``out_root`` (compatible with ``mad.cli_backup`` and the per-chunk supervisor):

    <job>-job.json                 identity: task, modes, question-set hashes, expected chunks, code
    <job>-status.json              progress (chunks done / total, seconds)
    <job>-<part>-cNNN/             one chunk: rows.parquet, calls.json, manifest.json, completion.json

A chunk directory appears only complete: it is written under a ``.tmp-`` name and renamed. Its
``completion.json`` lists every file's sha256 (``shard_sha256``), which the laptop backup checks
after transfer (``mad.job._check_hashes``) and ``verify_job`` checks again.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mad.job import JobMismatch, _check_hashes, chunk_tag
from mad.provenance import is_complete


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def chunk_dir(out_root, job: str, part: str, idx: int) -> Path:
    return Path(out_root) / f"{job}-{part}-{chunk_tag(idx)}"


def write_chunk(out_root, job: str, part: str, idx: int, rows: list[dict], calls: list[dict], meta: dict,
                arrays: dict | None = None) -> Path:
    final = chunk_dir(out_root, job, part, idx)
    tmp = final.with_name(final.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(tmp / "rows.parquet", index=False)
    (tmp / "calls.json").write_text(json.dumps(calls, indent=1))
    for name, arr in (arrays or {}).items():  # e.g. activations, row-aligned with rows.parquet
        np.save(tmp / f"{name}.npy", arr)
    (tmp / "manifest.json").write_text(json.dumps({"job": job, "part": part, "chunk": chunk_tag(idx), **meta}, indent=1))
    files = {f.name: _sha(f) for f in sorted(tmp.iterdir()) if f.is_file()}
    (tmp / "completion.json").write_text(json.dumps({
        "shard_sha256": files, "row_counts": {"rows": len(rows), "calls": len(calls)},
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=1))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    return final


def write_job(out_root, job: str, record: dict) -> Path:
    p = Path(out_root) / f"{job}-job.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    os.replace(tmp, p)
    return p


def write_status(out_root, job: str, status: dict) -> None:
    p = Path(out_root) / f"{job}-status.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, p)


def expected_chunks(record: dict) -> list[str]:
    return [f"{record['job']}-{part}-{chunk_tag(i)}" for part, n in record["parts"].items() for i in range(n)]


def incomplete(dest, job: str) -> dict:
    """Expected chunk directories (from the job record) that are not complete in ``dest``."""
    p = Path(dest) / f"{job}-job.json"
    if not p.exists():
        return {"job_record": "missing"}
    rec = json.loads(p.read_text())
    missing = [n for n in expected_chunks(rec) if not is_complete(Path(dest) / n)]
    return {"missing_chunks": missing} if missing else {}


def verify_job(dest, job: str) -> dict:
    """Every complete chunk's files against its completion record; raises JobMismatch on any
    difference. Returns {"complete": n, "missing": [...]}."""
    dest = Path(dest)
    rec = json.loads((dest / f"{job}-job.json").read_text())
    done, missing = [], []
    for name in expected_chunks(rec):
        d = dest / name
        if is_complete(d):
            _check_hashes(d)
            done.append(name)
        else:
            missing.append(name)
    return {"complete": len(done), "missing": missing, "views": [job]}


def read_rows(dest, job: str, part: str | None = None) -> pd.DataFrame:
    dest = Path(dest)
    frames = []
    for d in sorted(dest.glob(f"{job}-{part or '*'}-c*")):
        if d.is_dir() and is_complete(d) and ".tmp-" not in d.name:
            frames.append(pd.read_parquet(d / "rows.parquet"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_calls(dest, job: str, part: str | None = None) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(dest).glob(f"{job}-{part or '*'}-c*")):
        if d.is_dir() and is_complete(d) and ".tmp-" not in d.name:
            m = json.loads((d / "manifest.json").read_text())
            for c in json.loads((d / "calls.json").read_text()):
                rows.append({**c, "part": m["part"], "chunk": m["chunk"]})
    return pd.DataFrame(rows)


__all__ = ["JobMismatch", "write_chunk", "write_job", "write_status", "incomplete", "verify_job", "read_rows",
           "read_calls", "chunk_dir", "expected_chunks"]
