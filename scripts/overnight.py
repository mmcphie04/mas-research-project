"""Resumable overnight driver.

    nohup uv run python scripts/overnight.py > logs/overnight.log 2>&1 &

Work is split into chunks; each chunk writes its own completion record and is skipped on a
re-run. A crash at hour five costs one chunk, not the night. Nothing here needs Claude Code
running -- it is a plain CLI job, so it survives the terminal, the session, and WSL restarts
provided the VM itself stays up.

Jobs are ordered by value, so an interrupted night still yields the most important results.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from mad.collect import STATES_FILE, collect_states
from mad.config import load_config
from mad.data import load_mc_dataset, stratified_sample
from mad.debate import ROWS_FILE, run_debate
from mad.provenance import is_complete
from mad.store import merge_runs

BASE = ["configs/base.yaml", "configs/probe_arc.yaml"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def questions_for(dataset: str, n: int, seed: int = 7):
    pool = load_mc_dataset("allenai/ai2_arc", dataset, "test")
    return stratified_sample(pool, n, seed)


def run_chunked(job: dict) -> None:
    """Run one job as a sequence of resumable chunks, then merge."""
    name, kind = job["run_id"], job["kind"]
    out = Path("runs") / name
    if is_complete(out):
        log(f"SKIP {name} (already merged)")
        return

    qs = questions_for(job["dataset"], job["n"])
    chunk = job.get("chunk", 25)
    parts = [qs[i : i + chunk] for i in range(0, len(qs), chunk)]
    dirs = []
    for i, part in enumerate(parts):
        rid = f"{name}-c{i:03d}"
        d = Path("runs") / rid
        dirs.append(d)
        if is_complete(d):
            continue
        cfg = load_config(*BASE, **job.get("overrides", {}))
        cfg.run_id = rid
        t0 = time.time()
        if kind == "debate":
            cfg.debate.condition = job.get("condition", "debate")
            run_debate(cfg, part)
        else:
            collect_states(cfg, part, n_variants=job.get("variants", 3))
        log(f"  {rid}: {len(part)} questions in {time.time() - t0:.0f}s")

    rows_file = ROWS_FILE if kind == "debate" else STATES_FILE
    merge_runs(dirs, out, rows_file)
    log(f"DONE {name}: merged {len(dirs)} chunks -> {out}")


JOBS = [
    # 1. the real experiment in miniature: does debate help or hurt, and are there enough
    #    beneficial/harmful revisions to learn from? This answers the regime question with
    #    real data instead of simulation, so it runs first.
    {
        "kind": "debate", "run_id": "night-debate", "dataset": "ARC-Challenge", "n": 400,
        "chunk": 25, "condition": "debate",
        "overrides": {"model.max_new_tokens": 64, "model.batch_size": 16},
    },
    # 2. the control that separates peer influence from ordinary answer instability
    {
        "kind": "debate", "run_id": "night-control", "dataset": "ARC-Challenge", "n": 400,
        "chunk": 25, "condition": "self_reconsider",
        "overrides": {"model.max_new_tokens": 64, "model.batch_size": 16},
    },
    # 3. cross-dataset transfer: train the probe on ARC-Challenge, test on ARC-Easy. This is
    #    the best-documented failure mode of probing claims and nobody has told us the answer.
    {
        "kind": "readout", "run_id": "night-arce", "dataset": "ARC-Easy", "n": 400,
        "chunk": 200, "variants": 3, "overrides": {"model.batch_size": 16},
    },
    # 4. does the probe's advantage over logit confidence grow with model scale? The
    #    extrapolation the 12B run depends on.
    {
        "kind": "readout", "run_id": "night-270m", "dataset": "ARC-Challenge", "n": 400,
        "chunk": 200, "variants": 3,
        "overrides": {"model.repo_id": "google/gemma-3-270m-it", "model.batch_size": 32},
    },
]


def main(only: str | None = None) -> int:
    failures = []
    for job in JOBS:
        if only and job["run_id"] != only:
            continue
        log(f"=== {job['run_id']} ===")
        try:
            run_chunked(job)
        except Exception:
            failures.append(job["run_id"])
            log(f"FAILED {job['run_id']}\n{traceback.format_exc()}")
    log(f"finished; failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
