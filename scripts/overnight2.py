"""Second overnight run: corrected Milestone A, then a four-condition debate experiment.

    nohup uv run python scripts/overnight2.py > logs/overnight2.log 2>&1 &

Chunked and resumable: each chunk writes completion records for every condition and is skipped
on re-run, so an interrupted night yields complete four-condition data for however many
questions finished.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from mad.collect import STATES_FILE, collect_states
from mad.config import load_config
from mad.data import load_mc_dataset, stratified_sample
from mad.debate import ROWS_FILE
from mad.multi import run_conditions
from mad.provenance import is_complete
from mad.store import merge_runs

BASE = ["configs/base.yaml", "configs/probe_arc.yaml"]
CONDITIONS = [
    {"name": "self_reconsider", "kind": "self_reconsider"},
    {"name": "debate", "kind": "debate", "peer_confidence": False},
    {"name": "debate_conf", "kind": "debate_conf", "peer_confidence": True},
    {"name": "adversarial", "kind": "adversarial"},
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def questions(dataset, n, seed=7):
    return stratified_sample(load_mc_dataset("allenai/ai2_arc", dataset, "test"), n, seed)


def job_probe_rerun():
    """Milestone A again with the double readout suffix removed."""
    out = Path("runs/probe-arcc-fixed")
    if is_complete(out):
        log("SKIP probe rerun"); return
    qs = questions("ARC-Challenge", 400)
    dirs = []
    for i in range(0, len(qs), 200):
        rid = f"probe-arcc-fixed-c{i // 200:03d}"
        d = Path("runs") / rid
        dirs.append(d)
        if is_complete(d):
            continue
        cfg = load_config(*BASE, **{"model.batch_size": 16})
        cfg.run_id = rid
        t = time.time()
        collect_states(cfg, qs[i : i + 200], n_variants=3)
        log(f"  {rid}: 200 questions in {time.time() - t:.0f}s")
    merge_runs(dirs, out, STATES_FILE)
    log(f"DONE probe rerun -> {out}")


def job_four_condition(n_questions=350, chunk=25):
    qs = questions("ARC-Challenge", n_questions)
    parts = [qs[i : i + chunk] for i in range(0, len(qs), chunk)]
    per_cond = {c["name"]: [] for c in CONDITIONS}

    for i, part in enumerate(parts):
        tag = f"c{i:03d}"
        dirs = {c["name"]: Path("runs") / f"night2-{c['name']}-{tag}" for c in CONDITIONS}
        for n, d in dirs.items():
            per_cond[n].append(d)
        if all(is_complete(d) for d in dirs.values()):
            continue
        cfg = load_config(*BASE, **{"model.max_new_tokens": 64, "model.batch_size": 16})
        conds = [{**c, "run_id": f"night2-{c['name']}-{tag}"} for c in CONDITIONS]
        t = time.time()
        run_conditions(cfg, part, conds)
        log(f"  {tag}: {len(part)} questions x {len(CONDITIONS)} conditions in {time.time() - t:.0f}s")

    for name, dirs in per_cond.items():
        out = Path("runs") / f"night2-{name}"
        done = [d for d in dirs if is_complete(d)]
        if done and not is_complete(out):
            merge_runs(done, out, ROWS_FILE)
            log(f"DONE {name}: merged {len(done)}/{len(dirs)} chunks -> {out}")


if __name__ == "__main__":
    for label, fn in (("probe rerun (suffix fix)", job_probe_rerun),
                      ("four-condition debate", job_four_condition)):
        log(f"=== {label} ===")
        try:
            fn()
        except Exception:
            log(f"FAILED {label}\n{traceback.format_exc()}")
    log("finished")
