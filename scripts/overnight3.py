"""Third run: complete responses, a non-saturating misleading peer, and the pre-revision state.

    nohup uv run python scripts/overnight3.py > logs/overnight3.log 2>&1 &

Changes from the previous run, each for a reason found in the data:
  * 192 generation tokens, because at 64 the model was cut off before stating an answer and the
    agents were exchanging incomplete reasoning. Cap-hit and parse rates are logged per chunk so
    the budget can be judged empirically rather than assumed.
  * misleading_one replaces the adversarial arm, which hit 99% adoption and measured only whether
    a small model copies a confident assertion.
  * a peer_exposure readout after the agent sees the intervention but BEFORE it revises.
  * round 0 generated once and shared, so every condition starts from identical initial answers.
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import pandas as pd

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
    {"name": "misleading_one", "kind": "misleading_one"},
]
N_QUESTIONS, CHUNK, MAX_TOK = 160, 20, 192


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def diagnostics(run_dir: Path, tag: str) -> None:
    """Completion-length and parse-rate diagnostics -- the basis for the token budget."""
    r = pd.read_parquet(run_dir / ROWS_FILE)
    for phase in ("pre", "post"):
        s = r[r.phase == phase]
        if not len(s):
            continue
        cap = (s.n_completion_tokens >= MAX_TOK - 2).mean()
        log(f"    {tag} {phase}: parse-ok {s.freetext_parse_ok.mean():.3f}  "
            f"hit-cap {cap:.3f}  median tokens {s.n_completion_tokens.median():.0f}")


def main() -> int:
    qs = stratified_sample(load_mc_dataset("allenai/ai2_arc", "ARC-Challenge", "test"),
                           N_QUESTIONS, 7)
    parts = [qs[i : i + CHUNK] for i in range(0, len(qs), CHUNK)]
    per_cond = {c["name"]: [] for c in CONDITIONS}

    for i, part in enumerate(parts):
        tag = f"c{i:03d}"
        dirs = {c["name"]: Path("runs") / f"night3-{c['name']}-{tag}" for c in CONDITIONS}
        for n, d in dirs.items():
            per_cond[n].append(d)
        if all(is_complete(d) for d in dirs.values()):
            continue
        cfg = load_config(*BASE, **{"model.max_new_tokens": MAX_TOK, "model.batch_size": 16})
        conds = [{**c, "run_id": f"night3-{c['name']}-{tag}"} for c in CONDITIONS]
        t = time.time()
        try:
            run_conditions(cfg, part, conds)
        except Exception:
            log(f"FAILED {tag}\n{traceback.format_exc()}")
            continue
        log(f"  {tag}: {len(part)} questions x {len(CONDITIONS)} conditions in {time.time() - t:.0f}s")
        if i == 0:
            diagnostics(dirs["debate"], tag)

    for name, dirs in per_cond.items():
        out = Path("runs") / f"night3-{name}"
        done = [d for d in dirs if is_complete(d)]
        if done and not is_complete(out):
            merge_runs(done, out, ROWS_FILE)
            log(f"DONE {name}: {len(done)}/{len(dirs)} chunks -> {out}")
            diagnostics(out, name)
    return 0


if __name__ == "__main__":
    log(f"=== night3: {N_QUESTIONS} questions x {len(CONDITIONS)} conditions, {MAX_TOK} tokens ===")
    raise SystemExit(main())
