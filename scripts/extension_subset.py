"""Prespecified subset for the main collection's extensions E1 (second round) and E2 (structured
reconsideration without peers): an OUTCOME-BLIND, seeded rule, committed before any accuracy of
the main collection is computed.

    python scripts/extension_subset.py --job runs/main250/main250-job.json --seed 20261030 \\
        --per-chunk 2 --out artifacts/main250/plan/extension_subset.json

Rule: within each chunk of the job (in chunk order), ``per_chunk`` questions drawn without
replacement by numpy's default_rng(seed). Reads ONLY the job record's chunk membership (question
ids); no answers, readouts or correctness are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def select(chunks: list[list[str]], seed: int, per_chunk: int) -> list[str]:
    rng = np.random.default_rng(seed)
    out = []
    for members in chunks:
        if len(members) < per_chunk:
            raise SystemExit(f"a chunk has {len(members)} questions < {per_chunk}")
        idx = sorted(rng.choice(len(members), size=per_chunk, replace=False).tolist())
        out += [members[i] for i in idx]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per-chunk", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    job = json.loads(Path(a.job).read_text())
    chunks = [m for m in job["chunks"] if m]
    ids = select(chunks, a.seed, a.per_chunk)
    membership = hashlib.sha256(json.dumps(chunks).encode()).hexdigest()
    rec = {"question_ids": ids,
           "question_ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
           "rule": f"{a.per_chunk} per chunk, drawn without replacement within each chunk in chunk order by "
                   f"numpy.random.default_rng({a.seed}); outcome-blind (reads only the job's chunk membership)",
           "seed": a.seed, "per_chunk": a.per_chunk, "n_chunks": len(chunks),
           "job": job["identity"]["name"], "job_identity_sha256": job["identity_sha256"],
           "chunk_membership_sha256": membership,
           "uses": "E1 (second round, *_r2 conditions) and E2 (self_structured5); both run regardless of the "
                   "primary results (docs/NEXT_PHASE_PLAN.md)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("question_ids_sha256", "seed", "per_chunk", "n_chunks")}), len(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
