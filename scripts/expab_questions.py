"""Question set for Experiments A and B, committed BEFORE any generation (questions only; no model
outcome is read).

    python scripts/expab_questions.py --out artifacts/expab/plan

Rule:
* the MMLU-Pro test split at the pinned revision, in the project's master order (order seed 7;
  its sha256 must equal main250's: ``c4c67294...``);
* only positions above 520 (1-based 521..), never used by any earlier job (pilot 1-120,
  validation 121-320, main250 321-520);
* 1,500 questions by ``mad.data.stratified_sample`` (as-equal-as-possible per subject, as every
  earlier set), seed 20261120;
* split 600 development / 900 held-out, stratified by subject: development counts per subject by
  largest remainder (600 x subject share), the questions within a subject shuffled by
  ``default_rng(20261121)``.

Writes questions.jsonl (the question snapshot the runtime reads: id, subject, question, options,
gold index) and manifest.json (dataset revision, master-order hash, positions, rule, seeds, ordered
ids and their sha256, per-question content sha256, the split assignment and its sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MASTER_SHA = "c4c67294e44a85e4de0be2cab8d26c630e0859b8b37e434c65c960f4fa471afd"
USED = 520


def split_exact(qs, n_dev: int, seed: int) -> dict[str, str]:
    by_s = defaultdict(list)
    for q in qs:
        by_s[q.subject].append(q.question_id)
    subjects = sorted(by_s)
    share = {s: n_dev * len(by_s[s]) / len(qs) for s in subjects}
    alloc = {s: int(np.floor(share[s])) for s in subjects}
    for s in sorted(subjects, key=lambda s: (-(share[s] - alloc[s]), s))[: n_dev - sum(alloc.values())]:
        alloc[s] += 1
    rng = np.random.default_rng(seed)
    out = {}
    for s in subjects:
        ids = sorted(by_s[s])
        rng.shuffle(ids)
        for i, q in enumerate(ids):
            out[q] = "dev" if i < alloc[s] else "heldout"
    return out


def main(argv=None) -> int:
    from run_experiment import DATASETS, load_pool

    from mad.config import DataConfig
    from mad.data import stratified_sample
    from mad.job import master_order, question_sha256

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-dev", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--split-seed", type=int, default=20261121)
    a = ap.parse_args(argv)
    data = DataConfig(sample_seed=7, dataset_revision=REVISION)
    data.dataset, data.split = DATASETS["mmlu-pro"]
    pool = load_pool("mmlu-pro", SimpleNamespace(data=data))
    master = master_order(pool, 7)
    sha = hashlib.sha256(",".join(q.question_id for q in master).encode()).hexdigest()
    if sha != MASTER_SHA:
        raise SystemExit(f"master order {sha} != main250's {MASTER_SHA}")
    fresh = master[USED:]
    chosen = stratified_sample(fresh, a.n, seed=a.seed)
    pos = {q.question_id: i + 1 for i, q in enumerate(master)}
    assert all(pos[q.question_id] > USED for q in chosen) and len({q.question_id for q in chosen}) == a.n
    assignment = split_exact(chosen, a.n_dev, a.split_seed)
    ids = [q.question_id for q in chosen]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = out / "questions.jsonl"
    snap.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in chosen))
    manifest = {
        "rule": __doc__.split("Rule:\n")[1].split("\n\nWrites")[0],
        "dataset": "TIGER-Lab/MMLU-Pro", "split": "test", "dataset_revision": REVISION,
        "master_order_sha256": sha, "order_seed": 7, "positions_used_before": [1, USED],
        "positions_1_based_range": [min(pos[q] for q in ids), max(pos[q] for q in ids)],
        "sample_seed": a.seed, "split_seed": a.split_seed, "n": a.n,
        "counts": dict(Counter(assignment.values())),
        "subjects": {s: {"n": n, "dev": sum(1 for q in chosen if q.subject == s and assignment[q.question_id] == "dev")}
                     for s, n in sorted(Counter(q.subject for q in chosen).items())},
        "question_ids": ids, "question_ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
        "question_content_sha256": {q.question_id: question_sha256(q) for q in chosen},
        "assignment": assignment,
        "assignment_sha256": hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest(),
        "snapshot_sha256": hashlib.sha256(snap.read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "positions_1_based_range", "question_ids_sha256",
                                               "assignment_sha256", "snapshot_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
