"""Held-out EXTENSION for Experiments A and B, committed before any generation on it (questions
only; no model outcome is read).

    python scripts/expab_questions_ext.py --out artifacts/expab/plan_ext

Rule: the same pool and master order as ``expab_questions.py`` (MMLU-Pro test at revision
b189ec76..., order seed 7, master sha256 c4c67294...); positions above 520 that are NOT in the
original 1,500 (artifacts/expab/plan/manifest.json); 2,000 questions by
``mad.data.stratified_sample`` (as-equal-as-possible per subject), seed 20261124; every one
assigned to HELD-OUT. Development stays the original 600.

Writes questions.jsonl and manifest.json (ids, content hashes, assignment, hashes; and the
original manifest's hashes it extends).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from expab_questions import MASTER_SHA, REVISION, USED  # noqa: E402


def main(argv=None) -> int:
    from run_experiment import DATASETS, load_pool

    from mad.config import DataConfig
    from mad.data import stratified_sample
    from mad.job import master_order, question_sha256

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--original", default=str(ROOT / "artifacts/expab/plan/manifest.json"))
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261124)
    a = ap.parse_args(argv)
    orig = json.loads(Path(a.original).read_text())
    data = DataConfig(sample_seed=7, dataset_revision=REVISION)
    data.dataset, data.split = DATASETS["mmlu-pro"]
    pool = load_pool("mmlu-pro", SimpleNamespace(data=data))
    master = master_order(pool, 7)
    sha = hashlib.sha256(",".join(q.question_id for q in master).encode()).hexdigest()
    if sha != MASTER_SHA:
        raise SystemExit(f"master order {sha} != {MASTER_SHA}")
    used = set(orig["question_ids"])
    fresh = [q for q in master[USED:] if q.question_id not in used]
    chosen = stratified_sample(fresh, a.n, seed=a.seed)
    pos = {q.question_id: i + 1 for i, q in enumerate(master)}
    ids = [q.question_id for q in chosen]
    assert len(set(ids)) == a.n and not set(ids) & used and all(pos[i] > USED for i in ids)
    assignment = {i: "heldout" for i in ids}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = out / "questions.jsonl"
    snap.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in chosen))
    m = {"rule": __doc__.split("Rule: ")[1].split("\n\nWrites")[0].replace("\n", " "),
         "extends": {"question_ids_sha256": orig["question_ids_sha256"], "assignment_sha256": orig["assignment_sha256"],
                     "snapshot_sha256": orig["snapshot_sha256"]},
         "dataset_revision": REVISION, "master_order_sha256": sha, "sample_seed": a.seed, "n": a.n,
         "positions_1_based_range": [min(pos[i] for i in ids), max(pos[i] for i in ids)],
         "subjects": dict(sorted(Counter(q.subject for q in chosen).items())), "counts": {"heldout": a.n},
         "question_ids": ids, "question_ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
         "question_content_sha256": {q.question_id: question_sha256(q) for q in chosen},
         "assignment": assignment,
         "assignment_sha256": hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest(),
         "snapshot_sha256": hashlib.sha256(snap.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps({k: m[k] for k in ("positions_1_based_range", "subjects", "question_ids_sha256", "snapshot_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
