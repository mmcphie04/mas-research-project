"""Freeze a train / validation / test split of the main questions BEFORE any of them is generated.

    # before the main run: from the same deterministic selection the driver will make
    python scripts/make_split.py --dataset mmlu-pro --pool main --n-questions 900 \\
        --fracs 0.375 0.125 0.5 --seed 20261009 --out ROOT/main12b-split.json
    # then: run_experiment.py ... --name main12b --split-manifest ROOT/main12b-split.json

    # (or, for an existing job: --out-root ROOT --name JOB instead of the selection flags)

The split is by question: every agent, condition, child job and permutation of a question lands
in the same split. Stratified by subject. The manifest records the selection (or job) it was
made for, the ordered question ids, and a hash of the assignment. The driver refuses to start a
job whose questions differ from the manifest's and records the assignment hash in the job's
identity; reanalysis refuses a manifest whose hash, names, partitions or membership are wrong
(``load_split``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mad.data import Question, make_splits  # noqa: E402

SPLITS = ("train", "validation", "test")


def assignment_sha256(assignment: dict) -> str:
    return hashlib.sha256(json.dumps(assignment, sort_keys=True).encode()).hexdigest()


def load_split(path, question_ids=None, allow_subset: bool = True) -> tuple[dict, dict]:
    """Read a split manifest and refuse it unless it is intact: the assignment matches its
    recorded hash, every name is train/validation/test, and -- when ``question_ids`` are given --
    every one of them is assigned (exactly the manifest's set unless ``allow_subset``) and each
    partition is non-empty among them. Returns (assignment, meta)."""
    data = json.loads(Path(path).read_text())
    assignment, meta = data["assignment"], data["meta"]
    if assignment_sha256(assignment) != meta.get("assignment_sha256"):
        raise SystemExit(f"{path}: the assignment does not match its recorded hash")
    bad = sorted({v for v in assignment.values()} - set(SPLITS))
    if bad:
        raise SystemExit(f"{path}: invalid split names {bad}")
    if question_ids is not None:
        ids = set(question_ids)
        unmapped = sorted(ids - set(assignment))
        if unmapped:
            raise SystemExit(f"{path}: {len(unmapped)} questions are not in the split, e.g. {unmapped[:3]}")
        if not allow_subset and ids != set(assignment):
            raise SystemExit(f"{path}: the split covers {len(assignment)} questions, these are {len(ids)}")
        empty = [s for s in SPLITS if not any(assignment[q] == s for q in ids)]
        if empty:
            raise SystemExit(f"{path}: empty partitions among these questions: {empty}")
    return assignment, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", help="existing job mode: where the job lives")
    ap.add_argument("--name", help="existing job mode: the job name")
    ap.add_argument("--dataset", help="selection mode (before generation): mmlu-pro | arc-challenge | ...")
    ap.add_argument("--pool", choices=["pilot", "validation", "main"], default="main")
    ap.add_argument("--validation-size", type=int, default=0,
                    help="questions reserved after the pilot pool for validation (as passed to the driver)")
    ap.add_argument("--pilot-size", type=int, default=120)
    ap.add_argument("--n-questions", type=int)
    ap.add_argument("--order-seed", type=int, default=7, help="data.sample_seed (master-order seed)")
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.375, 0.125, 0.5])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.name:
        root = Path(args.out_root)
        job = json.loads((root / f"{args.name}-job.json").read_text())
        snap = [json.loads(x) for x in (root / f"{args.name}-questions.jsonl").read_text().splitlines()]
        qs = [Question(**{**d, "options": tuple(d["options"])}) for d in snap]
        if [q.question_id for q in qs] != [q["id"] for q in job["questions"]]:
            raise SystemExit("question snapshot does not match the job file")
        made_for = {"job": args.name, "job_identity_sha256": job["identity_sha256"]}
    else:
        if not (args.dataset and args.n_questions):
            raise SystemExit("give --dataset and --n-questions (before generation) or --out-root and --name")
        from types import SimpleNamespace

        from run_experiment import DATASETS, load_pool

        from mad.config import DataConfig
        from mad.job import master_order, select

        data = DataConfig(sample_seed=args.order_seed)
        data.dataset, data.split = DATASETS[args.dataset]
        pool = load_pool(args.dataset, SimpleNamespace(data=data))
        qs = select(master_order(pool, args.order_seed), args.pool, args.n_questions, args.pilot_size,
                    args.validation_size)
        made_for = {"selection": {"dataset": args.dataset, "pool": args.pool, "pilot_size": args.pilot_size,
                                  "n_questions": args.n_questions, "order_seed": args.order_seed,
                                  **({"validation_size": args.validation_size} if args.validation_size else {})}}
    raw = make_splits(qs, tuple(args.fracs), seed=args.seed)
    assignment = {q: ("validation" if s == "val" else s) for q, s in raw.items()}
    counts = {s: sum(v == s for v in assignment.values()) for s in SPLITS}
    ids = [q.question_id for q in qs]
    out = {"meta": {**made_for, "fracs": args.fracs, "seed": args.seed, "stratified_by": "subject",
                    "counts": counts, "question_ids": ids,
                    "question_ids_sha256": hashlib.sha256(",".join(ids).encode()).hexdigest(),
                    "assignment_sha256": assignment_sha256(assignment)},
           "assignment": assignment}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
