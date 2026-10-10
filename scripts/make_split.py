"""Freeze a train / validation / test split of a job's questions, before any outcome is seen.

    python scripts/make_split.py --out-root ROOT --name main12b --fracs 0.375 0.125 0.5 \\
        --seed 20261009 --out ROOT/main12b-split.json

The split is by question: every agent, condition, child job and permutation of a question lands
in the same split. Stratified by subject. The manifest records the job identity hash, so it
cannot be silently applied to a different job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.data import Question, make_splits  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--fracs", type=float, nargs=3, default=[0.375, 0.125, 0.5])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    job = json.loads((root / f"{args.name}-job.json").read_text())
    snap = [json.loads(x) for x in (root / f"{args.name}-questions.jsonl").read_text().splitlines()]
    qs = [Question(**{**d, "options": tuple(d["options"])}) for d in snap]
    if [q.question_id for q in qs] != [q["id"] for q in job["questions"]]:
        raise SystemExit("question snapshot does not match the job file")
    raw = make_splits(qs, tuple(args.fracs), seed=args.seed)
    assignment = {q: ("validation" if s == "val" else s) for q, s in raw.items()}
    counts = {s: sum(v == s for v in assignment.values()) for s in ("train", "validation", "test")}
    blob = json.dumps(assignment, sort_keys=True).encode()
    out = {"meta": {"job": args.name, "job_identity_sha256": job["identity_sha256"],
                    "fracs": args.fracs, "seed": args.seed, "stratified_by": "subject",
                    "counts": counts, "assignment_sha256": hashlib.sha256(blob).hexdigest()},
           "assignment": assignment}
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(f"wrote {args.out}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
