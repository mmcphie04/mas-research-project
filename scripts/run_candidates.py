"""Candidate evaluation on questions of an existing job (six readouts per question, no generation).

    python scripts/run_candidates.py --config configs/base.yaml configs/colab_12b.yaml \\
        --out-root /content/drive/MyDrive/mas_runs --parent main12b --n-per-split 120 30 150
    # a prompt check first, kept apart from the final output:
    python scripts/run_candidates.py ... --parent main12b --n 10 --check

Questions come from the parent job's verified snapshot, in the parent's order (restricted to the
given splits of a verified split manifest). Output: <out-root>/<parent>-candidates/ with
candidate_states.parquet, the rendered prompts and their token alignment, activations, and a
manifest recording the model identity. A completed run is never overwritten without --force.
It is one short job (six readouts per question, no generation; ~10 min for 300 questions on an
A100), not chunked: if it fails, rerun it. Inspect 5-10 questions first (--n 10) before a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_experiment import _child_questions, parse_overrides  # noqa: E402

from mad.backends.base import build_backend  # noqa: E402
from mad.candidates import run_candidates  # noqa: E402
from mad.config import load_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--n", type=int, help="first N questions of the parent, ignoring splits "
                                          "(only for a parent with no registered split)")
    ap.add_argument("--n-per-split", type=int, nargs=3, metavar=("TRAIN", "VALIDATION", "TEST"),
                    help="explicit sample: the first K questions of each split, in the parent's order")
    ap.add_argument("--split-manifest", help="default: the parent's archived <parent>-split.json")
    ap.add_argument("--check", action="store_true",
                    help="a prompt check on the first --n TRAINING questions (or a pilot job's questions): "
                         "written to <parent>-candidates-check, never the final output")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--force", action="store_true", help="overwrite a completed candidate run")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    cfg = load_config(*args.config, **parse_overrides(args.set))
    cfg.out_dir = str(root)
    parent = json.loads((root / f"{args.parent}-job.json").read_text())
    qs = _child_questions(root, parent, None)
    registered = (parent["identity"].get("split") or {}).get("assignment_sha256")
    split_path = args.split_manifest or (root / f"{args.parent}-split.json" if registered else None)
    assign, split_sha = {}, None
    if split_path:
        from make_split import load_split

        assign, smeta = load_split(split_path, [q.question_id for q in qs], allow_subset=True)
        split_sha = smeta["assignment_sha256"]
        if registered and split_sha != registered:
            raise SystemExit(f"{split_path} is not the split {args.parent} registered ({registered})")
    if args.n_per_split:
        if not assign:
            raise SystemExit("--n-per-split needs a split (the parent's registered one, or --split-manifest)")
        picked = []
        for split, k in zip(("train", "validation", "test"), args.n_per_split, strict=True):
            pool = [q for q in qs if assign[q.question_id] == split]
            if k > len(pool):
                raise SystemExit(f"asked for {k} {split} questions; the parent has {len(pool)}")
            picked += pool[:k]
        qs = picked
    elif args.n:
        if registered and not args.check:
            raise SystemExit(f"{args.parent} has a registered split: sample with --n-per-split")
        if args.check and assign:
            qs = [q for q in qs if assign[q.question_id] == "train"]  # prompt checks never touch test
        qs = qs[: args.n]
    else:
        raise SystemExit("give --n-per-split (or --n for a prompt check / a parent without a split)")
    out_dir = root / (f"{args.parent}-candidates-check" if args.check else f"{args.parent}-candidates")
    if (out_dir / "completion.json").exists() and not args.force:
        raise SystemExit(f"{out_dir} is already complete; refusing to overwrite it (pass --force)")
    backend = None if cfg.model.backend == "stub" else build_backend(cfg)
    t = time.time()
    rows = run_candidates(cfg, qs, out_dir, backend=backend,
                          question_split={q.question_id: assign.get(q.question_id, "unassigned") for q in qs},
                          extra_manifest={"parent_job": args.parent,
                                          "parent_identity_sha256": parent["identity_sha256"],
                                          "split_assignment_sha256": split_sha, "n_per_split": args.n_per_split,
                                          "prompt_check": args.check})
    print(f"{len(qs)} questions, {len(rows)} readouts in {time.time() - t:.0f}s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
