"""Candidate evaluation on questions of an existing job (six readouts per question, no generation).

    python scripts/run_candidates.py --config configs/base.yaml configs/colab_12b.yaml \\
        --out-root /content/drive/MyDrive/mas_runs --parent main12b --n 300 \\
        [--split-manifest main12b-split.json --splits train test]

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
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--split-manifest")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--force", action="store_true", help="overwrite a completed candidate run")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    cfg = load_config(*args.config, **parse_overrides(args.set))
    cfg.out_dir = str(root)
    parent = json.loads((root / f"{args.parent}-job.json").read_text())
    qs = _child_questions(root, parent, None)
    if args.split_manifest:
        from make_split import load_split

        assign, _ = load_split(args.split_manifest, [q.question_id for q in qs], allow_subset=True)
        qs = [q for q in qs if assign[q.question_id] in set(args.splits or ["test"])]
    out_dir = root / f"{args.parent}-candidates"
    if (out_dir / "completion.json").exists() and not args.force:
        raise SystemExit(f"{out_dir} is already complete; refusing to overwrite it (pass --force)")
    qs = qs[: args.n]
    backend = None if cfg.model.backend == "stub" else build_backend(cfg)
    t = time.time()
    rows = run_candidates(cfg, qs, out_dir, backend=backend,
                          extra_manifest={"parent_job": args.parent,
                                          "parent_identity_sha256": parent["identity_sha256"],
                                          "splits": args.splits})
    print(f"{len(qs)} questions, {len(rows)} readouts in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
