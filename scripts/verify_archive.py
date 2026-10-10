"""Independently verify a job's archive: every byte, every key, every value.

    python scripts/verify_archive.py --out-root /content/drive/MyDrive/mas_runs --name pilot12b
    python scripts/verify_archive.py --out-root ARCHIVE --name pilot12b --copy-to /tmp/restore

Run it after a session and before deleting anything from the runtime. ``--copy-to`` also copies
the whole job (job file, question snapshot, round-0 caches, chunks, views) to another location,
verifies the copy there, and checks that its activations load identically -- the recovery test
for an archive that will outlive the runtime. Views reference chunk files: archive the chunks
and round-0 caches, never only the views. Writes ``<name>-verify.json``; exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mad.job import JobMismatch, verify_archive, view_dirs  # noqa: E402
from mad.store import ActivationStore  # noqa: E402


def copy_job(src: Path, dst: Path, name: str) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob(f"{name}-*"):
        if ".tmp-" in p.name:
            continue
        target = dst / p.name
        if p.is_dir():
            shutil.copytree(p, target, dirs_exist_ok=True)
        else:
            shutil.copy2(p, target)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--copy-to", help="copy the job here, verify the copy, compare activations")
    args = ap.parse_args(argv)
    root = Path(args.out_root)
    try:
        report = verify_archive(root, args.name)
        if args.copy_to:
            dst = Path(args.copy_to)
            copy_job(root, dst, args.name)
            report["copy"] = verify_archive(dst, args.name)
            for cond, d in view_dirs(root, args.name).items():
                a, b = ActivationStore(d), ActivationStore(dst / d.name)
                for layer in (0, a.n_layers):
                    if not np.array_equal(a.load_layer(layer), b.load_layer(layer)):
                        raise JobMismatch(f"{cond}: the copy loads different activations")
            report["copy_loads_identically"] = True
    except JobMismatch as exc:
        print(f"VERIFY FAILED: {exc}")
        (root / f"{args.name}-verify.json").write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1
    report["ok"] = True
    (root / f"{args.name}-verify.json").write_text(json.dumps(report, indent=2))
    for cond, v in report["views"].items():
        print(f"{cond}: {v['questions']} questions, {v['states']} states, "
              f"{v['layers']} layers x {v['d_model']}")
    print(f"VERIFIED {len(report['chunks'])} chunk directories" +
          (" and a restored copy" if args.copy_to else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
