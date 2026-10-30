"""Stage a parent job's outputs on a fresh runtime for a CHILD job (second rounds, controls), one
chunk at a time and hash-checked.

A child reads from its parent: the job record, question snapshot and split copy; each chunk's
round-0 directory IN FULL (``load_round0`` checks every shard's hash); and, for a further round,
the previous round's condition directories -- of which it reads only ``manifest.json``,
``completion.json``, ``agent_states.parquet`` and ``transcripts.jsonl``
(``multi._load_previous_round``). Their activation arrays are never read, so they are left out
(``strip=True``), and every directory staged without them carries ``STAGED_PARTIAL.json``
listing what was omitted: such a directory must never be backed up or analysed as the parent's.

    pack_job_files / pack_chunk   (laptop)  -> tar + manifest {relative path: sha256}
    unpack_verified               (runtime) -> extract into an empty staging dir, check every file,
                                               then move into the output root

Standard library only: the runtime side is loaded by path from an uploaded copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

JOB_FILES = ("-job.json", "-questions.jsonl", "-split.json")
CHILD_READS = ("manifest.json", "completion.json", "agent_states.parquet", "transcripts.jsonl")
PARTIAL = "STAGED_PARTIAL.json"


class StageError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _pack(tar: Path, root: Path, rels: list[str], partial: dict[str, list[str]]) -> dict:
    files = {r: _sha(root / r) for r in rels}
    with tarfile.open(tar, "w") as t:
        for r in rels:
            t.add(root / r, arcname=r)
    return {"tar": str(tar), "tar_sha256": _sha(tar), "files": files, "partial": partial,
            "bytes": tar.stat().st_size}


def pack_job_files(src_root, job: str, tar) -> dict:
    src_root = Path(src_root)
    rels = [f"{job}{s}" for s in JOB_FILES if (src_root / f"{job}{s}").exists()]
    if f"{job}-job.json" not in rels or f"{job}-questions.jsonl" not in rels:
        raise StageError(f"{job}: job record or question snapshot missing in {src_root}")
    return _pack(Path(tar), src_root, rels, {})


def pack_chunk(src_root, job: str, tag: str, after: list[str], tar, strip: bool = True) -> dict:
    """One chunk: ``<job>-round0-<tag>`` in full, and each ``<job>-<cond>-<tag>`` for cond in
    ``after`` (only the files a child reads when ``strip``)."""
    src_root = Path(src_root)
    rels, partial = [], {}
    r0 = src_root / f"{job}-round0-{tag}"
    if not (r0 / "completion.json").exists():
        raise StageError(f"{r0.name} is not complete")
    rels += sorted(str(f.relative_to(src_root)) for f in r0.rglob("*") if f.is_file())
    for c in after:
        d = src_root / f"{job}-{c}-{tag}"
        if not (d / "completion.json").exists():
            raise StageError(f"{d.name} is not complete")
        allf = sorted(str(f.relative_to(d)) for f in d.rglob("*") if f.is_file())
        keep = [f for f in allf if f in CHILD_READS] if strip else allf
        if strip and set(CHILD_READS) - set(keep):
            raise StageError(f"{d.name} lacks {sorted(set(CHILD_READS) - set(keep))}")
        rels += [f"{d.name}/{f}" for f in keep]
        if strip:
            partial[d.name] = [f for f in allf if f not in keep]
    return _pack(Path(tar), src_root, rels, partial)


def unpack_verified(tar, manifest: dict, out_root, staging) -> list[str]:
    """Check the tar, extract into an empty staging dir, check every file against the manifest,
    then install the top-level items into ``out_root`` (never over an existing one). Returns the
    installed names. Raises StageError on any mismatch; nothing is installed then."""
    tar, out_root, staging = Path(tar), Path(out_root), Path(staging)
    if _sha(tar) != manifest["tar_sha256"]:
        raise StageError("uploaded tar does not match its checksum")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(tar) as t:
        t.extractall(staging, filter="data")
    got = {str(f.relative_to(staging)) for f in staging.rglob("*") if f.is_file()}
    if got != set(manifest["files"]):
        raise StageError(f"extracted files differ from the manifest: {sorted(got ^ set(manifest['files']))[:5]}")
    for rel, h in manifest["files"].items():
        if _sha(staging / rel) != h:
            raise StageError(f"{rel} differs from the manifest after extraction")
    for name, omitted in manifest.get("partial", {}).items():
        (staging / name / PARTIAL).write_text(json.dumps({"omitted": omitted, "reason": "staged for a child "
                                                          "job; activation arrays are not read by it"}) + "\n")
    names = sorted({rel.split("/", 1)[0] for rel in manifest["files"]})
    out_root.mkdir(parents=True, exist_ok=True)
    clash = [n for n in names if (out_root / n).exists()]
    if clash:
        raise StageError(f"already present in {out_root}: {clash[:5]}")
    for n in names:
        os.replace(staging / n, out_root / n)
    shutil.rmtree(staging)
    tar.unlink()
    return names
