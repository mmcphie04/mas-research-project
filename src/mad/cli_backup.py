"""Acknowledged backup of a job from a Colab runtime to a local archive (the CLI route).

The pilot's first backup scripts marked a chunk as packed BEFORE it was downloaded. If the
download then failed, the next pack skipped that chunk and could overwrite the only tar holding
it. This module fixes the order:

    prepare (runtime) -> transfer -> verify (local) -> acknowledge (runtime)

* ``prepare`` returns the PENDING batch again if one exists that was never acknowledged (same
  tar, same manifest: a retry resends exactly what failed). Otherwise it packs every item not
  yet acknowledged -- complete directories (``completion.json`` present; directories only become
  complete by an atomic rename) and the job's small files -- with a manifest of per-file
  SHA-256s and the tar's own SHA-256.
* ``receive`` (local) checks the tar's checksum, extracts into an empty staging directory, checks
  every file against the manifest, moves items into the archive, and then checks each
  directory against its own ``completion.json`` (which can only succeed once any round-0 files
  it references are in place). Any failure raises; nothing is acknowledged.
* ``acknowledge`` (runtime) records the batch's items as safely received and deletes its tar.
* ``final`` repeats sync until nothing is pending, runs ``verify_archive`` on the COMPLETE local
  archive, and only then stops the runtime. If anything fails, the runtime is left running.

The runtime side is plain functions on paths; ``Remote`` hides how they are reached (the Colab CLI
in ``scripts/colab_backup.py``, an in-process fake in the tests).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import uuid
from pathlib import Path

from mad.job import JobMismatch, _check_hashes, verify_archive

SMALL_SUFFIXES = ("-job.json", "-questions.jsonl", "-status.json", "-ledger.jsonl", "-stress.json",
                  "-split.json", "-verify.json", "-smoke-report.json", "-summary.json")


class BackupError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _stamp(path: Path) -> str:
    """What identifies an item's content: its completion record (dirs) or its bytes (files)."""
    return _sha(path / "completion.json") if path.is_dir() else _sha(path)


# ----------------------------------------------------------------------------- runtime side


def prepare(out_root, prefix: str, state_dir) -> dict:
    """The batch to send: the pending one if unacknowledged, else everything not yet acknowledged.
    Returns the manifest ({} items when there is nothing to send)."""
    out_root, state_dir = Path(out_root), Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    pending = state_dir / "pending.json"
    if pending.exists():
        m = json.loads(pending.read_text())
        if Path(m["tar"]).exists() and _sha(Path(m["tar"])) == m["tar_sha256"]:
            return m  # resend exactly the unacknowledged batch
        pending.unlink()  # its tar is gone or damaged: repack everything unacknowledged
    acked = json.loads((state_dir / "acked.json").read_text()) if (state_dir / "acked.json").exists() else {}
    items = {}
    for p in sorted(out_root.iterdir()):
        if not p.name.startswith(prefix + "-") or ".tmp-" in p.name or ".verify-" in p.name:
            continue
        if p.is_dir():
            if not (p / "completion.json").exists():
                continue  # still being written
        elif not p.name.endswith(SMALL_SUFFIXES):
            continue
        stamp = _stamp(p)
        if acked.get(p.name) == stamp:
            continue
        files = ({str(f.relative_to(out_root)): _sha(f) for f in sorted(p.rglob("*")) if f.is_file()}
                 if p.is_dir() else {p.name: _sha(p)})
        items[p.name] = {"type": "dir" if p.is_dir() else "file", "stamp": stamp, "files": files}
    if not items:
        return {"batch_id": None, "items": {}}
    batch_id = time.strftime("%Y%m%dT%H%M%S") + f"-{uuid.uuid4().hex[:8]}"  # unique per batch
    tar = state_dir / f"batch-{batch_id}.tar"
    with tarfile.open(tar, "w") as t:
        for name in items:
            t.add(out_root / name, arcname=name)
    m = {"batch_id": batch_id, "tar": str(tar), "tar_sha256": _sha(tar), "items": items}
    tmp = pending.with_suffix(".tmp")
    tmp.write_text(json.dumps(m))
    os.replace(tmp, pending)
    return m


def acknowledge(state_dir, batch_id: str) -> int:
    """Record a received batch. Only the pending batch can be acknowledged."""
    state_dir = Path(state_dir)
    pending = state_dir / "pending.json"
    if not pending.exists():
        raise BackupError("nothing pending to acknowledge")
    m = json.loads(pending.read_text())
    if m["batch_id"] != batch_id:
        raise BackupError(f"pending batch is {m['batch_id']}, not {batch_id}")
    acked_p = state_dir / "acked.json"
    acked = json.loads(acked_p.read_text()) if acked_p.exists() else {}
    acked.update({k: v["stamp"] for k, v in m["items"].items()})
    tmp = acked_p.with_suffix(".tmp")
    tmp.write_text(json.dumps(acked, indent=1))
    os.replace(tmp, acked_p)
    Path(m["tar"]).unlink(missing_ok=True)
    pending.unlink()
    return len(m["items"])


# ----------------------------------------------------------------------------- local side


def receive(tar_path, manifest: dict, dest, staging) -> list[str]:
    """Verify and install one downloaded batch. Raises BackupError on any mismatch."""
    tar_path, dest, staging = Path(tar_path), Path(dest), Path(staging)
    if _sha(tar_path) != manifest["tar_sha256"]:
        raise BackupError("downloaded tar does not match its checksum")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with tarfile.open(tar_path) as t:
            t.extractall(staging, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise BackupError(f"extraction failed: {exc}") from exc
    for item in manifest["items"].values():
        for rel, h in item["files"].items():
            f = staging / rel
            if not f.is_file() or _sha(f) != h:
                raise BackupError(f"{rel} is missing or differs from the manifest after extraction")
    dest.mkdir(parents=True, exist_ok=True)
    for name in manifest["items"]:  # install: whole items, replacing older copies
        src, dst = staging / name, dest / name
        if dst.is_dir():
            old = dst.with_name(dst.name + ".old-backup")
            if old.exists():
                shutil.rmtree(old)
            os.replace(dst, old)
            os.replace(src, dst)
            shutil.rmtree(old)
        else:
            os.replace(src, dst)
    for name, item in manifest["items"].items():  # each directory against its own record
        if item["type"] == "dir":
            try:
                _check_hashes(dest / name)
            except JobMismatch as exc:
                raise BackupError(f"{name} fails its completion checksums locally: {exc}") from exc
    shutil.rmtree(staging)
    return sorted(manifest["items"])


class Remote:
    """How the runtime side is reached. Subclasses: ColabRemote (CLI), FakeRemote (tests)."""

    def prepare(self, prefix: str) -> dict: ...
    def download(self, remote_path: str, local_path: Path) -> None: ...
    def acknowledge(self, batch_id: str) -> None: ...
    def process_alive(self) -> bool: ...
    def stop(self) -> None: ...


def sync_once(remote: Remote, prefix: str, dest, work) -> list[str]:
    """One prepare -> transfer -> verify -> acknowledge cycle. Returns the items installed
    ([] when nothing was pending). Raises on failure, leaving the batch unacknowledged."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    m = remote.prepare(prefix)
    if not m["items"]:
        return []
    local_tar = work / "batch.tar"
    local_tar.unlink(missing_ok=True)
    remote.download(m["tar"], local_tar)
    names = receive(local_tar, m, dest, work / "staging")
    remote.acknowledge(m["batch_id"])
    local_tar.unlink(missing_ok=True)
    return names


def final(remote: Remote, prefix: str, dest, work, attempts: int = 3, stop: bool = True,
          log=print) -> bool:
    """After the job's process has exited: sync until nothing is pending, verify the complete
    local archive, then (only then) stop the runtime. Returns True if stopped/verified."""
    for a in range(1, attempts + 1):
        try:
            while sync_once(remote, prefix, dest, work):
                pass
            report = verify_archive(Path(dest), prefix)
            log(f"local archive verified: {sorted(report['views'])}")
            if stop:
                remote.stop()
                log("runtime stopped")
            return True
        except (BackupError, JobMismatch, OSError, RuntimeError) as exc:
            log(f"final attempt {a} failed: {exc}")
    log("FINAL BACKUP/VERIFICATION FAILED: runtime left running so nothing is lost")
    return False
