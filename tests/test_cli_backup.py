"""Acknowledged backup: nothing is marked received until it is verified locally.

All failures are injected into a fake runtime built from a stub job in a temporary directory --
never against a real archive.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from mad import cli_backup as B

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _driver():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeRemote(B.Remote):
    def __init__(self, out_root, state_dir):
        self.out_root, self.state_dir = Path(out_root), Path(state_dir)
        self.fail_downloads = 0
        self.corrupt_downloads = 0
        self.stopped = False
        self.alive = False

    def prepare(self, prefix):
        return B.prepare(self.out_root, prefix, self.state_dir)

    def download(self, remote_path, local_path):
        if self.fail_downloads:
            self.fail_downloads -= 1
            raise OSError("simulated lost connection during download")
        shutil.copy(remote_path, local_path)
        if self.corrupt_downloads:
            self.corrupt_downloads -= 1
            b = bytearray(Path(local_path).read_bytes())
            b[len(b) // 2] ^= 0xFF
            Path(local_path).write_bytes(bytes(b))

    def acknowledge(self, batch_id):
        B.acknowledge(self.state_dir, batch_id)

    def process_alive(self):
        return self.alive

    def stop(self):
        self.stopped = True


@pytest.fixture
def job(tmp_path):
    remote = tmp_path / "runtime" / "mas_runs"
    assert _driver().main(["--config", *STUB, "--out-root", str(remote), "--name", "j", "--dataset",
                           "synthetic", "--n-questions", "4", "--chunk", "2",
                           "--conditions", "debate", "self_reconsider"]) == 0
    return FakeRemote(remote, tmp_path / "runtime" / "sync"), tmp_path / "laptop", tmp_path / "work"


def test_regression_a_failed_transfer_does_not_lose_the_batch(job):
    """The pilot bug: chunks were marked packed before download, so a repack after a failed
    transfer omitted them. Now the unacknowledged batch is resent unchanged."""
    remote, _, _ = job
    first = remote.prepare("j")
    assert "j-debate-c000" in first["items"]
    again = remote.prepare("j")  # the transfer "failed": no acknowledgment
    assert again["batch_id"] == first["batch_id"] and again["items"] == first["items"]


def test_failed_download_then_retry_installs_and_acknowledges(job):
    remote, dest, work = job
    remote.fail_downloads = 1
    with pytest.raises(OSError):
        B.sync_once(remote, "j", dest, work)
    assert not (remote.state_dir / "acked.json").exists(), "acknowledged without a download"
    names = B.sync_once(remote, "j", dest, work)
    assert "j-round0-c000" in names and (dest / "j-debate-c001" / "completion.json").exists()
    assert B.sync_once(remote, "j", dest, work) == [], "an acknowledged batch was sent again"


def test_a_corrupted_download_is_rejected_and_resent(job):
    remote, dest, work = job
    remote.corrupt_downloads = 1
    with pytest.raises(B.BackupError, match="checksum"):
        B.sync_once(remote, "j", dest, work)
    assert not (dest / "j-debate-c000").exists(), "a corrupted batch was installed"
    assert B.sync_once(remote, "j", dest, work)


def test_a_file_that_differs_after_extraction_is_rejected(job):
    remote, dest, work = job
    m = remote.prepare("j")
    rel = next(iter(m["items"]["j-debate-c000"]["files"]))
    bad = json.loads(json.dumps(m))
    bad["items"]["j-debate-c000"]["files"][rel] = "0" * 64
    local = work / "t.tar"
    work.mkdir(parents=True, exist_ok=True)
    remote.download(m["tar"], local)
    with pytest.raises(B.BackupError, match="differs from the manifest"):
        B.receive(local, bad, dest, work / "staging")
    assert not (dest / "j-debate-c000").exists()


def test_a_damaged_pending_tar_is_repacked_with_everything_unacknowledged(job):
    remote, _, _ = job
    m = remote.prepare("j")
    Path(m["tar"]).write_bytes(b"truncated")
    m2 = remote.prepare("j")
    assert m2["batch_id"] != m["batch_id"] and set(m2["items"]) == set(m["items"])


def test_a_chunk_completing_while_a_batch_is_pending_is_sent_next(job):
    remote, dest, work = job
    first = remote.prepare("j")
    late = remote.out_root / "j-debate-c999"
    shutil.copytree(remote.out_root / "j-debate-c000", late)  # appears after packing
    assert remote.prepare("j")["batch_id"] == first["batch_id"]
    B.sync_once(remote, "j", dest, work)
    nxt = remote.prepare("j")
    assert "j-debate-c999" in nxt["items"] and "j-debate-c000" not in nxt["items"]


def test_final_stops_only_after_the_complete_local_archive_verifies(job, tmp_path):
    remote, dest, work = job
    logs = []
    shutil.rmtree(remote.out_root / "j-round0-c001")  # the runtime lost a dependency
    assert B.final(remote, "j", dest, work, attempts=1, log=logs.append) is False
    assert not remote.stopped, "stopped although the local archive does not verify"
    remote2 = FakeRemote(tmp_path / "r2" / "mas_runs", tmp_path / "r2" / "sync")
    assert _driver().main(["--config", *STUB, "--out-root", str(remote2.out_root), "--name", "j",
                           "--dataset", "synthetic", "--n-questions", "4", "--chunk", "2",
                           "--conditions", "debate"]) == 0
    assert B.final(remote2, "j", tmp_path / "laptop2", tmp_path / "work2", log=logs.append) is True
    assert remote2.stopped
