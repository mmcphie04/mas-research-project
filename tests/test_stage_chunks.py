"""Staging generation chunks onto a (fake) runtime: one chunk per upload, hash-checked there,
resumable; a chunk that does not verify locally is never sent."""

from __future__ import annotations

import importlib.util
import shutil
import tarfile
from pathlib import Path

import pytest

from mad.gen import archive as A
from mad.job import _check_hashes
from mad.provenance import is_complete

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Fake:
    def __init__(self, root):
        self.root, self.installs = Path(root), 0

    def present(self, names):
        return [n for n in names if is_complete(self.root / n)]

    def install(self, tar, name):
        self.installs += 1
        self.root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar) as t:
            t.extractall(self.root, filter="data")
        _check_hashes(self.root / name)

    def put_file(self, local, remote_name):
        shutil.copy(local, self.root / remote_name)


def test_stage_chunks(tmp_path):
    G, S = _load("gen_run"), _load("stage_chunks")
    src = tmp_path / "laptop"
    assert G.main(["--task", "bank", "--mode", "qwen_nt", "--out-root", str(src), "--job", "x", "--engine", "stub",
                   "--limit", "6", "--chunk", "2", "--tag", "orig"]) == 0
    f = Fake(tmp_path / "rt")
    out = S.stage(f, src, "x", ["bank_qwen_nt__all_orig"], log=lambda m: None)
    assert out == {**out, "chunks": 3, "sent": 3, "skipped": 0} and (tmp_path / "rt" / "x-job.json").exists()
    assert len(A.read_rows(tmp_path / "rt", "x")) == 60
    assert S.stage(f, src, "x", ["bank_qwen_nt__all_orig"], log=lambda m: None)["sent"] == 0 and f.installs == 3
    d = next(src.glob("x-bank_qwen_nt__all_orig-c000"))
    (d / "calls.json").write_text("[]")
    with pytest.raises(Exception):  # noqa: B017 -- the local hash check refuses before sending
        S.stage(Fake(tmp_path / "rt2"), src, "x", ["bank_qwen_nt__all_orig"], log=lambda m: None)
