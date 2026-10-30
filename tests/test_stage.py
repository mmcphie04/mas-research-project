"""Staging a parent job for a child job: one chunk at a time, hash-checked, activation arrays of
the previous round left out -- and the child job then runs on the staged copy. CPU stub only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mad import stage as S
from mad.job import chunk_tag

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml", "--set", "debate.n_agents=10", "debate.brevity_words=250",
        "model.max_new_tokens=1280"]
AFTER = ["debate3", "structured3", "debate5", "structured5"]
CHILD = ["debate3_r2", "structured3_r2", "debate5_r2", "structured5_r2", "self_structured5"]


def _driver():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    out = tmp_path_factory.mktemp("parent")
    assert _driver().main(["--config", *STUB, "--out-root", str(out), "--name", "m", "--dataset", "synthetic",
                           "--n-questions", "6", "--chunk", "2", "--conditions", "pool", *AFTER]) == 0
    return out


def stage_all(parent, dst, work, chunks):
    work.mkdir(parents=True, exist_ok=True)
    m = S.pack_job_files(parent, "m", work / "job.tar")
    S.unpack_verified(work / "job.tar", m, dst, work / "staging")
    sizes = []
    for i in chunks:
        m = S.pack_chunk(parent, "m", chunk_tag(i), AFTER, work / "c.tar")
        sizes.append(m["bytes"])
        S.unpack_verified(work / "c.tar", m, dst, work / "staging")
    return sizes


def test_staged_copy_runs_the_child_job(parent, tmp_path):
    from mad.job import verify_archive

    dst = tmp_path / "runtime"
    stage_all(parent, dst, tmp_path / "work", [0, 1, 2])
    d = dst / "m-debate5-c001"
    assert not list(d.glob("acts_*.npy")) and json.loads((d / S.PARTIAL).read_text())["omitted"]
    assert sorted(p.name for p in (dst / "m-round0-c001").iterdir()) == sorted(
        p.name for p in (parent / "m-round0-c001").iterdir()), "round 0 must be staged in full"
    job = json.loads((parent / "m-job.json").read_text())
    want = [job["chunks"][0][1], job["chunks"][2][0]]
    ids = tmp_path / "ids.json"
    ids.write_text(json.dumps({"question_ids": want}))
    assert _driver().main(["--config", *STUB, "--out-root", str(dst), "--name", "m_ext", "--parent", "m",
                           "--subset-ids", str(ids), "--conditions", *CHILD]) == 0
    st = json.loads((dst / "m_ext-status.json").read_text())
    assert st["state"] == "finished"
    verify_archive(dst, "m_ext")
    spec = importlib.util.spec_from_file_location("g4_supervisor", ROOT / "scripts" / "g4_supervisor.py")
    sup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sup)
    assert sup.incomplete(dst, "m_ext") == {}, "a child job was expected to own round-0 directories"


def test_a_corrupted_upload_installs_nothing(parent, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    m = S.pack_chunk(parent, "m", "c000", AFTER, work / "c.tar")
    b = bytearray((work / "c.tar").read_bytes())
    b[len(b) // 2] ^= 0xFF
    (work / "c.tar").write_bytes(bytes(b))
    with pytest.raises(S.StageError, match="checksum"):
        S.unpack_verified(work / "c.tar", m, tmp_path / "rt", work / "staging")
    assert not (tmp_path / "rt").exists() or not any((tmp_path / "rt").iterdir())


def test_a_file_differing_from_the_manifest_is_rejected(parent, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    m = S.pack_chunk(parent, "m", "c000", AFTER, work / "c.tar")
    m["files"][next(iter(m["files"]))] = "0" * 64
    with pytest.raises(S.StageError, match="differs"):
        S.unpack_verified(work / "c.tar", m, tmp_path / "rt", work / "staging")


def test_an_incomplete_parent_chunk_is_refused(parent, tmp_path):
    import shutil

    src = tmp_path / "p"
    shutil.copytree(parent, src)
    (src / "m-debate3-c000" / "completion.json").unlink()
    with pytest.raises(S.StageError, match="not complete"):
        S.pack_chunk(src, "m", "c000", AFTER, tmp_path / "c.tar")


class FakeStager:
    def __init__(self, root, work):
        self.root, self.work, self.installs = Path(root), Path(work), 0

    def present(self, names):
        return [n for n in names if (self.root / n / "completion.json").exists() or (self.root / n).is_file()]

    def install(self, tar, manifest):
        import shutil

        self.installs += 1
        up = self.work / "uploaded.tar"
        shutil.copy(tar, up)
        return S.unpack_verified(up, manifest, self.root, self.work / "staging")


def _driver_stage():
    spec = importlib.util.spec_from_file_location("stage_extension", ROOT / "scripts" / "stage_extension.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_driver_stages_only_needed_chunks_and_resumes(parent, tmp_path):
    D = _driver_stage()
    job = json.loads((parent / "m-job.json").read_text())
    subset = [job["chunks"][0][0], job["chunks"][2][1]]  # chunks 0 and 2 only
    (tmp_path / "up").mkdir()
    f = FakeStager(tmp_path / "rt", tmp_path / "up")
    out = D.stage(f, parent, "m", subset, AFTER, tmp_path / "work", log=lambda m: None)
    assert out["staged"] == ["job", "c000", "c002"] and f.installs == 3
    assert not (tmp_path / "rt" / "m-round0-c001").exists()
    again = D.stage(f, parent, "m", subset, AFTER, tmp_path / "work", log=lambda m: None)
    assert again["staged"] == [] and f.installs == 3, "a rerun re-uploaded installed chunks"
    with pytest.raises(SystemExit, match="not in the parent"):
        D.stage(f, parent, "m", ["no-such-question"], AFTER, tmp_path / "work", log=lambda m: None)
