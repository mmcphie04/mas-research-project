"""G4 supervisor: the runtime is stopped however the job ends, and backups run per chunk.

Driven against the fake runtime of test_cli_backup (a stub job in a temporary directory) with a
scripted balance; no CLI, no GPU.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from test_cli_backup import STUB, FakeRemote, _driver

ROOT = Path(__file__).resolve().parents[1]


def _sup():
    spec = importlib.util.spec_from_file_location("g4_supervisor", ROOT / "scripts" / "g4_supervisor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def setup(tmp_path):
    remote_root = tmp_path / "runtime" / "out"
    assert _driver().main(["--config", *STUB, "--out-root", str(remote_root), "--name", "j", "--dataset",
                           "synthetic", "--n-questions", "4", "--chunk", "2",
                           "--conditions", "debate", "self_reconsider"]) == 0
    remote = FakeRemote(remote_root, tmp_path / "runtime" / "sync")
    return remote, tmp_path / "laptop", tmp_path / "work"


def make(setup, balances, **kw):
    remote, dest, work = setup
    S = _sup()
    seq = iter(balances)
    calls = {"offdevice": 0}

    def off():
        calls["offdevice"] += 1
        return True

    sup = S.Supervisor(remote, "j", dest, work, stop_balance=100.0, wind_down=1.0, poll=0,
                       balance=kw.pop("balance", lambda: next(seq)), offdevice=off, sleep=lambda s: None,
                       log=lambda m: None, events=dest / "events.jsonl", final_seconds=kw.pop("final_seconds", 60), **kw)
    return sup, calls


def test_job_exit_syncs_verifies_and_stops(setup):
    remote, dest, _ = setup
    remote.alive = False
    sup, calls = make(setup, [150.0])
    out = sup.run()
    assert out["reason"] == "job process exited" and out["local_verified"] and out["runtime_stopped"]
    assert remote.stopped and calls["offdevice"] >= 1 and out["incomplete"] == {}
    assert (dest / "j-debate-c001" / "completion.json").exists()


def test_backs_up_while_running_then_stops_when_the_job_ends(setup):
    remote, dest, _ = setup
    remote.alive = True
    sup, calls = make(setup, [150.0, 149.0, 148.0])
    orig = remote.process_alive
    n = {"k": 0}

    def alive():
        n["k"] += 1
        return n["k"] < 2

    remote.process_alive = alive
    out = sup.run()
    assert out["runtime_stopped"] and calls["offdevice"] >= 2  # after the first chunk batch and at the end
    remote.process_alive = orig


def test_stop_line_stops_at_once_without_a_final_backup(setup):
    remote, dest, _ = setup
    remote.alive = True
    sup, _ = make(setup, [99.9])
    out = sup.run()
    assert out["reason"] == "stop line reached" and remote.stopped and not out["local_verified"]
    assert not (dest / "j-debate-c000").exists()


def test_wind_down_backs_up_then_stops(setup):
    remote, dest, _ = setup
    remote.alive = True
    sup, _ = make(setup, [100.5])
    out = sup.run()
    assert out["reason"].startswith("stop line approaching") and out["local_verified"] and remote.stopped


def test_final_backup_failure_still_stops_and_reports_incomplete(setup):
    remote, dest, _ = setup
    remote.alive = False
    shutil.rmtree(remote.out_root / "j-round0-c001")  # the runtime lost a dependency
    sup, _ = make(setup, [150.0], final_seconds=0.5)
    out = sup.run()
    assert remote.stopped and out["runtime_stopped"] and not out["local_verified"]
    assert out["incomplete"], "nothing reported as incomplete"


def test_repeated_cli_failures_stop_the_runtime(setup):
    remote, _, _ = setup

    def broken():
        raise RuntimeError("Connection was lost")

    sup, _ = make(setup, [], balance=broken, max_failures=3)
    out = sup.run()
    assert out["reason"] == "3 consecutive failed cycles" and remote.stopped


def test_events_are_logged_and_never_mention_accuracy(setup):
    remote, dest, _ = setup
    remote.alive = False
    sup, _ = make(setup, [150.0])
    sup.run()
    text = (dest / "events.jsonl").read_text()
    kinds = [json.loads(x)["event"] for x in text.splitlines()]
    assert kinds[0] == "cycle" and "runtime_stopped" in kinds and kinds[-1] == "finished"
    assert "correct" not in text and "accuracy" not in text


def test_report_reads_no_correctness_column(setup, monkeypatch):
    remote, dest, work = setup
    from mad import cli_backup as B

    while B.sync_once(remote, "j", dest, work):
        pass
    spec = importlib.util.spec_from_file_location("g4_report", ROOT / "scripts" / "g4_report.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    import pandas as pd

    seen = []
    real = pd.read_parquet

    def spy(path, columns=None, **kw):
        seen.append(columns)
        return real(path, columns=columns, **kw)

    monkeypatch.setattr(R.pd, "read_parquet", spy)
    r = R.report(dest, "j")
    assert seen and all(c is not None and not any("correct" in x for x in c) for c in seen)
    assert r["conditions"]["debate"]["responses"] > 0 and "accuracy" not in json.dumps(r)


class PerChunkRemote(FakeRemote):
    def prepare(self, prefix):
        from mad import cli_backup as B

        return B.prepare(self.out_root, prefix, self.state_dir, per_chunk=True)


def test_a_backlog_is_fetched_chunk_by_chunk_with_a_drive_copy_after_each(setup):
    remote, dest, work = setup
    r = PerChunkRemote(remote.out_root, remote.state_dir)
    r.alive = False
    S = _sup()
    order = []
    sup = S.Supervisor(r, "j", dest, work, stop_balance=100.0, poll=0, balance=lambda: 150.0,
                       offdevice=lambda: order.append("drive") or True, sleep=lambda s: None,
                       log=lambda m: order.append(json.loads(m)["event"]) if m.startswith("{") else None)
    out = sup.run()
    assert out["local_verified"] and r.stopped
    fetched = [i for i, e in enumerate(order) if e in ("cycle", "backed_up")]
    drives = [i for i, e in enumerate(order) if e == "drive"]
    assert len(fetched) >= 2 and all(any(f < d < g for d in drives) for f, g in zip(fetched, fetched[1:], strict=False))


def test_no_further_fetch_until_the_drive_copy_verifies(setup):
    remote, dest, work = setup
    r = PerChunkRemote(remote.out_root, remote.state_dir)
    r.alive = True
    S = _sup()
    drive = iter([False, False, True, True, True, True, True, True])
    alive = iter([True, True, True, False])
    r.process_alive = lambda: next(alive)
    events = []
    sup = S.Supervisor(r, "j", dest, work, stop_balance=100.0, poll=0, balance=lambda: 150.0,
                       offdevice=lambda: next(drive), sleep=lambda s: None,
                       log=lambda m: events.append(json.loads(m)))
    sup.run()
    cycles = [e for e in events if e["event"] == "cycle"]
    assert cycles[0]["backed_up"] and cycles[1]["backed_up"] == [], "fetched again before the Drive copy verified"
    assert r.stopped


def test_final_backup_is_not_cut_off_while_it_makes_progress(setup):
    remote, dest, work = setup
    r = PerChunkRemote(remote.out_root, remote.state_dir)
    r.alive = False
    S = _sup()
    clock = {"t": 0.0}
    import time as _t

    real = _t.time
    try:
        S.time.time = lambda: clock["t"]

        def slow_offdevice():
            clock["t"] += 50  # each chunk takes 50 s; the no-progress limit is 60 s, total > 60 s
            return True

        sup = S.Supervisor(r, "j", dest, work, stop_balance=100.0, poll=0, balance=lambda: 150.0,
                           offdevice=slow_offdevice, sleep=lambda s: None, log=lambda m: None, final_seconds=60)
        out = sup.run()
    finally:
        S.time.time = real
    assert out["local_verified"] and r.stopped
