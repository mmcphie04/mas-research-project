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
