"""Judge-run supervisor against a fake runtime: fetches after each finished step (hash-checked),
always stops the runtime when the run ends, at the stop line, or after repeated failures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sup():
    spec = importlib.util.spec_from_file_location("judge_supervisor", ROOT / "scripts" / "judge_supervisor.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Fake:
    def __init__(self, root: Path, script):
        self.root, self.script, self.i, self.stopped, self.corrupt = root, script, 0, False, 0
        (root / "judge").mkdir(parents=True)

    def status(self):
        st = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        for s in st["done"]:
            d = self.root / "judge" / s
            d.mkdir(exist_ok=True)
            (d / "DONE").write_text("{}")
            (d / "out.jsonl").write_text(s)
        if isinstance(st, Exception):
            raise st
        return st

    def pack(self):
        p = self.root / "fetch.tgz"
        with tarfile.open(p, "w:gz") as t:
            t.add(self.root / "judge", arcname="judge")
        return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": os.path.getsize(p)}

    def download(self, remote, local):
        b = bytearray(Path(remote).read_bytes())
        if self.corrupt:
            self.corrupt -= 1
            b[len(b) // 2] ^= 0xFF
        Path(local).write_bytes(bytes(b))

    def stop(self):
        self.stopped = True


def run(tmp_path, script, balances=None, **kw):
    S = _sup()
    f = Fake(tmp_path / "rt", script)
    bal = iter(balances or [100.0] * 50)
    corrupt = kw.pop("corrupt", 0)
    f.corrupt = corrupt
    sup = S.JudgeSupervisor(f, tmp_path / "laptop", stop_balance=50.0, poll=0, balance=lambda: next(bal),
                            sleep=lambda s: None, log=lambda m: None, **kw)
    return f, sup.run()


def test_fetch_per_step_and_stop_on_exit(tmp_path):
    script = [{"alive": True, "done": []}, {"alive": True, "done": ["gemma_judge"]},
              {"alive": True, "done": ["gemma_judge", "qwen3"]}, {"alive": False, "done": ["gemma_judge", "qwen3"]}]
    f, out = run(tmp_path, script)
    assert f.stopped and out["fetched"] and out["reason"] == "job process exited"
    assert (tmp_path / "laptop" / "judge" / "qwen3" / "out.jsonl").read_text() == "qwen3"
    assert out["incomplete"] == ["llama", "qwen3_think", "gemma_own"]


def test_stop_line_stops_without_fetch(tmp_path):
    f, out = run(tmp_path, [{"alive": True, "done": []}], balances=[49.0])
    assert f.stopped and out["reason"] == "stop line reached" and not out["fetched"]


def test_wind_down_fetches_then_stops(tmp_path):
    f, out = run(tmp_path, [{"alive": True, "done": ["gemma_judge"]}], balances=[50.5])
    assert f.stopped and out["fetched"] and out["steps_fetched"] == ["gemma_judge"]


def test_corrupted_download_is_refused_then_retried(tmp_path):
    script = [{"alive": False, "done": ["gemma_judge"]}] * 5
    f, out = run(tmp_path, script, corrupt=1)
    assert f.stopped and out["fetched"]


def test_repeated_failures_stop_the_runtime(tmp_path):
    S = _sup()
    f = Fake(tmp_path / "rt", [{"alive": True, "done": []}])

    def broken():
        raise RuntimeError("Connection was lost")

    sup = S.JudgeSupervisor(f, tmp_path / "laptop", stop_balance=50.0, poll=0, balance=broken, sleep=lambda s: None,
                            log=lambda m: None, max_failures=3)
    out = sup.run()
    assert f.stopped and out["reason"] == "3 consecutive failed cycles"
    assert json.dumps(out)
