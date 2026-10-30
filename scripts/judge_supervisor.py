"""Laptop supervisor for the cross-model judge run: fetch outputs after every finished step, watch
the balance, and ALWAYS stop the runtime when the run ends.

    setsid nohup systemd-run --user --scope -p MemoryMax=6G uv run python scripts/judge_supervisor.py \\
        --session judge --stop-balance <balance at launch - 18> >> runs/judge_full/supervisor.out 2>&1 &

Each cycle (every --poll seconds): the balance (at or below --stop-balance: stop at once; within
--wind-down of it: one final fetch, then stop); the run's status (process alive, finished steps);
when a new step has finished, a fetch: the remote output directory packed as one tar.gz (small:
answers, readouts, counts -- no activations), downloaded, checked against its sha256 and
extracted into --dest. When the process has exited: a final fetch (retried), then the runtime is
stopped whatever happened, and unfinished steps are logged. --max-failures consecutive failed
cycles also stop the runtime. Never opens any result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from g4_supervisor import colab_balance  # noqa: E402

STEPS = ["gemma_judge", "qwen3", "llama", "qwen3_think", "gemma_own"]
MARK = "<<<BACKUP_JSON>>>"


class ColabJudgeRemote:
    def __init__(self, session, code_dir, remote_dir, pidfile):
        from colab_backup import ColabRemote

        self.cli = ColabRemote(session, code_dir, "/content", "/content/backup_state", "unused")
        self.remote_dir, self.pidfile = remote_dir, pidfile

    def status(self) -> dict:
        body = (f"import os\npid = open({self.pidfile!r}).read().strip() if os.path.exists({self.pidfile!r}) else None\n"
                f"d = {self.remote_dir!r}\n"
                f"done = sorted(s for s in os.listdir(d) if os.path.exists(os.path.join(d, s, 'DONE'))) if os.path.isdir(d) else []\n"
                f"print({MARK!r} + json.dumps({{'alive': bool(pid) and os.path.exists('/proc/' + pid), 'done': done}}))")
        return self.cli._exec(body, timeout=150)

    def pack(self) -> dict:
        body = (f"import tarfile, hashlib, os\np = '/content/judge_fetch.tgz'\n"
                f"t = tarfile.open(p, 'w:gz'); t.add({self.remote_dir!r}, arcname='judge'); t.close()\n"
                f"print({MARK!r} + json.dumps({{'path': p, 'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest(), "
                f"'bytes': os.path.getsize(p)}}))")
        return self.cli._exec(body, timeout=600)

    def download(self, remote_path, local_path):
        self.cli.download(remote_path, local_path)

    def stop(self):
        self.cli.stop()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class JudgeSupervisor:
    def __init__(self, remote, dest, *, stop_balance, wind_down=1.0, poll=180, max_failures=8,
                 balance=colab_balance, sleep=time.sleep, log=print, events=None):
        self.remote, self.dest = remote, Path(dest)
        self.stop_balance, self.wind_down, self.poll, self.max_failures = stop_balance, wind_down, poll, max_failures
        self.balance, self.sleep, self.log = balance, sleep, log
        self.events = Path(events) if events else None
        self.fetched: list[str] = []

    def event(self, kind, **kw):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **kw}
        self.log(json.dumps(rec))
        if self.events:
            self.events.parent.mkdir(parents=True, exist_ok=True)
            with open(self.events, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def fetch(self, done) -> bool:
        m = self.remote.pack()
        work = self.dest / ".fetch"
        work.mkdir(parents=True, exist_ok=True)
        local = work / "judge_fetch.tgz"
        local.unlink(missing_ok=True)
        self.remote.download(m["path"], local)
        if _sha(local) != m["sha256"]:
            raise RuntimeError("downloaded archive does not match its sha256")
        with tarfile.open(local) as t:
            t.extractall(self.dest, filter="data")
        (self.dest / "fetch_manifest.json").write_text(json.dumps({**m, "done": done, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
        self.fetched = list(done)
        self.event("fetched", done=done, bytes=m["bytes"], sha256=m["sha256"][:12])
        return True

    def stop_runtime(self, reason) -> bool:
        for attempt in range(1, 6):
            try:
                self.remote.stop()
                self.event("runtime_stopped", reason=reason, attempt=attempt)
                return True
            except Exception as exc:  # noqa: BLE001
                self.event("stop_failed", attempt=attempt, error=str(exc)[:300])
                self.sleep(30)
        self.event("STOP_FAILED_ALL_ATTEMPTS", reason=reason)
        return False

    def finish(self, reason, fetch=True) -> dict:
        ok = False
        if fetch:
            for attempt in range(1, 4):
                try:
                    done = self.remote.status()["done"]
                    ok = self.fetch(done)
                    break
                except Exception as exc:  # noqa: BLE001
                    self.event("final_fetch_failed", attempt=attempt, error=str(exc)[:300])
                    self.sleep(30)
        stopped = self.stop_runtime(reason)
        out = {"reason": reason, "fetched": ok, "runtime_stopped": stopped, "steps_fetched": self.fetched,
               "incomplete": [s for s in STEPS if s not in self.fetched]}
        self.event("finished", **out)
        return out

    def run(self) -> dict:
        failures = 0
        while True:
            try:
                bal = self.balance()
                if bal <= self.stop_balance:
                    self.event("stop_line", balance=bal)
                    return self.finish("stop line reached", fetch=False)
                if bal <= self.stop_balance + self.wind_down:
                    self.event("wind_down", balance=bal)
                    return self.finish("stop line approaching (wind-down)")
                st = self.remote.status()
                if set(st["done"]) - set(self.fetched):
                    self.fetch(st["done"])
                failures = 0
                self.event("cycle", balance=bal, alive=st["alive"], done=st["done"])
            except Exception as exc:  # noqa: BLE001
                failures += 1
                self.event("cycle_failed", failures=failures, error=str(exc)[:300])
                if failures >= self.max_failures:
                    return self.finish(f"{failures} consecutive failed cycles", fetch=False)
                self.sleep(60)
                continue
            if not st["alive"]:
                return self.finish("job process exited")
            self.sleep(self.poll)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--stop-balance", type=float, required=True)
    ap.add_argument("--dest", default=str(ROOT / "runs/judge_full"))
    ap.add_argument("--remote-dir", default="/content/judge")
    ap.add_argument("--pidfile", default="/content/judge_full.pid")
    ap.add_argument("--code-dir", default="/content/mas-research-project")
    ap.add_argument("--poll", type=int, default=180)
    a = ap.parse_args(argv)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sup = JudgeSupervisor(ColabJudgeRemote(a.session, a.code_dir, a.remote_dir, a.pidfile), a.dest,
                          stop_balance=a.stop_balance, poll=a.poll, log=log, events=Path(a.dest) / "supervisor-events.jsonl")
    out = sup.run()
    return 0 if out["runtime_stopped"] and out["fetched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
