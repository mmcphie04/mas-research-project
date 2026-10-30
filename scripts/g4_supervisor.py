"""Laptop supervisor for the G4 main collection: back up every finished chunk, watch the compute
balance, and ALWAYS stop the runtime when the job ends -- however it ends.

    setsid nohup systemd-run --user --scope -p MemoryMax=6G uv run python scripts/g4_supervisor.py \\
        --session main --stop-balance 129.98 >> runs/main250/supervisor.out 2>&1 &

Each cycle (every --poll seconds):

1. Read the balance (``colab usage``). At or below --stop-balance: stop the runtime at once, no
   final backup. At or below --stop-balance + --wind-down: one bounded final backup, then stop.
2. Read whether the job's process is alive, then one acknowledged backup cycle
   (mad.cli_backup.sync_once) of ONE chunk (~0.4 GB; never one tar of a whole backlog) into
   --dest, then copy --dest to Google Drive and check it (scripts/offdevice_backup.py --sets
   main250) before the next chunk is fetched. A backlog is worked through without waiting.
3. Job process gone (finished, time cap, out of memory, any failure): sync until nothing is
   pending, verify the complete local archive, stop the runtime, and log what is incomplete.
   Unlike ``cli_backup.final`` the runtime is stopped even if the final backup fails: what
   is not yet backed up is lost, which is logged; an idle runtime burning units is not allowed.
4. --max-failures consecutive failed cycles (CLI unreachable): stop the runtime.

Never reads answers or correctness: the only job files it opens are the status and job records.
Writes one JSON line per event to --events (default <dest>/supervisor-events.jsonl).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mad import cli_backup as B  # noqa: E402
from mad.job import JobMismatch, chunk_tag, verify_archive  # noqa: E402
from mad.provenance import is_complete  # noqa: E402


def colab_balance(timeout=120) -> float:
    r = subprocess.run(["colab", "--auth=adc", "usage"], capture_output=True, text=True, timeout=timeout)
    m = re.search(r"Current balance:\s*([\d.]+)", r.stdout)
    if r.returncode != 0 or not m:
        raise RuntimeError(f"colab usage failed: {(r.stderr or r.stdout)[-300:]}")
    return float(m.group(1))


def offdevice(set_name="main250", timeout=3600) -> bool:
    r = subprocess.run([sys.executable, str(ROOT / "scripts/offdevice_backup.py"), "--sets", set_name],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0 and "ALL VERIFIED" in r.stdout


def incomplete(dest: Path, job: str) -> dict:
    """Chunks of the job (from its local job record) with no complete local directory."""
    jp = dest / f"{job}-job.json"
    if not jp.exists():
        return {"job_record": "missing locally"}
    rec = json.loads(jp.read_text())
    # a child job owns no round 0: it reads its parent's
    conds = ([] if rec["identity"].get("parent") else ["round0"]) + list(rec["identity"]["conditions"])
    out = {}
    for c in conds:
        missing = [chunk_tag(i) for i, m in enumerate(rec["chunks"]) if m and not is_complete(dest / f"{job}-{c}-{chunk_tag(i)}")]
        if missing:
            out[c] = missing
    return out


class Supervisor:
    def __init__(self, remote, job, dest, work, *, stop_balance, wind_down=1.0, poll=240, max_failures=8,
                 balance=colab_balance, offdevice=offdevice, sleep=time.sleep, log=print, events=None,
                 final_seconds=1800):
        self.remote, self.job, self.dest, self.work = remote, job, Path(dest), Path(work)
        self.stop_balance, self.wind_down, self.poll, self.max_failures = stop_balance, wind_down, poll, max_failures
        self.balance, self.offdevice, self.sleep, self.log = balance, offdevice, sleep, log
        self.events = Path(events) if events else None
        self.final_seconds = final_seconds
        self.offdevice_ok: bool | None = None

    def event(self, kind, **kw):
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": kind, **kw}
        self.log(json.dumps(rec))
        if self.events:
            self.events.parent.mkdir(parents=True, exist_ok=True)
            with open(self.events, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    def stop_runtime(self, reason) -> bool:
        for attempt in range(1, 6):
            try:
                self.remote.stop()
                self.event("runtime_stopped", reason=reason, attempt=attempt)
                return True
            except Exception as exc:  # noqa: BLE001 -- keep trying; report every failure
                self.event("stop_failed", reason=reason, attempt=attempt, error=str(exc)[:300])
                self.sleep(30)
        self.event("STOP_FAILED_ALL_ATTEMPTS", reason=reason)
        return False

    def backup_offdevice(self):
        try:
            self.offdevice_ok = bool(self.offdevice())
        except Exception as exc:  # noqa: BLE001
            self.offdevice_ok = False
            self.event("offdevice_error", error=str(exc)[:300])
        self.event("offdevice", verified=self.offdevice_ok)

    def final_sync(self) -> bool:
        """Sync until nothing is pending, then verify the local archive. Gives up after
        ``final_seconds`` WITHOUT progress (each installed batch restarts the clock, so a long
        backlog fetched chunk by chunk is not cut off) or when the balance reaches the stop line."""
        deadline = time.time() + self.final_seconds
        while time.time() < deadline:
            try:
                bal = self.balance()
                if bal <= self.stop_balance:
                    self.event("stop_line_during_final_backup", balance=bal)
                    return False
            except Exception as exc:  # noqa: BLE001 -- an unreadable balance does not stop the backup
                self.event("balance_unreadable", error=str(exc)[:200])
            try:
                names = B.sync_once(self.remote, self.job, self.dest, self.work)
                if names:
                    self.event("backed_up", items=names)
                    self.backup_offdevice()
                    deadline = time.time() + self.final_seconds
                    continue
                report = verify_archive(self.dest, self.job)
                self.event("local_archive_verified", views=sorted(report["views"]))
                return True
            except (B.BackupError, JobMismatch, OSError, RuntimeError) as exc:
                self.event("final_sync_failed", error=str(exc)[:300])
                self.sleep(30)
        return False

    def finish(self, reason, sync=True) -> dict:
        verified = self.final_sync() if sync else False
        stopped = self.stop_runtime(reason)
        self.backup_offdevice()
        out = {"reason": reason, "local_verified": verified, "runtime_stopped": stopped,
               "offdevice_verified": self.offdevice_ok, "incomplete": incomplete(self.dest, self.job)}
        self.event("finished", **out)
        return out

    def run(self) -> dict:
        failures = 0
        while True:
            try:
                bal = self.balance()
                if bal <= self.stop_balance:
                    self.event("stop_line", balance=bal)
                    return self.finish("stop line reached", sync=False)
                if bal <= self.stop_balance + self.wind_down:
                    self.event("wind_down", balance=bal)
                    return self.finish("stop line approaching (wind-down)")
                alive = self.remote.process_alive()
                if self.offdevice_ok is False:  # the last chunk is not yet on Google Drive: retry first
                    self.backup_offdevice()
                if self.offdevice_ok is False:
                    names = []  # fetch nothing more until the Drive copy verifies
                else:
                    names = B.sync_once(self.remote, self.job, self.dest, self.work)
                failures = 0
                self.event("cycle", balance=bal, alive=alive, backed_up=names)
                if names:
                    self.backup_offdevice()
            except Exception as exc:  # noqa: BLE001 -- transient CLI errors; the batch stays unacknowledged
                failures += 1
                self.event("cycle_failed", failures=failures, error=str(exc)[:300])
                if failures >= self.max_failures:
                    return self.finish(f"{failures} consecutive failed cycles", sync=False)
                self.sleep(60)
                continue
            if not alive:
                self.event("job_exited")
                return self.finish("job process exited")
            if not names:  # a backlog is fetched batch after batch (one chunk each) without waiting
                self.sleep(self.poll)


def main(argv=None) -> int:
    from colab_backup import ColabRemote

    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--job", default="main250")
    ap.add_argument("--dest", default=str(ROOT / "runs/main250"))
    ap.add_argument("--code-dir", default="/content/backup_code",
                    help="code the runtime-side backup calls import (a copy, so the job's checkout stays clean)")
    ap.add_argument("--remote-root", default="/content/main")
    ap.add_argument("--state-dir", default="/content/backup_state")
    ap.add_argument("--stop-balance", type=float, required=True)
    ap.add_argument("--wind-down", type=float, default=1.0)
    ap.add_argument("--poll", type=int, default=240)
    ap.add_argument("--max-failures", type=int, default=8)
    ap.add_argument("--offdevice-set", default="main250", help="scripts/offdevice_backup.py set holding --dest")
    a = ap.parse_args(argv)
    dest = Path(a.dest)

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    sup = Supervisor(ColabRemote(a.session, a.code_dir, a.remote_root, a.state_dir, a.job), a.job, dest,
                     dest / ".backup-work", stop_balance=a.stop_balance, wind_down=a.wind_down, poll=a.poll,
                     max_failures=a.max_failures, log=log, offdevice=lambda: offdevice(a.offdevice_set),
                     events=dest / ("supervisor-events.jsonl" if a.job == "main250" else f"supervisor-{a.job}-events.jsonl"))
    out = sup.run()
    return 0 if out["runtime_stopped"] and out["local_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
