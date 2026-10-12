"""Acknowledged backup of a job from a Colab CLI runtime to the laptop (see mad.cli_backup).

    # one prepare -> transfer -> verify -> acknowledge cycle
    python scripts/colab_backup.py sync  --session mas --job main12b --dest runs/colab
    # watch a running job: back up after every completed item; when its process exits, sync
    # until nothing is pending, verify the COMPLETE local archive, and only then stop the runtime
    python scripts/colab_backup.py watch --session mas --job main12b --dest runs/colab

Conventions on the runtime: code at --code-dir (a checkout of the pinned commit), outputs in
--remote-root, the job's process id in <remote-root>/<job>.pid, backup state in --state-dir.
Run ``watch`` as a long-lived background process (it does not expire). Mounted Drive (the
notebook) remains the recommended destination for main collection; this is the CLI route.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import cli_backup as B  # noqa: E402

MARK = "<<<BACKUP_JSON>>>"


class ColabRemote(B.Remote):
    def __init__(self, session, code_dir, remote_root, state_dir, job):
        self.session, self.code_dir, self.root, self.state_dir, self.job = session, code_dir, remote_root, state_dir, job

    def _colab(self, *args, timeout=600) -> str:
        r = subprocess.run(["colab", "--auth=adc", *args], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"colab {args[0]} failed: {(r.stderr or r.stdout)[-400:]}")
        return r.stdout

    def _exec(self, body: str, timeout=900):
        code = (f"import sys, json\nsys.path.insert(0, {self.code_dir + '/src'!r})\n"
                f"from mad import cli_backup as B\n{body}\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
        out = self._colab("exec", "-s", self.session, "-f", fh.name, timeout=timeout)
        Path(fh.name).unlink(missing_ok=True)
        if MARK not in out:
            raise RuntimeError(f"no result from the runtime: {out[-400:]}")
        return json.loads(out.split(MARK, 1)[1].strip().splitlines()[0])

    def prepare(self, prefix):
        return self._exec(f"print({MARK!r} + json.dumps(B.prepare({self.root!r}, {prefix!r}, {self.state_dir!r})))")

    def download(self, remote_path, local_path):
        self._colab("download", "-s", self.session, remote_path, str(local_path), timeout=3600)

    def acknowledge(self, batch_id):
        self._exec(f"print({MARK!r} + json.dumps(B.acknowledge({self.state_dir!r}, {batch_id!r})))")

    def process_alive(self):
        body = (f"import subprocess\npid = int(open({self.root + '/' + self.job + '.pid'!r}).read())\n"
                f"print({MARK!r} + json.dumps(subprocess.run(['ps', '-p', str(pid)], capture_output=True).returncode == 0))")
        return bool(self._exec(body, timeout=150))

    def stop(self):
        self._colab("stop", "-s", self.session, timeout=300)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["sync", "watch", "final"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--dest", required=True, help="the local archive root")
    ap.add_argument("--code-dir", default="/content/mas-research-project")
    ap.add_argument("--remote-root", default="/content/mas_runs")
    ap.add_argument("--state-dir", default="/content/backup_state")
    ap.add_argument("--work", default=None, help="local scratch for downloads (default: <dest>/.backup-work)")
    ap.add_argument("--poll", type=int, default=150)
    ap.add_argument("--no-stop", action="store_true", help="final: never stop the runtime")
    args = ap.parse_args(argv)
    remote = ColabRemote(args.session, args.code_dir, args.remote_root, args.state_dir, args.job)
    dest = Path(args.dest)
    work = Path(args.work or dest / ".backup-work")

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    if args.command == "sync":
        log(f"installed: {B.sync_once(remote, args.job, dest, work)}")
        return 0
    if args.command == "final":
        return 0 if B.final(remote, args.job, dest, work, stop=not args.no_stop, log=log) else 4
    failures = 0
    while True:  # watch
        try:
            alive = remote.process_alive()
            names = B.sync_once(remote, args.job, dest, work)
            if names:
                log(f"backed up and acknowledged: {names}")
            failures = 0
        except Exception as exc:  # transient CLI errors: retry; the batch stays unacknowledged
            failures += 1
            log(f"cycle failed ({failures}): {str(exc)[:200]}")
            if failures >= 20:
                log("giving up after 20 failed cycles; the runtime is left running")
                return 3
            time.sleep(60)
            continue
        if not alive:
            log("job process exited")
            return 0 if B.final(remote, args.job, dest, work, stop=not args.no_stop, log=log) else 4
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
