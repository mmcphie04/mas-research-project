"""Off-device copy of the research archives to Google Drive with rclone, then a hash check.

    # one-time (interactive; opens a browser sign-in): least-privilege scope -- rclone sees only
    # the files it created
    rclone config create gdrive drive scope=drive.file
    # copy + verify every archive set (idempotent; re-running only sends what changed)
    python scripts/offdevice_backup.py --remote gdrive: --dest mas_research_backup
    # verify only
    python scripts/offdevice_backup.py --remote gdrive: --dest mas_research_backup --check-only

Sets: the pilot archive (pilot12b-* only), the validation archive, the frozen scorers and the
next-phase curve outputs. ``rclone copy --checksum`` then ``rclone check --one-way``: every
local file must exist remotely with the same size and MD5 (Google Drive stores MD5s). Writes a
dated record to artifacts/next_phase/backup/offdevice_log.jsonl. Never deletes anything remotely.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "artifacts/next_phase/backup/offdevice_log.jsonl"
SETS = {  # name -> (local dir, rclone filter args)
    "pilot12b": (ROOT / "runs/colab", ["--include", "pilot12b-*/**", "--include", "pilot12b-*"]),
    "val12b": (ROOT / "runs/val12b", []),
    "frozen": (ROOT / "runs/frozen", ["--include", "pilot12b_scorers_3857a72.pkl"]),
    "next_commit": (ROOT / "runs/next_commit", []),
}


def rclone(*args, timeout=None) -> subprocess.CompletedProcess:
    exe = shutil.which("rclone")
    if not exe:
        raise SystemExit("rclone is not installed (see docs/NEXT_PHASE_CPU_RESULTS.md s15)")
    return subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="gdrive:")
    ap.add_argument("--dest", default="mas_research_backup")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args(argv)
    remotes = rclone("listremotes").stdout.split()
    if a.remote not in remotes:
        raise SystemExit(f"rclone remote {a.remote!r} is not configured (have {remotes}); run "
                         "`rclone config create gdrive drive scope=drive.file` first")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    ok_all = True
    for name in a.sets:
        src, filt = SETS[name]
        dst = f"{a.remote}{a.dest}/{name}"
        t0 = time.time()
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "set": name, "source": str(src.relative_to(ROOT)),
               "destination": dst}
        if not a.check_only:
            cp = rclone("copy", str(src), dst, "--checksum", "--transfers", "4", *filt)
            rec["copy_returncode"] = cp.returncode
            rec["copy_stderr_tail"] = cp.stderr[-500:]
        ck = rclone("check", str(src), dst, "--one-way", *filt)
        rec.update({"check_returncode": ck.returncode, "check_summary": [x for x in ck.stderr.splitlines()
                                                                        if "differences" in x or "matching" in x or "ERROR" in x][-6:],
                    "seconds": round(time.time() - t0, 1)})
        sz = rclone("size", dst, "--json")
        rec["remote_size"] = json.loads(sz.stdout) if sz.returncode == 0 else None
        ok = ck.returncode == 0 and rec.get("copy_returncode", 0) == 0
        rec["verified"] = ok
        ok_all &= ok
        with open(LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("set", "verified", "check_summary", "remote_size", "seconds")}))
    print("ALL VERIFIED" if ok_all else "NOT VERIFIED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
