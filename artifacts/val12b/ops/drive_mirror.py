"""Runtime, detached: mirror every COMPLETE job directory and the job's small files from the
local out root to mounted Google Drive. Each copied directory is checked against its own
completion record (SHA-256 of every file it lists, including referenced round-0 files) before it
is put in place; round-0 directories go first because chunk directories reference them. Loops
until the job's process exits, then does a final pass and writes <dst>/<prefix>-mirror.json.

    python drive_mirror.py /content/mas_runs /content/drive/MyDrive/mas_runs val12b
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/content/mas-research-project/src")
from mad.cli_backup import SMALL_SUFFIXES  # noqa: E402
from mad.job import JobMismatch, _check_hashes  # noqa: E402

SRC, DST, PREFIX = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
DST.mkdir(parents=True, exist_ok=True)
STATE = SRC / f".{PREFIX}-mirror-state.json"
done = json.loads(STATE.read_text()) if STATE.exists() else {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def alive():
    pf = SRC / f"{PREFIX}.pid"
    return pf.exists() and os.path.exists(f"/proc/{int(pf.read_text())}")


def mirror_once():
    items = [p for p in SRC.iterdir() if p.name.startswith(PREFIX + "-") and ".tmp-" not in p.name]
    items.sort(key=lambda p: (0 if "-round0-" in p.name else 1, p.name))
    n = 0
    for p in items:
        if p.is_dir() and (p / "completion.json").exists():
            stamp = sha(p / "completion.json")
            if done.get(p.name) == stamp:
                continue
            tmp, final = DST / (p.name + ".mirror-tmp"), DST / p.name
            shutil.rmtree(tmp, ignore_errors=True)
            shutil.copytree(p, tmp)
            try:
                _check_hashes(tmp)
            except JobMismatch as e:
                print(time.strftime("%H:%M:%S"), "VERIFY FAILED", p.name, e, flush=True)
                continue
            shutil.rmtree(final, ignore_errors=True)
            os.replace(tmp, final)
            _check_hashes(final)
            done[p.name] = stamp
            STATE.write_text(json.dumps(done))
            n += 1
            print(time.strftime("%H:%M:%S"), "mirrored+verified", p.name, flush=True)
        elif p.is_file() and p.name.endswith(SMALL_SUFFIXES + (".log",)):
            shutil.copy2(p, DST / (p.name + ".tmp"))
            os.replace(DST / (p.name + ".tmp"), DST / p.name)
    return n


while True:
    a = alive()
    try:
        mirror_once()
    except Exception as e:  # noqa: BLE001  (transient Drive errors: retry next pass)
        print(time.strftime("%H:%M:%S"), "pass failed:", e, flush=True)
    if not a:
        break
    time.sleep(60)
mirror_once()
(DST / f"{PREFIX}-mirror.json").write_text(json.dumps({"verified_dirs": sorted(done), "finished": time.time()}, indent=1))
print(time.strftime("%H:%M:%S"), "final pass done:", len(done), "directories verified on Drive", flush=True)
