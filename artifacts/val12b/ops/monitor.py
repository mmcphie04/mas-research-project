"""Laptop: operational monitor for a running job (no accuracy or scorer output is read).

    python monitor.py --session val --name val12b --start-units 281.34 --cap-units 26.5

Every --poll seconds: job state, chunks done, elapsed hours, last chunk's seconds per question,
failures (runtime status.py) and the account balance (`colab usage`). Prints one line when
anything changes or each 2 more units are used. Exits on a terminal job state, a vanished
session, or when the units used reach the cap (it prints USAGE CAP; stopping is done separately).
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = {"PATH": f"{Path.home()}/.local/bin:{Path.home()}/google-cloud-sdk/bin:/usr/bin:/bin"}


def colab(*a, timeout=120):
    try:
        r = subprocess.run(["colab", "--auth=adc", *a], capture_output=True, text=True, timeout=timeout, env=ENV)
    except subprocess.TimeoutExpired:  # a hung CLI call (seen live) must not end the watch
        return 124, "timed out"
    return r.returncode, r.stdout + r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--start-units", type=float, required=True)
    ap.add_argument("--cap-units", type=float, required=True)
    ap.add_argument("--poll", type=int, default=300)
    a = ap.parse_args()
    last, last_units_step = None, -1
    while True:
        rc, out = colab("exec", "-s", a.session, "-f", str(HERE / "status.py"), "--env", f"NAME={a.name}", timeout=150)
        st = {}
        for line in out.splitlines():
            if line.startswith("{"):
                try:
                    st = json.loads(line)
                except ValueError:
                    pass
                break
        _, u = colab("usage", timeout=60)
        m = re.search(r"balance:\s*([0-9.]+)", u)
        rate = re.search(r"rate:\s*([0-9.]+)", u)
        used = round(a.start_units - float(m.group(1)), 2) if m else None
        tm = st.get("timings") or []
        spq = tm[-1]["seconds_per_question"] if tm else None
        key = (st.get("state"), st.get("chunks_done"), st.get("failures"), st.get("alive"))
        step = int(used // 2) if used is not None else -1
        if key != last or step != last_units_step:
            print(f"{time.strftime('%H:%M')} state={st.get('state')} alive={st.get('alive')} "
                  f"chunks={st.get('chunks_done')}/{st.get('chunks_total')} elapsed_h={st.get('elapsed_hours_total')} "
                  f"last_s_per_q={spq} failures={st.get('failures')} units_used={used} "
                  f"rate={rate.group(1) if rate else None}", flush=True)
            last, last_units_step = key, step
        if used is not None and used >= a.cap_units:
            print(f"{time.strftime('%H:%M')} USAGE CAP reached: {used} units", flush=True)
            return
        if rc != 0 and "not found" in out.lower() and "session" in out.lower():
            rc2, s2 = colab("sessions", timeout=60)
            if a.session not in s2:
                print(f"{time.strftime('%H:%M')} SESSION GONE", flush=True)
                return
        if st.get("state") in ("finished", "failed", "stopped_time_cap") and not st.get("alive"):
            print(f"{time.strftime('%H:%M')} TERMINAL state={st.get('state')}", flush=True)
            return
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
