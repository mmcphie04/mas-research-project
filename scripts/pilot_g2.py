"""G2 development timing pilot: the parent job, then -- only if it FINISHED -- the second-round /
control child job (docs/NEXT_PHASE_CPU_RESULTS.md s13.2). One process, so one detached launch.

    python scripts/pilot_g2.py --out-root /content/pilot            # on the GPU runtime
    python scripts/pilot_g2.py --out-root /tmp/x --stub             # CPU dry run (tier-0 stub)

The parent's exit code is 0 for both "finished" and "stopped_time_cap", so the child is gated on
the parent's status file (state == "finished", every chunk done), never on the exit code.
Writes <out-root>/pilot_g2-sequence.json with each step's state and wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PARENT_CONDS = ["pool", "debate3", "structured3", "debate5", "structured5"]
CHILD_CONDS = ["debate3_r2", "structured3_r2", "debate5_r2", "structured5_r2", "self_structured5"]
REAL = ["--config", "configs/base.yaml", "configs/colab_12b.yaml", "configs/main_250.yaml"]
STUB = ["--config", "configs/base.yaml", "configs/smoke_stub.yaml", "--set", "debate.n_agents=10",
        "debate.brevity_words=250", "model.max_new_tokens=1280"]


def main(argv=None) -> int:
    import run_experiment as RE

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--stub", action="store_true", help="CPU dry run with the tier-0 stub and synthetic questions")
    ap.add_argument("--parent-hours", type=float, default=1.1)
    ap.add_argument("--child-hours", type=float, default=0.35)
    a = ap.parse_args(argv)
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    seq, t0 = [], time.time()
    if a.stub:
        parent = [*STUB, "--out-root", str(out), "--name", "pilot250", "--dataset", "synthetic",
                  "--n-questions", "6", "--chunk", "2", "--conditions", *PARENT_CONDS,
                  "--max-hours", str(a.parent_hours)]
    else:
        parent = [*REAL, "--out-root", str(out), "--name", "pilot250", "--dataset", "mmlu-pro",
                  "--pool", "validation", "--pilot-size", "120", "--validation-size", "200",
                  "--n-questions", "12", "--chunk", "4",
                  "--set", "data.dataset_revision=b189ec765aa7ed75c8acfea42df31fdae71f97be",
                  "--conditions", *PARENT_CONDS, "--max-hours", str(a.parent_hours)]
    rc = RE.main(parent)
    sp = out / "pilot250-status.json"  # absent if the job stopped before its first chunk (e.g. stress)
    st = json.loads(sp.read_text()) if sp.exists() else {"state": f"no status (exit {rc})"}
    seq.append({"step": "parent", "exit": rc, "state": st.get("state"), "chunks_done": st.get("chunks_done"),
                "chunks_total": st.get("chunks_total"), "seconds": round(time.time() - t0, 1)})
    finished = st.get("state") == "finished" and st.get("chunks_done") == st.get("chunks_total")
    if finished:
        if a.stub:  # synthetic ids: the first question of each parent chunk
            job = json.loads((out / "pilot250-job.json").read_text())
            ids = out / "pilot_r2_subset_stub.json"
            ids.write_text(json.dumps({"question_ids": [c[0] for c in job["chunks"] if c][:2]}))
        else:
            ids = ROOT / "artifacts/next_phase/pilot_r2_subset.json"
        t1 = time.time()
        child = [*(STUB if a.stub else REAL), "--out-root", str(out), "--name", "pilot250_r2", "--parent", "pilot250",
                 "--subset-ids", str(ids), "--conditions", *CHILD_CONDS, "--max-hours", str(a.child_hours)]
        rc2 = RE.main(child)
        sp2 = out / "pilot250_r2-status.json"
        st2 = json.loads(sp2.read_text()) if sp2.exists() else {"state": f"no status (exit {rc2})"}
        seq.append({"step": "child", "exit": rc2, "state": st2.get("state"), "chunks_done": st2.get("chunks_done"),
                    "chunks_total": st2.get("chunks_total"), "seconds": round(time.time() - t1, 1)})
    else:
        seq.append({"step": "child", "skipped": f"parent state {st.get('state')!r}"})
    (out / "pilot_g2-sequence.json").write_text(json.dumps(seq, indent=1))
    print(json.dumps(seq))
    ok = finished and seq[-1].get("state") == "finished"
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
