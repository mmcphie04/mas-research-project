"""Fix the Stage 3 development set and the actual-checking cases BEFORE any new generation
(docs/NEXT_PHASE_PLAN.md, review decisions 6 and 9). CPU only; deterministic (seed 2026).

    python scripts/dev_set.py

Development set: every validation question whose three round-0 answers are not unanimous (70),
plus 20 unanimous questions drawn with the seed. The two groups are reported separately; raw
accuracy on this enriched sample is not benchmark accuracy. Fresh questions (positions 321+)
are not touched.

Checking cases: ~40 question-agent pairs, the SAME pairs coded in every condition: 32 from
disputed questions and 8 from unanimous ones, one agent per question, drawn with the seed
before any new output exists. A 20% subset (8 cases) is marked for independent second coding.

Writes artifacts/next_phase/dev_set.json and the id lists the child jobs take with
``--subset-ids``: dev_questions.json (all 90) and dev_self_fill.json (development questions
without an existing plain self-reconsideration revision).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/next_phase"
SEED, N_UNANIMOUS, N_CASES_DISPUTED, N_CASES_UNANIMOUS, N_SECOND = 2026, 20, 32, 8, 8
CALC = {"math", "physics", "chemistry", "engineering"}


def main() -> int:
    st = pd.read_parquet(ROOT / "runs/val12b/val12b-debate/agent_states.parquet")
    st["question_id"] = st.question_id.astype(str)
    pre = st[st.phase == "pre"]
    nd = pre.groupby("question_id").answer.nunique()
    subj = pre.groupby("question_id").subject.first()
    job = json.loads((ROOT / "runs/val12b/val12b-job.json").read_text())
    order = [q["id"] for q in job["questions"]]
    chunk_of = {q: i for i, members in enumerate(job["chunks"]) for q in members}
    selfq = set(pd.read_parquet(ROOT / "runs/val12b/val12b-self-self_reconsider/agent_states.parquet")
                .question_id.astype(str))
    rng = np.random.default_rng(SEED)
    disputed = sorted(nd[nd > 1].index, key=order.index)
    unanimous_all = sorted(nd[nd == 1].index, key=order.index)
    unanimous = sorted(rng.choice(unanimous_all, N_UNANIMOUS, replace=False).tolist(), key=order.index)
    dev = sorted(disputed + unanimous, key=order.index)
    group = {q: "disputed" for q in disputed} | {q: "unanimous" for q in unanimous}
    cases = []
    for g, pool, n in (("disputed", disputed, N_CASES_DISPUTED), ("unanimous", unanimous, N_CASES_UNANIMOUS)):
        qs = rng.choice(pool, n, replace=False)
        for q in sorted(qs, key=order.index):
            cases.append({"question_id": q, "agent_id": int(rng.integers(0, 3)), "group": g,
                          "subject": subj[q], "calculation_subject": subj[q] in CALC})
    second = sorted(rng.choice(len(cases), N_SECOND, replace=False).tolist())
    for i, c in enumerate(cases):
        c["second_coding"] = i in second
    self_fill = [q for q in dev if q not in selfq]
    rec = {
        "rule": f"all {len(disputed)} disputed validation questions (round-0 answers not unanimous) + "
                f"{N_UNANIMOUS} unanimous drawn with numpy default_rng({SEED}); checking cases: "
                f"{N_CASES_DISPUTED} disputed + {N_CASES_UNANIMOUS} unanimous question-agent pairs, one per "
                f"question, same rng; second coding: {N_SECOND} of them",
        "parent_job": "val12b", "seed": SEED,
        "question_ids": dev, "groups": group, "counts": {"disputed": len(disputed), "unanimous": len(unanimous)},
        "parent_chunk": {q: chunk_of[q] for q in dev},
        "self_reconsider_exists": [q for q in dev if q in selfq], "self_reconsider_needed": self_fill,
        "checking_cases": cases,
        "question_ids_sha256": hashlib.sha256(",".join(dev).encode()).hexdigest(),
        "note": "selected from round-0 answers only, before any Stage 3 output exists",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dev_set.json").write_text(json.dumps(rec, indent=1))
    (OUT / "dev_questions.json").write_text(json.dumps({"question_ids": dev}))
    (OUT / "dev_self_fill.json").write_text(json.dumps({"question_ids": self_fill}))
    print(json.dumps({k: rec[k] for k in ("counts", "question_ids_sha256")}),
          f"self fill {len(self_fill)}; cases {len(cases)} ({sum(c['calculation_subject'] for c in cases)} calculation); "
          f"second coding {N_SECOND}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
