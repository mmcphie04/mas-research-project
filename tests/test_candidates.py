"""Candidate evaluation on the stub: six readouts per question, correct record format."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_candidate_run_and_analysis(tmp_path):
    drv = _load("run_experiment")
    assert drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "p", "--dataset", "synthetic",
                     "--n-questions", "40", "--chunk", "20", "--conditions", "debate"]) == 0
    rc = _load("run_candidates")
    assert rc.main(["--config", *STUB, "--out-root", str(tmp_path), "--parent", "p", "--n", "30"]) == 0
    d = tmp_path / "p-candidates"
    r = pd.read_parquet(d / "candidate_states.parquet")
    assert len(r) == 30 * 6 and r.groupby("question_id").size().eq(6).all()
    per_q = r.groupby("question_id")
    assert per_q.candidate.nunique().eq(2).all(), "one correct and one incorrect candidate per question"
    assert (r[r.candidate_kind == "incorrect"].candidate != r[r.candidate_kind == "incorrect"].gold_display).all()
    assert set(r.context) == {"neutral", "endorse_correct", "endorse_incorrect"}
    man = json.loads((d / "manifest.json").read_text())
    assert man["parent_job"] == "p" and len(man["question_ids"]) == 30
    ac = _load("analyze_candidates")
    out = tmp_path / "a"
    assert ac.main(["--run", str(d), "--out", str(out)]) == 0
    rep = json.loads((out / "candidates_report.json").read_text())
    assert {"output_only", "probe", "combined"} <= set(rep["candidate_correctness"])
    assert len(rep["endorsement_shifts"]) == 4
