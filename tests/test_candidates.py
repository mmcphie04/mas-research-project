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
    assert {"output_raw_p_candidate", "output_model", "probe", "combined"} <= set(rep["candidate_correctness"])
    assert "endorsement_agrees_with_candidate" in rep["output_features"]
    assert not any(k in n for n in rep["output_features"] for k in ("gold", "kind", "correct"))
    assert len(rep["endorsement_shifts"]) == 4


def test_candidates_follow_the_parents_registered_split(tmp_path):
    import pytest

    ms, drv, rc, ac = _load("make_split"), _load("run_experiment"), _load("run_candidates"), _load("analyze_candidates")
    split = tmp_path / "x.json"
    ms.main(["--dataset", "synthetic", "--n-questions", "40", "--fracs", "0.5", "0.2", "0.3", "--seed", "4",
             "--out", str(split)])
    assert drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "p", "--dataset", "synthetic",
                     "--n-questions", "40", "--chunk", "20", "--conditions", "debate",
                     "--split-manifest", str(split)]) == 0
    base = ["--config", *STUB, "--out-root", str(tmp_path), "--parent", "p"]
    with pytest.raises(SystemExit, match="n-per-split"):
        rc.main([*base, "--n", "10"])  # a registered split: no unstratified sampling
    assert rc.main([*base, "--n", "4", "--check"]) == 0
    assert (tmp_path / "p-candidates-check" / "completion.json").exists()
    chk = pd.read_parquet(tmp_path / "p-candidates-check" / "candidate_states.parquet")
    assert set(chk.split) == {"train"}, "a prompt check touched non-training questions"
    assert not (tmp_path / "p-candidates").exists(), "a prompt check must not become the final output"
    assert rc.main([*base, "--n-per-split", "10", "4", "8"]) == 0
    r = pd.read_parquet(tmp_path / "p-candidates" / "candidate_states.parquet")
    assert r.groupby("split").question_id.nunique().to_dict() == {"train": 10, "validation": 4, "test": 8}
    man = json.loads((tmp_path / "p-candidates" / "manifest.json").read_text())
    assert man["split_assignment_sha256"] == json.loads(split.read_text())["meta"]["assignment_sha256"]
    out = tmp_path / "a"
    assert ac.main(["--run", str(tmp_path / "p-candidates"), "--out", str(out)]) == 0
    assert json.loads((out / "candidates_report.json").read_text())["route"].startswith("split")
    other = tmp_path / "other.json"
    ms.main(["--dataset", "synthetic", "--n-questions", "40", "--seed", "99", "--out", str(other)])
    with pytest.raises(SystemExit, match="not the split p registered"):
        rc.main([*base, "--n-per-split", "2", "2", "2", "--split-manifest", str(other), "--force"])
