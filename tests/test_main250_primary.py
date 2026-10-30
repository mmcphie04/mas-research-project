"""The main250 primary analysis runs end to end on a stub job with the main collection's design
(pool of 10, nested 3- and 5-agent panels, ordinary and structured debate). CPU only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_primary_analysis_on_a_stub_main_job(tmp_path):
    out = tmp_path / "runs"
    assert _load("run_experiment").main([
        "--config", *STUB, "--set", "debate.n_agents=10", "debate.brevity_words=250", "model.max_new_tokens=1280",
        "--out-root", str(out), "--name", "m", "--dataset", "synthetic", "--n-questions", "6", "--chunk", "2",
        "--conditions", "pool", "debate3", "structured3", "debate5", "structured5"]) == 0
    P = _load("main250_primary")
    r = P.analyze(out, "m")
    assert r["questions_analyzed"] == 6 and set(r["missing_by_condition"].values()) == {0}
    assert r["n_primary_contrasts"] == 6
    assert set(r["primary_contrasts"]) == {"debate3 vs independent 6", "structured3 vs independent 6",
                                           "debate5 vs independent 10", "structured5 vs independent 10",
                                           "structured3 vs debate3", "structured5 vs debate5"}
    assert sorted(r["independent_curve"]) == list(range(3, 11))
    for c in r["primary_contrasts"].values():
        assert c["questions"] == 6 and c["lo"] <= c["mean_diff"] <= c["hi"]
    e = r["revision_events"]["debate5"]
    assert e["per_agent"]["revisions"] == 30 and e["per_agent"]["helpful"] <= e["per_agent"]["changed"]
    assert r["fallback"]["pool"]["answers"] == 60
    assert r["cost"]["conditions"]["debate3"]["revisions"] == 18
    # independent arm of the 3-agent contrast is 6 answers, the debate arm 3 answers + 3 revisions
    ind = r["primary_contrasts"]["debate3 vs independent 6"]["cost_independent"]
    assert ind["output_tokens_per_q"] > 0
    json.dumps(r)  # serializable
    assert "| debate3 vs independent 6 |" in P.markdown(r)
