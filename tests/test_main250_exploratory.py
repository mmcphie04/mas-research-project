"""The exploratory main250 analyses run end to end on a stub job with the main design, and the
fixed selection rules and the independence model behave as defined. CPU only."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _load("main250_exploratory")


def test_rules_as_defined():
    p = np.array([[0.6, 0.4, 0.0], [0.1, 0.9, 0.0], [0.2, 0.7, 0.1]])
    ans = [0, 1, 1]
    assert E.soft_vote(p, 1) == 1.0                      # sums 0.9, 2.0, 0.1
    assert E.weighted_plurality(ans, p, 0) == 0.0         # weights 0.6 vs 0.9 + 0.7
    assert E.most_confident(ans, p, 1) == 1.0             # agent 1 (0.9)
    assert E.argmax_score(np.array([1.0, 1.0, 0.0]), 0) == 0.5   # exact tie: expected value


def test_independence_table_is_a_probability_and_monotone():
    for m, k in [(3, 3), (9, 5), (9, 10)]:
        h = E.wrong_win_table(m, k)
        assert h[0] == 0 and math.isclose(h[-1], 1) and np.all(np.diff(h) >= -1e-12)
        assert np.all((h >= -1e-12) & (h <= 1 + 1e-12))
    assert math.isclose(E.icc1(np.tile([[1.0], [0.0]], (5, 4))), 1.0)


def test_exploratory_on_a_stub_main_job(tmp_path):
    out = tmp_path / "runs"
    assert _load("run_experiment").main([
        "--config", *STUB, "--set", "debate.n_agents=10", "debate.brevity_words=250", "model.max_new_tokens=1280",
        "--out-root", str(out), "--name", "m", "--dataset", "synthetic", "--n-questions", "8", "--chunk", "4",
        "--conditions", "pool", "debate3", "structured3", "debate5", "structured5"]) == 0
    r = E.analyze(out, "m", n_boot=200)
    assert r["label"] == "EXPLORATORY" and r["questions"] == 8
    assert sorted(r["1_coverage_vs_selection"]["pool"]) == list(range(3, 11))
    cov = r["1_coverage_vs_selection"]["pool"]
    assert all(v["coverage"]["mean"] >= v["plurality"]["mean"] - 1e-12 for v in cov.values())
    assert set(r["2_martingale"]["debate5"]["per_agent"]) == set(range(5))
    assert set(r["6_fixed_selection_rules"]["pool"]) == {3, 5, 10}
    assert set(r["6_fixed_selection_rules"]["post_debate"]["debate3"]) == {"plurality", "a_soft_vote",
                                                                          "b_confidence_weighted_plurality"}
    assert len(r["7_power"]) == 6
    assert sum(v["questions"] for v in r["8_by_subject_group"].values()) == 8
    json.dumps(r, default=E._json_default)
    assert "exploratory" in E.markdown(r).lower()
