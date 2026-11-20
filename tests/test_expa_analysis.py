"""Experiment A analysis on a stub-generated bank and chooser runs: invariants and hand-checked
pieces (no real data)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_vote_and_sc_curve_by_hand():
    X = _load("expa_analysis")
    assert X.vote_score([0, 0, 1], 0) == 1.0 and X.vote_score([0, 1], 0) == 0.5 and X.vote_score([-1, -1], 0) == 0.0
    c = X.sc_curve([0, 1, 1], 0, 3)
    # n=1: 1/3; n=2: pairs (0,1)->1/2, (0,1)->1/2, (1,1)->0 => 1/3; n=3: 0
    assert np.allclose(c, [1 / 3, 1 / 3, 0])
    assert X.interp(np.array([0.2, 0.4, 0.5]), 2.0, 3.0) == (0.30000000000000004, "")
    assert X.interp(np.array([0.2, 0.4]), 2.0, 1.0) == (0.2, "below_curve")
    assert X.interp(np.array([0.2, 0.4]), 2.0, 9.0) == (0.4, "beyond_curve")


def test_analysis_invariants_on_the_stub(tmp_path):
    G, X = _load("gen_run"), _load("expa_analysis")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "all", "--limit", "30", "--chunk", "10"]
    for mode in ("gemma", "qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    for arm in ("A1_qt5", "A1_qnt10", "A2_5", "A3"):
        assert G.main(["--task", "choose", "--arm", arm, *base]) == 0
    r = X.run(tmp_path, "x", ROOT / "artifacts/expab/plan")
    assert r["headline_arm"] in {"A1_qt5", "A1_qnt10", "A2_5", "A3"}
    for arm, v in r["heldout"].items():
        d = v["decomposition"]
        assert math.isclose(d["R_recovery"] + d["I_new_correct"] - d["L_damage"], d["net"], abs_tol=1e-9)
        assert math.isclose(v["P1_minus_pool_plurality"]["mean"], v["accuracy"]["mean"] - v["pool_plurality"]["mean"], abs_tol=1e-9)
        assert d["I_new_correct"] == 0.0  # a chooser picks among the pool's answers: never new-correct
        assert v["called"] <= v["questions"] and v["P3_gap"]["questions"] <= v["called"]
        assert v["mean_cost"]["seconds"] >= 0
    sc = r["self_consistency_heldout"]
    assert len(sc["gemma"]["accuracy"]) == 10 and len(sc["qwen_t"]["accuracy"]) == 5
    assert set(r["costs"]) >= {"bank_gemma", "bank_qwen_nt", "bank_qwen_t", "choose_A3"}
