"""Exploratory n_eff model: hand checks and stub mechanics."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_independent_voter_model():
    N = _load("expa_neff")
    assert math.isclose(N.indep(0.7, 1, 3), 0.7)
    # k=3, one wrong option (m=1): correct wins iff >= 2 correct votes
    assert math.isclose(N.indep(0.6, 3, 1), 3 * 0.6 ** 2 * 0.4 + 0.6 ** 3)
    assert math.isclose(N.predict(0.6, 2.5, 1), (N.indep(0.6, 2, 1) + N.indep(0.6, 3, 1)) / 2)


def test_neff_on_the_stub(tmp_path):
    G, N = _load("gen_run"), _load("expa_neff")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "all", "--limit", "24", "--chunk", "12"]
    for mode in ("gemma", "qwen_nt"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    r = N.run(tmp_path, "x", ROOT / "artifacts/expab/plan")
    assert set(r["fit_development"]["p"]) == {"gemma", "qwen_nt"}
    c = r["curves_heldout"]
    assert "gemma x10" in c and "gemma x5 + qwen_nt x5" in c
    assert all(0 <= v["predicted"] <= 1 and 0 <= v["observed"] <= 1 for v in c.values())
