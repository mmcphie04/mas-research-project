"""Selection-signal rules behave as defined and run end to end on a stub main job. CPU only."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load("main250_signals")


def test_rules_as_defined_and_missing_signals():
    ans = [0, 1, 1]
    d = [np.array([0.9, 0.1, 0]), np.array([0.3, 0.6, 0.1]), None]
    assert S.rule_a(ans, d, 0) == 1.0                          # 1.2 vs 0.7; the missing one skipped
    assert S.rule_b(ans, [0.9, 0.3, 0.3], 0, 3) == 1.0         # 0.9 vs 0.6: option 0 wins
    assert S.rule_b(ans, [None, None, None], 1, 3) == 1.0      # no signal -> plurality
    assert S.rule_c(ans, [0.5, None, 0.7], 1) == 1.0
    assert S.rule_c(ans, [0.7, 0.7, None], 0) == 0.5           # exact tie: expected value
    assert S.auroc(np.array([0.1, 0.9, 0.5, 0.5]), np.array([0, 1, 0, 1])) == 0.875


def test_signals_on_a_stub_main_job(tmp_path):
    out = tmp_path / "runs"
    assert _load("run_experiment").main([
        "--config", *STUB, "--set", "debate.n_agents=10", "debate.brevity_words=250", "model.max_new_tokens=1280",
        "--out-root", str(out), "--name", "m", "--dataset", "synthetic", "--n-questions", "8", "--chunk", "4",
        "--conditions", "pool", "debate3", "structured3", "debate5", "structured5"]) == 0
    r = S.analyze(out, "m", n_boot=100)
    assert r["label"] == "EXPLORATORY"
    assert set(r["selection"]) == {"pool first 3", "pool first 5", "pool first 10", "debate3 post-debate",
                                   "structured3 post-debate", "debate5 post-debate", "structured5 post-debate"}
    assert "iv_peer_exposure" not in r["selection"]["pool first 3"]["rules"]
    assert "iv_peer_exposure" in r["selection"]["debate5 post-debate"]["rules"]
    assert set(r["selection"]["pool first 5"]["rules"]["i_before_stated_answer"]) == {
        "a_soft_vote", "b_weighted_plurality", "c_highest_signal"}
    assert set(r["selection"]["pool first 5"]["rules"]["ii_response_logprob_mean"]) == {
        "b_weighted_plurality", "c_highest_signal"}
    json.dumps(r, default=S._json_default)
    assert "exploratory" in S.markdown(r).lower()


def test_within_question_pairs_and_ties():
    gold = {"q": 0}
    d = {"qs": ["q"], "gold": gold, "sets": {"pool": {"q": {
        "answers": np.array([0, 0, 1, 2, 0, 0, 0, 0, 0, 0]),
        "signals": {"ii_response_logprob_mean": [0.9, 0.5, 0.5, 0.2, None, None, None, None, None, None],
                    "iii_verbalized_confidence": [None] * 10}}}}}
    for c in S.DEBATES:
        d["sets"][c] = {"q": {"answers": np.array([0, 0, 0]), "signals": {
            "ii_response_logprob_mean": [0.5, 0.5, 0.5], "iii_verbalized_confidence": [0.5, 0.5, 0.5],
            "iv_peer_exposure": [np.array([1.0, 0.0, 0.0])] * 3}}}
    r = S.within_question(d, S.Boot(50))
    w = r["pool first 10"]["ii_response_logprob_mean"]
    # correct {0.9, 0.5} vs wrong {0.5, 0.2}: pairs win, win, tie, win -> 3.5 / 4
    assert w["pairs"] == 4 and w["within_question"]["mean"] == 0.875 and w["tie_share"] == 0.25
    assert r["debate3 post-debate"]["ii_response_logprob_mean"]["pairs"] == 0  # unanimous: no pairs
