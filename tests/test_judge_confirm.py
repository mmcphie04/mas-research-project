"""Judge confirmation pipeline on the CPU stub: vLLM-style trace generation, fake "1"/"2" readouts,
and the committed analysis (primary on the new events, secondary on all). Mechanics only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not (ROOT / "runs/judge_full/judge").exists(), reason="needs the full judge run's outputs")
def test_confirmation_pipeline_on_the_stub(tmp_path, monkeypatch):
    G, H, C = _load("gen_run"), _load("hf_readouts"), _load("judge_confirm_analysis")
    import judge_analysis
    import revision_gate
    monkeypatch.setattr(revision_gate, "N_BOOT", 50)
    monkeypatch.setattr(judge_analysis, "N_BOOT", 50)
    assert G.main(["--task", "judge", "--out-root", str(tmp_path), "--job", "jc", "--engine", "stub", "--limit", "40"]) == 0
    assert H.main(["--task", "judge", "--out-root", str(tmp_path), "--job", "jc", "--fake"]) == 0
    r = C.analyze(tmp_path, "jc", ROOT / "runs/judge_full")
    p = r["primary_303_new_events"]
    assert p["helpful_harmful"] == 40 and r["missing_new_events"] and len(r["missing_new_events"]) == 303 - 40
    assert set(p["decision_bar"]) == {"a_accuracy_ge_0.75", "b_clearly_beats_both_oof_gates", "c_clearly_beats_own_answer_rule", "met"}
    assert r["secondary_all_391"]["helpful_harmful"] == 88 + 40
