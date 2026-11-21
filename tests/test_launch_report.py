"""Launch report mechanics on a stub bank (no real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_launch_report_on_the_stub(tmp_path):
    G, R = _load("gen_run"), _load("expab_launch_report")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--set", "original", "--limit", "30",
            "--chunk", "15", "--tag", "orig"]
    for mode in ("gemma", "qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    assert R.main(["--root", str(tmp_path), "--job", "x", "--tag", "orig", "--out", str(tmp_path / "rep")]) == 0
    r = json.loads((tmp_path / "rep" / "launch_report.json").read_text())
    assert set(r["bank"]) == {"gemma", "qwen_nt", "qwen_t"} and r["bank"]["qwen_t"]["rows"] == 300
    assert r["bank"]["qwen_t"]["thinking_cap_rate"] is not None and r["bank"]["gemma"]["thinking_cap_rate"] is None
    assert set(r["chooser_pools"]) == set(_load("expa_analysis").T.ARMS)
    s = r["disputed_panel_supply"]["qwen_t dev"]
    assert s["disputed_panels"] == sum(int(k) * v for k, v in s["panels_per_question"].items())
    assert 0 <= r["gemma_accuracy_check"]["vllm_bank_accuracy"] <= 1


def test_main250_like_for_like_check(tmp_path):
    import pytest

    if not (ROOT / "runs/main250/main250-pool/agent_states.parquet").exists():
        pytest.skip("needs the main250 archive (runs/main250)")
    G, R = _load("gen_run"), _load("expab_launch_report")
    assert G.main(["--task", "sanity_main250", "--mode", "gemma", "--out-root", str(tmp_path), "--job", "m",
                   "--engine", "stub", "--samples", "2", "--chunk", "100"]) == 0
    r = R.main250_like_for_like(tmp_path, "m")
    assert r["questions"] == 200 and r["vllm_samples"] == 400 and 0.5 < r["hf_accuracy"] < 0.8
    assert r["ci95"][0] <= r["difference_vllm_minus_hf"] <= r["ci95"][1] and isinstance(r["real_gap"], bool)
