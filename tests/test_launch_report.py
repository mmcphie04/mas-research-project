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


def test_choose_report_is_outcome_blind(tmp_path):
    G, C = _load("gen_run"), _load("expab_choose_report")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--set", "original", "--limit", "20", "--chunk", "10"]
    for mode in ("qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    assert G.main(["--task", "choose", "--arm", "A1_qnt5", "A2_5", *base]) == 0
    seen = []
    orig = C.A.read_rows
    C.A.read_rows = lambda *a, **k: (lambda df: (seen.append(set(df.columns)), df)[1])(orig(*a, **k))
    r = C.report(tmp_path, "x")
    C.A.read_rows = orig
    assert set(r) == {"A1_qnt5", "A2_5"} and r["A2_5"]["thinking_cap_rate"] is not None
    assert all(0 <= v["pick_parse_rate"] <= 1 and v["requests"] == 3 * v["questions"] for v in r.values())
    assert "meta_gold" in set().union(*seen)  # the rows carry gold ...
    assert not any(k for v in r.values() for k in v if "acc" in k or "correct" in k)  # ... the report never uses it


def test_b_ops_report_is_outcome_blind(tmp_path):
    G, B = _load("gen_run"), _load("expb_ops_report")
    base = ["--out-root", str(tmp_path), "--job", "k", "--engine", "stub", "--split", "dev", "--limit", "40", "--chunk", "20"]
    assert G.main(["--task", "bank", "--mode", "qwen_nt", *base]) == 0
    assert G.main(["--task", "b", "--mode", "qwen_nt", "--arms", "debate", "alone", "--n-panels", "6", "--tag", "full", *base]) == 0
    assert G.main(["--task", "b", "--mode", "qwen_nt", "--arms", "alone", "--panel-start", "6", "--n-panels", "5",
                   "--tag", "fullx", *base]) == 0
    r = B.report(tmp_path, "k")
    assert set(r) == {"b_qwen_nt_debate__dev_full", "b_qwen_nt_alone__dev_full", "b_qwen_nt_alone__dev_fullx"}
    assert r["b_qwen_nt_debate__dev_full"]["panels"] == 6 and r["b_qwen_nt_alone__dev_fullx"]["panels"] == 5
    assert all(v["chunks_done"] == v["chunks_expected"] and v["readouts"]["rows"] == 0 for v in r.values())
    assert not any(k for v in r.values() for k in v if any(w in k for w in ("acc", "correct", "change", "helpful", "harmful")))
