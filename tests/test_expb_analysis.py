"""Experiment B analysis on stub-generated panels and fake readouts: kinds, pilot sizing, paired
arm contrasts, order effect, commitment and curves compute (no real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_kinds():
    B = _load("expb_analysis")
    df = pd.DataFrame({"meta_before": [0, 0, 1, 1, 2], "answer": [0, 1, 0, 2, -1], "meta_gold": [0, 0, 0, 0, 0],
                       "parsed": [True, True, True, True, False]})
    assert B.kinds(df).tolist() == ["unchanged", "harmful", "helpful", "wrong_wrong", "unparsed"]


def test_pipeline_on_the_stub(tmp_path):
    G, H, B = _load("gen_run"), _load("hf_readouts"), _load("expb_analysis")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "dev", "--limit", "20", "--chunk", "10"]
    for mode in ("qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
        assert G.main(["--task", "b", "--mode", mode, "--arms", "debate", "alone", "dissent", "anon", "--n-panels", "6", *base]) == 0
    assert H.main(["--task", "b", "--out-root", str(tmp_path), "--job", "x", "--fake",
                   "--parts", *[f"b_{m}_{a}__dev" for m in ("qwen_nt", "qwen_t") for a in ("debate", "alone", "dissent", "anon")]]) == 0
    assert B.main(["--root", str(tmp_path), "--job", "x", "--out", str(tmp_path / "an"), "--pilot"]) == 0
    assert B.main(["--root", str(tmp_path), "--job", "x", "--out", str(tmp_path / "an"), "--split", "dev"]) == 0  # stub data: dev
    r = json.loads((tmp_path / "an" / "expb_results.json").read_text())
    s = r["summary"]["qwen_t debate"]
    assert s["panels"] <= 6 and s["agents"] == 3 * s["panels"]
    assert s["questions"] <= s["panels"] and sum(c * n for c, n in ((int(k), v) for k, v in s["panels_per_question"]["counts"].items())) == s["panels"]
    assert s["helpful"] + s["harmful"] + s["wrong_wrong"] <= s["agents"]
    assert {"qwen_nt: debate - alone", "qwen_t: anon - debate"} <= set(r["contrasts"])
    assert "qwen_t alone" not in r["order_effect"] and "qwen_nt anon" in r["order_effect"]
    assert set(r["commitment"]["qwen_t debate"]) == {"primary_closed_think", "secondary_open_think"}
    assert set(r["commitment"]["qwen_nt debate"]) == {"primary_closed_think"}
    assert set(r["commitment"]["qwen_t debate"]["primary_closed_think"]) == {"unchanged", "changed"}
    cv = r["curves"].get("qwen_t debate", {})
    assert all(len(v["p_final_answer"]) == 6 for v in cv.values()) and "qwen_nt debate" not in r["curves"]
