"""Experiment B probe script mechanics on stub data with a fake model (no real signal expected)."""

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


def test_within_question_auroc_by_hand():
    P = _load("expb_probe")
    # group a: positive 0.9 vs negatives 0.1, 0.9 -> 1 + 0.5; group b: positive 0.2 vs negative 0.3 -> 0
    n, d = P.within_q_auroc([0.9, 0.1, 0.9, 0.2, 0.3], [1, 0, 0, 1, 0], ["a", "a", "a", "b", "b"])
    assert (n, d) == (1.5, 3.0)


def test_probe_pipeline_runs(tmp_path):
    G, H, P = _load("gen_run"), _load("hf_readouts"), _load("expb_probe")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "all", "--limit", "60", "--chunk", "20"]
    assert G.main(["--task", "bank", "--mode", "qwen_t", *base]) == 0
    assert G.main(["--task", "b", "--mode", "qwen_t", "--arms", "debate", "alone", "--n-panels", "20", *base[:-4], "--split", "dev",
                   "--chunk", "20"]) == 0
    assert G.main(["--task", "b", "--mode", "qwen_t", "--arms", "debate", "--n-panels", "20", "--split", "heldout",
                   "--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--chunk", "20", "--panel-seed", "5"]) == 0
    assert H.main(["--task", "b", "--out-root", str(tmp_path), "--job", "x", "--fake", "--parts", "b_qwen_t_debate__dev", "b_qwen_t_alone__dev",
                   "b_qwen_t_debate__heldout"]) == 0
    assert H.main(["--task", "posctl", "--out-root", str(tmp_path), "--job", "x", "--fake", "--n-posctl", "50"]) == 0
    assert P.main(["--root", str(tmp_path), "--job", "x", "--out", str(tmp_path / "an")]) == 0
    r = json.loads((tmp_path / "an" / "expb_probes.json").read_text())
    pc = r["positive_control"]["qwen_t"]
    assert pc["contexts"] == 50 and set(pc["probe_accuracy_by_layer"]) == {"14", "22", "31"} and pc["passed"] is False
    assert "qwen_t will_change_from_A" in r["probes"]
