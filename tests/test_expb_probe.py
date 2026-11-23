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
    pc = r["positive_control"]["original_50"]["qwen_t"]
    assert pc["contexts"] == 50 and set(pc["probe_accuracy_by_position_layer"]) == {"A", "R"}
    assert set(pc["probe_accuracy_by_position_layer"]["R"]) == {"14", "22", "31"}
    assert set(pc["passed_by_position"]) == {"A", "R"}
    pr = r["probes"]
    assert "A" in pr["excluded_positions"] and pr["qwen_t helpful_vs_harmful"]["positions_considered"] == ["B", "R"]
    assert pr["qwen_t will_change"]["positions_considered"] == ["R"]
    assert not any(k.startswith("posA") for v in pr.values() if isinstance(v, dict) for k in v.get("dev_within_question_auroc", {}))

    # as in the real data: development (pilot) readouts hold [A, B] only, held-out ones [A, B, R]
    import numpy as np

    for d in tmp_path.glob("x-b_qwen_t_*__dev_ro-c*"):
        a = np.load(d / "acts.npy")
        np.save(d / "acts.npy", a[:, :2])
    assert P.main(["--root", str(tmp_path), "--job", "x", "--out", str(tmp_path / "an2")]) == 0
    pr2 = json.loads((tmp_path / "an2" / "expb_probes.json").read_text())["probes"]
    wc = pr2["qwen_t will_change"]
    assert "not computable as committed" in wc["position_notes"]["R"] and wc["chosen"] is None
    hh = pr2["qwen_t helpful_vs_harmful"]
    assert "not computable as committed" in hh["position_notes"]["R"]
    assert all(k.startswith("posB") for k in hh["dev_within_question_auroc"])
