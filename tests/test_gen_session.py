"""The session driver and the smoke report on the CPU stub: the smoke plan's steps run in order
(both environments = this Python; stub engine, fake readouts), the session summary is written, the
first failing step ends the session, and the report computes throughput, sanity and pricing."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stubbed_plan(tmp_path) -> Path:
    plan = json.loads((ROOT / "artifacts/expab/ops/smoke_session.json").read_text())
    for s in plan["steps"]:  # grouped steps: add the stub engine / fake model to every sub-step
        i = s["args"].index("--steps")
        sub = json.loads(s["args"][i + 1])
        extra = ["--engine", "stub"] if "gen_run" in s["args"] else ["--fake"]
        s["args"][i + 1] = json.dumps([x + extra for x in sub])
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return p


def test_smoke_session_and_report_on_the_stub(tmp_path):
    S, R = _load("gen_session"), _load("smoke_report")
    root = tmp_path / "rt"
    assert S.main(["--plan", str(stubbed_plan(tmp_path)), "--out-root", str(root), "--job", "smoke",
                   "--python-vllm", sys.executable]) == 0
    summ = json.loads((root / "smoke-session-summary.json").read_text())
    assert [s["exit"] for s in summ["steps"]] == [0] * 3 and [s["name"] for s in summ["steps"]] == ["gemma", "qwen", "readouts"]
    assert R.main(["--root", str(root), "--job", "smoke", "--out", str(tmp_path / "rep")]) == 0
    r = json.loads((tmp_path / "rep" / "smoke_report.json").read_text())
    assert r["parts"]["bank_qwen_t__dev_smoke"]["thinking_cap_rate"] is not None
    pr = r["pricing"]
    assert {"bank_gemma", "bank_qwen_nt", "bank_qwen_t", "judge_traces", "choose_A3"} <= set(pr["seconds_per_request"])
    assert list(pr["launches"]) == ["1 bank + judge confirmation", "2 B pilot", "3 choosers", "4 B full"]
    assert "b_qwen_t_dissent" in pr["launches"]["4 B full"]["items_gpu_hours"] and pr["total_units"] > 0
    assert "A3" in pr["p2_coverage"] and pr["p2_coverage"]["A3"]["in_qwen_t_samples"] > 0
    assert set(r["position_bias"]) >= {"choose_A3", "choose_A1_g5"}
    tl = r["think_location"]["bank_qwen_t__dev_smoke"]
    assert tl["think_in_generation"] == 1.0 and tl["think_in_prompt"] == 0.0  # the stub generates its own <think>
    assert r["parts"]["bank_qwen_t__dev_smokenovis"]["answer_cap_rate"] is not None
    if r["sanity"] is not None:
        assert set(r["sanity"]) == {"gemma", "qwen_nt"}


def test_a_failing_step_ends_the_session(tmp_path):
    S = _load("gen_session")
    plan = {"steps": [{"name": "bad", "env": "hf", "script": "gen_run.py", "args": ["--task", "nope"]},
                      {"name": "never", "env": "hf", "script": "gen_run.py", "args": ["--task", "bank"]}]}
    p = tmp_path / "p.json"
    p.write_text(json.dumps(plan))
    assert S.main(["--plan", str(p), "--out-root", str(tmp_path / "rt"), "--job", "j"]) != 0
    steps = json.loads((tmp_path / "rt" / "j-session-summary.json").read_text())["steps"]
    assert [s["name"] for s in steps] == ["bad"]
