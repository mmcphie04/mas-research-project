"""Cross-model judge: prompts, order averaging, and the runtime's code paths on a tiny real model
(gemma-3-270m-it, CPU; not science)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from mad import judge as J

ROOT = Path(__file__).resolve().parents[1]


def test_orders_and_order_average():
    e = {"question_block": "Q", "original": "ORIG", "revised": "REV"}
    o = J.both_orders(e)
    assert o["original_first"][1]["content"].index("ORIG") < o["original_first"][1]["content"].index("REV")
    assert o["revised_first"][1]["content"].index("REV") < o["revised_first"][1]["content"].index("ORIG")
    assert "original" not in o["original_first"][1]["content"].lower().split("response 1")[1][:20]
    assert J.p_revision(0.8, 0.6) == 0.7 and J.position_bias(0.7, 0.7) == pytest.approx(0.2)


def test_own_answer_prompt_is_the_main_collection_system_prompt():
    rec = next(json.loads(x) for x in open(ROOT / "runs/main250/main250-debate3/transcripts.jsonl")
               if json.loads(x)["phase"] == "pre")
    sys_msg = next(m["content"] for m in rec["messages"] if m["role"] == "system")
    user = next(m["content"] for m in rec["messages"] if m["role"] == "user")
    assert J.own_messages(user) == [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]


def test_after_thinking():
    assert J.after_thinking("<think>x</think>\n\nAnswer: (B)") == "Answer: (B)"
    assert J.after_thinking("Answer: (C)") == "Answer: (C)"


def test_runtime_on_a_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    spec = importlib.util.spec_from_file_location("judge_run", ROOT / "scripts" / "judge_run.py")
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    assert R.main(["--model", "tiny", "--task", "judge", "own", "--events-limit", "2", "--own-samples", "1",
                   "--out", str(tmp_path)]) == 0
    s = json.loads((tmp_path / "tiny_summary.json").read_text())
    assert s["judge"]["passes"] == 4 and s["own"]["responses"] >= 1
    rows = [json.loads(x) for x in open(tmp_path / "tiny_judge.jsonl")]
    assert all(abs(r["p1"] + r["p2"] - 1) < 1e-5 for r in rows)
    own = [json.loads(x) for x in open(tmp_path / "tiny_own.jsonl")]
    assert all(len(r["readout_logprobs"]) > 0 for r in own)
