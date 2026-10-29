"""Structured-debate conditions (docs/NEXT_PHASE_PLAN.md Stage 3), run as a CHILD job of an
existing parent so every condition starts from the parent's saved round 0. Tier-0 stub only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mad import prompts as P

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]
NEW = ["debate_s2", "debate_structured", "debate_structured_open", "self_structured"]


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parent(driver, out):
    assert driver.main(["--config", *STUB, "--out-root", str(out), "--name", "t", "--dataset", "synthetic",
                        "--n-questions", "6", "--chunk", "2", "--conditions", "debate"]) == 0
    return json.loads((out / "t-job.json").read_text())


def test_structured_templates_carry_the_decided_wording():
    for block in (P.STRUCTURED_PEER_BLOCK, P.STRUCTURED_SELF_BLOCK):
        assert ("Choose the answer best supported by your check. Explain the specific evidence or reasoning "
                "supporting your decision; agreement alone is insufficient.") in block
        assert "change only if" not in block.lower()
        assert block.index("Decisive issue") < block.index("Check:") < block.index("Decision:")
    assert "No substantive disagreement" in P.STRUCTURED_PEER_BLOCK and "{peers}" in P.STRUCTURED_PEER_BLOCK
    assert "{peers}" not in P.STRUCTURED_SELF_BLOCK
    assert P.OPENING_PREFILL == "Decisive issue:"
    m = P.round1_messages(base=[{"role": "user", "content": "q"}], own_response="mine", peer_responses=["a", "b"],
                          structured=True)
    assert P.BREVITY in m[-1]["content"] and P.ENDING_CONF in m[-1]["content"]
    for k in ("structured_peer_block", "structured_self_block", "opening_prefill"):
        assert k in P.TEMPLATES


def test_structured_child_job_reuses_round0_and_records_prefill(driver, tmp_path):
    from mad.job import verify_archive

    parent = _parent(driver, tmp_path)
    pick = parent["chunks"][0][:1] + parent["chunks"][2]  # skip the middle chunk entirely
    ids = tmp_path / "dev.json"
    ids.write_text(json.dumps({"question_ids": pick}))
    child = ["--config", *STUB, "--out-root", str(tmp_path), "--name", "dev", "--parent", "t",
             "--subset-ids", str(ids), "--conditions", *NEW]
    assert driver.main(child) == 0
    job = json.loads((tmp_path / "dev-job.json").read_text())
    assert [len(c) for c in job["chunks"]] == [1, 0, 2]
    assert "structured_peer_block" in job["identity"]["prompt_template_hashes"]
    k = ["question_id", "agent_id"]
    par = pd.read_parquet(tmp_path / "t-debate" / "agent_states.parquet").query("phase == 'pre'")
    for c in NEW:
        st = pd.read_parquet(tmp_path / f"dev-{c}" / "agent_states.parquet")
        assert sorted(st.question_id.unique()) == sorted(pick)
        m = st.query("phase == 'pre'").merge(par, on=k, suffixes=("", "_p"))
        assert len(m) == 3 * len(pick) and (m.answer == m.answer_p).all()  # identical round 0
        assert set(st.phase) == {"pre", "peer_exposure", "post"}
    # the second current-debate sample has its own round-1 seed
    seed = lambda d: json.loads((tmp_path / d / "manifest.json").read_text())["round1_seed"]  # noqa: E731
    assert seed("dev-debate_s2-c000") != seed("t-debate-c000")
    tr = {c: [json.loads(x) for x in open(tmp_path / f"dev-{c}" / "transcripts.jsonl")] for c in NEW}
    post = {c: [r for r in tr[c] if r["phase"] == "post"] for c in NEW}
    assert all("Decisive issue:" in r["messages"][-1]["content"] and "Panel member" in r["messages"][-1]["content"]
               for r in post["debate_structured"])
    assert all("Panel member" not in r["messages"][-1]["content"] and "Decisive issue:" in r["messages"][-1]["content"]
               for r in post["self_structured"])
    assert all("Decisive issue" not in r["messages"][-1]["content"] for r in post["debate_s2"])
    for r in post["debate_structured_open"]:
        assert r["prefill"] == "Decisive issue:" and r["prompt"].endswith("Decisive issue:")
        assert r["gen_start_char"] == len(r["prompt"])
    assert all(r["prefill"] == "" for r in post["debate_structured"])
    st = pd.read_parquet(tmp_path / "dev-debate_structured_open" / "agent_states.parquet").query("phase == 'post'")
    assert (st.freetext_parse_ok == 1).all()
    assert verify_archive(tmp_path, "dev")["views"]["self_structured"]["questions"] == len(pick)


def test_template_rule_allows_additions_but_not_changes(driver):
    old = {"system": "a", "question": "b"}
    assert driver.template_changes(old, {"system": "a", "question": "b", "new": "c"}) == []
    assert driver.template_changes(old, {"system": "a2", "question": "b"}) == ["system"]
    assert driver.template_changes(old, {"system": "a"}) == ["question"]


def test_subset_ids_must_belong_to_the_parent(driver, tmp_path):
    _parent(driver, tmp_path)
    ids = tmp_path / "bad.json"
    ids.write_text(json.dumps(["not-a-question"]))
    with pytest.raises(SystemExit, match="not in the parent job"):
        driver.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "dev", "--parent", "t",
                     "--subset-ids", str(ids), "--conditions", "debate_structured"])
