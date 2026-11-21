"""gen_run.py end to end on the CPU stub: bank -> chooser -> Experiment B -> judge confirmation, with
resumable, hash-checked chunks and a job record listing every part."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mad.gen import archive as A

ROOT = Path(__file__).resolve().parents[1]


def _gen():
    spec = importlib.util.spec_from_file_location("gen_run", ROOT / "scripts" / "gen_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pipeline_on_the_stub(tmp_path):
    G = _gen()
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "dev", "--limit", "8", "--chunk", "4"]
    for mode in ("gemma", "qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    bank = A.read_rows(tmp_path, "x")
    assert len(bank) == 8 * (10 + 10 + 10) and set(bank["mode"]) == {"gemma", "qwen_nt", "qwen_t"}
    calls = A.read_calls(tmp_path, "x")
    assert set(calls.part) == {"bank_gemma__dev", "bank_qwen_nt__dev", "bank_qwen_t__dev"} and (calls.seconds >= 0).all()
    # resume: a second run of a complete part generates nothing new
    n_dirs = len(list(tmp_path.glob("x-bank_gemma__dev-c*")))
    assert G.main(["--task", "bank", "--mode", "gemma", *base]) == 0
    assert len(list(tmp_path.glob("x-bank_gemma__dev-c*"))) == n_dirs
    for arm in ("A1_qt5", "A2_10", "A3", "A1_g5"):
        assert G.main(["--task", "choose", "--arm", arm, *base]) == 0
    ch = A.read_rows(tmp_path, "x", "choose_A3__dev")
    assert len(ch) % 3 == 0 and ch.pick_position.between(-1, 10).all()
    assert (ch.pick_answer[ch.pick_position > 0] >= 0).all()
    assert G.main(["--task", "b", "--mode", "qwen_nt", "--arms", "debate", "alone", "--n-panels", "3", *base]) == 0
    b = A.read_rows(tmp_path, "x", "b_qwen_nt_debate__dev")
    assert len(b) == 3 * len(set(b.question_id)) and set(b.meta_arm) == {"debate"}
    job = json.loads((tmp_path / "x-job.json").read_text())
    assert {"bank_gemma__dev", "choose_A3__dev", "b_qwen_nt_alone__dev"} <= set(job["parts"])
    assert job["part_info"]["choose_A3__dev"]["pool"] == [["gemma", 5], ["qwen_nt", 5]]
    assert A.verify_job(tmp_path, "x")["missing"] == [] and A.incomplete(tmp_path, "x") == {}


def test_judge_confirmation_on_the_stub(tmp_path):
    G = _gen()
    assert G.main(["--task", "judge", "--out-root", str(tmp_path), "--job", "j", "--engine", "stub", "--limit", "3"]) == 0
    tr = A.read_rows(tmp_path, "j", "judge_traces")
    own = A.read_rows(tmp_path, "j", "judge_own")
    assert len(tr) == 6 and (tr.answer_tokens == 0).all() and (tr.thinking_tokens <= 2000).all()
    assert len(own) == 9 and own.parsed.all()
    rec = json.loads((tmp_path / "j-job.json").read_text())
    assert rec["part_info"]["judge_traces"]["events_sha256"] == json.loads(
        (ROOT / "artifacts/revision_gate/judge/confirm_events.json").read_text())["sha256"]


def test_splits_and_tags_get_their_own_parts_and_resume_refuses_a_mismatch(tmp_path):
    G = _gen()
    common = ["--out-root", str(tmp_path), "--job", "s", "--engine", "stub", "--limit", "4", "--chunk", "2"]
    assert G.main(["--task", "bank", "--mode", "qwen_nt", "--split", "dev", *common]) == 0
    assert G.main(["--task", "bank", "--mode", "qwen_nt", "--split", "heldout", *common]) == 0
    dev, held = A.read_rows(tmp_path, "s", "bank_qwen_nt__dev"), A.read_rows(tmp_path, "s", "bank_qwen_nt__heldout")
    assert len(dev) == len(held) == 40 and not set(dev.question_id) & set(held.question_id)
    assert G.main(["--task", "bank", "--mode", "qwen_nt", "--split", "dev", "--tag", "pilot", *common]) == 0
    assert (tmp_path / "s-bank_qwen_nt__dev_pilot-c000").exists()
    import pytest
    with pytest.raises(SystemExit, match="holds other questions"):
        G.main(["--task", "bank", "--mode", "qwen_nt", "--split", "dev", "--out-root", str(tmp_path), "--job", "s",
                "--engine", "stub", "--limit", "6", "--chunk", "3"])  # same part name, different selection


def test_b_panels_come_from_bank_parts_only(tmp_path):
    """Chooser rows (runs numbered 0-2, mode qwen_t) must never be mistaken for bank agents."""
    G = _gen()
    base = ["--out-root", str(tmp_path), "--job", "k", "--engine", "stub", "--split", "dev", "--limit", "10", "--chunk", "10"]
    for mode in ("qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
    assert G.main(["--task", "choose", "--arm", "A2_5", "A1_qt5", *base]) == 0
    assert G.main(["--task", "b", "--mode", "qwen_t", "--arms", "debate", "--n-panels", "5", *base]) == 0
    b = A.read_rows(tmp_path, "k", "b_qwen_t_debate__dev")
    bank = A.read_rows(tmp_path, "k", "bank_qwen_t__dev")
    # every revised agent's "before" answer is its bank answer (not a chooser run's pick)
    ans = {(r.question_id, int(r.agent)): int(r.answer) for r in bank.itertuples()}
    assert all(int(r.meta_before) == ans[(r.question_id, int(r.agent))] for r in b.itertuples())
