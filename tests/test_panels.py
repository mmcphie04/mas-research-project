"""Main-collection design (docs/NEXT_PHASE_PLAN.md, 2026-10-29 evening) on the tier-0 stub: a
pool of 10 independent answers, nested 3- and 5-agent panels, ordinary and structured debate
matched in initial answers / peer texts / peer order, synchronized second rounds with an
explicit history policy, and the versioned 250-word / 1,280-token protocol."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from mad import prompts as P
from mad.config import load_config

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]
CONDS = ["pool", "debate3", "structured3", "debate5", "structured5", "debate3_r2", "structured5_r2", "self_structured5"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    out = tmp_path_factory.mktemp("main")
    assert drv.main(["--config", *STUB, "--out-root", str(out), "--name", "m", "--dataset", "synthetic",
                     "--n-questions", "4", "--chunk", "2", "--conditions", *CONDS,
                     "--set", "debate.n_agents=10", "debate.brevity_words=250", "model.max_new_tokens=1280"]) == 0
    return drv, out


def _rows(out, c):
    return pd.read_parquet(out / f"m-{c}" / "agent_states.parquet")


def _tr(out, c, phase="post"):
    return {(str(r["question_id"]), r["agent_id"]): r for r in map(json.loads, open(out / f"m-{c}" / "transcripts.jsonl"))
            if r["phase"] == phase}


def test_pool_and_nested_panels_share_identical_initial_answers(run):
    _, out = run
    pool = _rows(out, "pool")
    assert set(pool.phase) == {"pre"} and (pool.groupby("question_id").size() == 10).all()
    base = pool.set_index(["question_id", "agent_id"]).answer
    for c, k in (("debate3", 3), ("structured3", 3), ("debate5", 5), ("structured5", 5), ("self_structured5", 5)):
        r = _rows(out, c)
        assert (r.groupby("question_id").size() == 3 * k).all()
        assert set(r.agent_id) == set(range(k))  # agents 0..k-1: 3-panel nested in 5-panel nested in pool
        pre = r[r.phase == "pre"].set_index(["question_id", "agent_id"]).answer
        assert (pre == base.loc[pre.index]).all()


def test_five_agent_panels_show_four_peers_matched_across_prompts(run):
    _, out = run
    for k, (a, b) in ((3, ("debate3", "structured3")), (5, ("debate5", "structured5"))):
        ta, tb = _tr(out, a), _tr(out, b)
        assert set(ta) == set(tb)
        for key in ta:
            sa, sb = ta[key]["shown_peer_sources"].split(","), tb[key]["shown_peer_sources"].split(",")
            assert sa == sb and len(sa) == k - 1 and f"agent{key[1]}" not in sa
            blk = lambda r: r["messages"][-1]["content"].split("\n\nReconsider the question in light")[0].split("\n\nBefore giving")[0]  # noqa: E731
            assert blk(ta[key]) == blk(tb[key])  # identical peer texts in identical order
            assert all(int(s.replace("agent", "")) < k for s in sa)


def test_second_round_history_policy_and_order(run):
    _, out = run
    r1, r2 = _tr(out, "debate3"), _tr(out, "debate3_r2")
    rows1 = _rows(out, "debate3").query("phase == 'post'").set_index(["question_id", "agent_id"])
    rows2 = _rows(out, "debate3_r2").query("phase == 'post'").set_index(["question_id", "agent_id"])
    for key, t in r2.items():
        prev = r1[key]
        assert t["messages"][: len(prev["messages"])] == prev["messages"]  # own full transcript kept
        assert t["messages"][len(prev["messages"])] == {"role": "assistant", "content": prev["text"].strip()}
        assert t["shown_peer_sources"] == prev["shown_peer_sources"]  # same peer order as round 1
        last = t["messages"][-1]["content"]
        for s in prev["shown_peer_sources"].split(","):  # peers' ROUND-1 responses are what is shown
            peer_r1 = P.strip_confidence(r1[(key[0], int(s.replace("agent", "")))]["text"])
            assert peer_r1 in last
        assert rows2.loc[key].previous_round_answer == rows1.loc[key].answer
    man = json.loads((out / "m-structured5_r2-c000" / "manifest.json").read_text())
    assert man["round"] == 2 and man["panel"] == 5 and man["history_policy"] == P.HISTORY_POLICY


def test_longer_response_protocol_is_versioned_and_applied_everywhere(run):
    _, out = run
    for c in ("pool", "debate5", "structured3", "debate3_r2", "self_structured5"):
        for phase in ("pre", "post"):
            for r in _tr(out, c, phase).values():
                text = " ".join(m["content"] for m in r["messages"])
                assert P.BREVITY_250 in text and P.BREVITY not in text, (c, phase)
    job = json.loads((out / "m-job.json").read_text())
    assert job["identity"]["config"]["debate"]["brevity_words"] == 250
    assert job["identity"]["config"]["model"]["max_new_tokens"] == 1280


def test_archive_verification_counts_rows_per_condition(run):
    from mad.job import verify_archive

    _, out = run
    rep = verify_archive(out, "m")
    assert rep["views"]["pool"]["questions"] == 4 and set(rep["views"]) == set(CONDS)


def test_legacy_protocol_keeps_its_identity_and_validates():
    cfg = load_config(*STUB)
    assert "brevity_words" not in cfg.to_dict()["debate"]  # every existing job identity unchanged
    with pytest.raises(ValueError, match="brevity_words"):
        load_config(*STUB, **{"debate.brevity_words": 200})
    assert P.brevity_text(150) == P.BREVITY and "brevity_250" in P.TEMPLATES


def test_a_second_round_must_follow_its_first_round(run, tmp_path):
    drv, _ = run
    with pytest.raises(SystemExit, match="listed before"):
        drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "x", "--dataset", "synthetic",
                  "--n-questions", "2", "--chunk", "2", "--conditions", "debate3_r2", "debate3",
                  "--set", "debate.n_agents=10"])


def test_worst_case_lengths_grow_with_panel_and_round(run):
    drv, _ = run
    from mad.backends.stub import StubBackend
    from mad.data import synthetic_questions

    cfg = load_config(*STUB, **{"debate.n_agents": 10, "model.max_new_tokens": 1280})
    qs = synthetic_questions(4, seed=1)
    b = StubBackend(cfg, {})
    legacy = drv.coverage(load_config(*STUB), b, qs)
    assert legacy["panel_max"] == 3 and legacy["rounds_max"] == 1
    big = drv.coverage(cfg, b, qs, [{"kind": "debate_structured", "panel": 5, "round": 2}])
    assert big["worst_round1_prompt_tokens"] >= max(legacy["round0_prompt_tokens"].values()) + 6 * 1280


def test_panel_events_categories_and_fractional_scores():
    from mad.analysis import panels as PN

    # correct minority (gold 2 held by one of three) becomes the plurality
    e = PN.panel_events([1, 1, 2], [2, 2, 1], gold=2)
    assert e["acc_before"] == 0 and e["acc_after"] == 1 and e["minority_correct_recovered"] == 1
    assert e["helpful"] == 2 and e["harmful"] == 1
    # a correct decision lost to a three-way split: fractional, not rounded
    e = PN.panel_events([2, 2, 1], [2, 3, 1], gold=2)
    assert e["correct_decision_lost"] == 1 and e["acc_after"] == pytest.approx(1 / 3)
    # a correct answer absent from the starting panel appears
    e = PN.panel_events([0, 1, 3], [2, 1, 3], gold=2)
    assert e["absent_correct_found"] == 1 and e["gold_present_initially"] == 0


def test_independent_curve_uses_nested_prefixes_and_matched_baselines():
    from mad.analysis import panels as PN

    pool = {"q": [1, 2, 2, 1, 1, 3, 3, 3, 3, 1]}
    cur = PN.independent_curve(pool, {"q": 3}).set_index("k").acc
    assert cur[3] == 0 and cur[6] == pytest.approx(0.0) and cur[9] == 1.0  # first k answers only
    out = PN.compare(pool, {"q": 1}, {"debate3": (3, {"q": [1, 1, 1]})}, n_boot=50)
    d = out["debate"]["debate3"]
    assert "vs_independent_6" in d and d["acc_after"] == 1.0 and out["independent_accuracy"][3] == 0.0


def test_second_round_on_a_prespecified_subset_uses_the_parents_first_round(run, tmp_path):
    drv, out = run
    job = json.loads((out / "m-job.json").read_text())
    pick = [job["chunks"][0][0], job["chunks"][1][1]]
    ids = tmp_path / "sub.json"
    ids.write_text(json.dumps({"question_ids": pick}))
    assert drv.main(["--config", *STUB, "--out-root", str(out), "--name", "m_r2", "--parent", "m",
                     "--subset-ids", str(ids), "--conditions", "structured5_r2"]) == 0
    r1, r2 = _tr(out, "structured5"), _tr(out, "structured5_r2")
    kid = {(str(r["question_id"]), r["agent_id"]): r
           for r in map(json.loads, open(out / "m_r2-structured5_r2" / "transcripts.jsonl")) if r["phase"] == "post"}
    assert {k[0] for k in kid} == set(pick) and len(kid) == 2 * 5
    for k, t in kid.items():
        assert t["messages"][: len(r1[k]["messages"])] == r1[k]["messages"]
        assert t["shown_peer_sources"] == r1[k]["shown_peer_sources"]
    assert set(r2) >= set(kid)  # the parent's own full second round is untouched
