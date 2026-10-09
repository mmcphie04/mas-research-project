"""Shared-round-0 multi-condition collection, and the readout-suffix guard."""

import pandas as pd
import pytest
from conftest import make_cfg

from mad.data import Question
from mad.multi import run_chunk, run_conditions, wrong_target
from mad.prompts import READOUT_SUFFIX, strip_confidence

CONDS = [
    {"name": "self_reconsider", "kind": "self_reconsider", "run_id": "m-self"},
    {"name": "debate", "kind": "debate", "run_id": "m-deb", "peer_confidence": False},
    {"name": "debate_conf", "kind": "debate_conf", "run_id": "m-debc", "peer_confidence": True},
    {"name": "adversarial", "kind": "adversarial", "run_id": "m-adv"},
    {"name": "misleading_one", "kind": "misleading_one", "run_id": "m-mild"},
    {"name": "debate_answer_only", "kind": "debate_answer_only", "run_id": "m-ans"},
]


def _qs(n=20):
    return [Question(f"q{i}", "sci", f"Q{i}?", tuple(f"opt{j}" for j in range(4)), i % 4)
            for i in range(n)]


@pytest.fixture(scope="module")
def multi(tmp_path_factory):
    d = tmp_path_factory.mktemp("multi")
    cfg = make_cfg(d, run_id="multi", n_questions=20)
    return run_conditions(cfg, _qs(20), [dict(c) for c in CONDS]), d


def test_all_conditions_produced(multi):
    res, _ = multi
    assert set(res) == {c["name"] for c in CONDS}
    for r in res.values():
        # three states per agent now: pre, peer_exposure, post
        assert len(r.rows) == 20 * 3 * 3
        assert r.n_states == 180
        assert set(r.rows.phase) == {"pre", "peer_exposure", "post"}


def test_round0_is_byte_identical_across_conditions(multi):
    """The whole point of sharing round 0: the matched comparison is structural."""
    res, _ = multi
    ref = None
    for name, r in res.items():
        pre = r.rows[r.rows.phase == "pre"].sort_values(["question_id", "agent_id"])
        key = list(zip(pre.question_id, pre.agent_id, pre.readout_answer, strict=True))
        if ref is None:
            ref = key
        assert key == ref, f"{name} has a different round 0"


def test_conditions_actually_diverge_after_round0(multi):
    res, _ = multi
    post = {n: r.rows[r.rows.phase == "post"].sort_values(["question_id", "agent_id"])
            .readout_answer.tolist() for n, r in res.items()}
    assert post["adversarial"] != post["self_reconsider"], "adversarial peers changed nothing"


def test_peer_exposure_state_precedes_the_revision(multi):
    """The pre-revision state must be a readout on the round-1 PROMPT, with no revision text."""
    res, _ = multi
    r = res["debate"].rows
    pe = r[r.phase == "peer_exposure"]
    post = r[r.phase == "post"]
    assert len(pe) == len(post)
    # it carries no generated text of its own
    assert (pe.rationale_chars == 0).all()
    assert (pe.n_completion_tokens == 0).all()
    # but it is a real answer-bearing state
    assert pe.readout_answer.between(0, 3).all()
    assert (pe.readout_confidence > 0).all()


def test_peer_exposure_differs_from_the_committed_answer_sometimes(multi):
    res, _ = multi
    r = res["debate"].rows
    k = ["question_id", "agent_id"]
    pe = r[r.phase == "peer_exposure"].set_index(k).readout_answer
    post = r[r.phase == "post"].set_index(k).readout_answer
    assert not pe.equals(post), "pre-revision and post-revision states are identical; the readout is not being taken before generation"


def test_misleading_one_uses_a_single_dissenter(multi):
    res, _ = multi
    assert "misleading_one" in res
    post = res["misleading_one"].rows.query("phase == 'post'")
    # one fabricated dissenter plus one genuine peer, so they rarely agree with each other
    assert post.peers_agree_each_other.mean() < 0.9


def test_wrong_target_is_wrong_new_and_not_position_biased():
    from collections import Counter

    picks = Counter()
    for i in range(3000):
        gold, pre = i % 4, (i // 4) % 4
        t = wrong_target(0, "adversarial", f"q{i}", 0, gold, pre, 4)
        assert t != gold, "fabricated peers must assert a WRONG answer"
        if gold != pre:
            assert t != pre, "and one the focal agent is not already giving"
        picks[t] += 1
    # the old rule always took the lowest eligible letter, so A dominated
    assert max(picks.values()) / sum(picks.values()) < 0.3, picks
    assert wrong_target(0, "x", "q", 1, 0, 1, 4) == wrong_target(0, "x", "q", 1, 0, 1, 4)


def test_adversarial_peers_agree_with_each_other(multi):
    res, _ = multi
    post = res["adversarial"].rows[res["adversarial"].rows.phase == "post"]
    assert (post.peers_agree_each_other == 1).all()
    assert (post.peer_majority >= 0).all()


def test_peer_confidence_is_stripped_only_in_the_plain_debate_condition(multi):
    res, d = multi
    def peertext(run):
        t = pd.read_json(d / run / "transcripts.jsonl", lines=True)
        return "\n".join(t[t.phase == "pre"].text.tolist())
    assert "Confidence:" in peertext("m-deb"), "round-0 text should still record confidence"
    assert strip_confidence("Answer: (A)\nConfidence: 90") == "Answer: (A)"


def test_readout_suffix_is_appended_exactly_once(tmp_path):
    """Regression: collect_states used to pre-append the suffix that the backend also adds."""
    from mad.backends.base import build_backend
    from mad.collect import collect_states

    seen = []
    cfg = make_cfg(tmp_path, run_id="sfx")
    backend = build_backend(cfg, oracle={})
    real = backend.readout

    def spy(texts, n_options, meta=None):
        seen.extend(texts)
        return real(texts, n_options, meta=meta)

    backend.readout = spy
    oracle_qs = _qs(3)
    from mad.backends.base import OracleEntry
    backend.oracle = {q.question_id: OracleEntry(q.gold_index, 4) for q in oracle_qs}
    collect_states(cfg, oracle_qs, backend=backend, n_variants=1)
    assert seen, "readout was never called"
    for t in seen:
        assert t.count(READOUT_SUFFIX.strip()) == 0, (
            "collect_states must not pre-append the readout suffix; the backend owns it"
        )


def test_peer_order_is_randomised_and_recorded(multi):
    res, _ = multi
    post = res["misleading_one"].rows.query("phase == 'post'")
    first = post.shown_peer_sources.str.split(",").str[0]
    assert 0.2 < (first == "fabricated").mean() < 0.8, "the dissenter is always in the same slot"
    targets = post.misleading_target
    assert (targets >= 0).all() and (targets != post.gold_display).all()


def test_shown_and_panel_metadata_are_distinguished(multi):
    res, _ = multi
    self_post = res["self_reconsider"].rows.query("phase == 'post'")
    assert (self_post.shown_peer_answers == "").all() and (self_post.n_shown_agree_pre == -1).all()
    assert (self_post.n_panel_agree_pre >= 0).all(), "panel agreement must still be recorded"
    adv = res["adversarial"].rows.query("phase == 'post'")
    assert (adv.shown_peer_sources == "fabricated,fabricated").all()
    assert (adv.shown_peer_answers != adv.panel_peer_answers).any()


def test_answer_only_peers_show_no_reasoning(multi):
    res, d = multi
    t = pd.read_json(d / "m-ans" / "transcripts.jsonl", lines=True)
    user = t[t.phase == "post"].messages.map(lambda m: m[-1]["content"])
    assert user.str.contains("Panel member 1").all()
    assert not user.str.contains("Step 1:").any(), "peer reasoning leaked into answer-only"


def test_transcripts_keep_structured_messages_and_request_ids(multi):
    _, d = multi
    t = pd.read_json(d / "m-deb" / "transcripts.jsonl", lines=True)
    assert t.request_id.is_unique
    assert {"messages", "prompt", "stop_reason", "token_ids"} <= set(t.columns)
    assert "planned_answer" not in t.columns, "stub ground truth must not leak into transcripts"


def test_round1_seeds_differ_by_condition(multi):
    import json

    res, _ = multi
    seeds = {n: json.loads((r.run_dir / "manifest.json").read_text())["round1_seed"]
             for n, r in res.items()}
    assert len(set(seeds.values())) == len(seeds)


def test_retry_reuses_saved_round0_and_skips_finished_conditions(tmp_path, monkeypatch):
    """A crash after condition 1 must not regenerate round 0 for conditions 2 and 3."""
    import json

    import mad.multi as M

    cfg = make_cfg(tmp_path, run_id="r", n_questions=4)
    conds = [dict(c) for c in CONDS[:3]]
    real = M.run_condition
    calls = {"n": 0}

    def crash_on_second(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated disconnect")
        return real(*a, **k)

    monkeypatch.setattr(M, "run_condition", crash_on_second)
    with pytest.raises(RuntimeError):
        run_chunk(cfg, _qs(4), conds, round0_dir=tmp_path / "r0", job="j", chunk="c000")
    assert (tmp_path / "m-self" / "completion.json").exists()
    assert not (tmp_path / "m-deb").exists(), "a failed condition left a directory behind"
    monkeypatch.setattr(M, "run_condition", real)
    res, r0 = run_chunk(cfg, _qs(4), conds, round0_dir=tmp_path / "r0", job="j", chunk="c000")
    assert r0.loaded, "round 0 was regenerated instead of reloaded"
    pre = [r.rows.query("phase == 'pre'").sort_values(["question_id", "agent_id"]).answer.tolist()
           for r in res.values()]
    assert all(p == pre[0] for p in pre)
    man = json.loads((tmp_path / "m-deb" / "manifest.json").read_text())
    assert man["round0_reloaded"] is True


def test_saved_round0_for_other_questions_is_refused(tmp_path):
    cfg = make_cfg(tmp_path, run_id="r", n_questions=4)
    conds = [dict(CONDS[0])]
    run_chunk(cfg, _qs(4), conds, round0_dir=tmp_path / "r0", job="j", chunk="c000")
    with pytest.raises(RuntimeError, match="does not match"):
        run_chunk(cfg, _qs(5), [dict(CONDS[1])], round0_dir=tmp_path / "r0", job="j", chunk="c000")
