"""Shared-round-0 multi-condition collection, and the readout-suffix guard."""

import pandas as pd
import pytest
from conftest import make_cfg

from mad.data import Question
from mad.multi import adversarial_answer, run_conditions
from mad.prompts import READOUT_SUFFIX, strip_confidence

CONDS = [
    {"name": "self_reconsider", "kind": "self_reconsider", "run_id": "m-self"},
    {"name": "debate", "kind": "debate", "run_id": "m-deb", "peer_confidence": False},
    {"name": "debate_conf", "kind": "debate_conf", "run_id": "m-debc", "peer_confidence": True},
    {"name": "adversarial", "kind": "adversarial", "run_id": "m-adv"},
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
    assert set(res) == {"self_reconsider", "debate", "debate_conf", "adversarial"}
    for r in res.values():
        assert len(r.rows) == 20 * 3 * 2
        assert r.n_states == 120


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


def test_adversarial_peers_assert_one_wrong_answer_the_focal_agent_did_not_give():
    for gold in range(4):
        for pre in range(4):
            adv = adversarial_answer(gold, pre, 4)
            assert adv != gold, "adversarial peers must assert a WRONG answer"
            if gold != pre:
                assert adv != pre, "and one the focal agent is not already giving"


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
