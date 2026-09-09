import numpy as np
import pytest

from mad.analysis.labels import (
    BENEFICIAL,
    HARMFUL,
    NONE,
    UNHELPFUL,
    apply_policy,
    build_events,
    panel_vote,
)


def test_all_three_labels_and_the_no_change_case(states):
    ev = build_events(
        states(
            [
                ("q1", 0, 1, 5, 5),  # wrong -> right   beneficial
                ("q1", 1, 5, 2, 5),  # right -> wrong   harmful
                ("q1", 2, 1, 3, 5),  # wrong -> wrong   unhelpful
                ("q2", 0, 7, 7, 7),  # no change
            ]
        )
    )
    got = dict(zip(zip(ev.question_id, ev.agent_id, strict=True), ev.label, strict=True))
    assert got[("q1", 0)] == BENEFICIAL
    assert got[("q1", 1)] == HARMFUL
    assert got[("q1", 2)] == UNHELPFUL
    assert got[("q2", 0)] == NONE


def test_labels_are_single_state_correctness_by_construction(states):
    # Among revised events beneficial == post_correct and harmful == pre_correct. This is the
    # fact that rules out any non-degenerate "revision quality" label (docs/PLAN.md 3.2).
    ev = build_events(states([(f"q{i}", 0, i % 4, (i + 1) % 4, 1) for i in range(20)]))
    rev = ev[ev.revised == 1]
    assert ((rev.label == BENEFICIAL) == (rev.post_correct == 1)).all()
    assert ((rev.label == HARMFUL) == (rev.pre_correct == 1)).all()


def test_panel_vote_unanimous_and_majority(states):
    assert panel_vote([3, 3, 3], [0.1, 0.2, 0.3]) == 3
    assert panel_vote([3, 3, 7], [0.1, 0.2, 0.9]) == 3


def test_three_way_tie_falls_back_to_round0_plurality(states):
    assert panel_vote([1, 2, 3], [0.1, 0.2, 0.3], fallback=[2, 2, 9]) == 2


def test_three_way_tie_is_neutral_by_default_not_confidence_driven(states):
    # Default must NOT pick the most confident agent, or the panel metric quietly becomes a
    # confidence baseline (docs/RISKS.md risk 13).
    answers, conf = [1, 2, 3], [0.1, 0.2, 0.99]
    assert panel_vote(answers, conf, fallback=[4, 5, 6]) == 1  # agent order
    assert panel_vote(answers, conf, fallback=[4, 5, 6], tie_break="round0_then_confidence") == 3


def test_unknown_tie_break_is_rejected(states):
    with pytest.raises(ValueError, match="unknown tie_break"):
        panel_vote([1, 2, 3], [0.1, 0.2, 0.3], tie_break="coin_flip")


def test_apply_policy_rejecting_a_harmful_revision_restores_the_answer(states):
    states = states([("q1", 0, 5, 5, 5), ("q1", 1, 5, 5, 5), ("q1", 2, 5, 2, 5)])
    ev = build_events(states)
    accept_all = np.ones(len(ev), dtype=bool)
    reject_harmful = (ev.label != HARMFUL).to_numpy()
    assert apply_policy(ev, accept_all)["panel_correct"].iloc[0] == 1
    assert apply_policy(ev, reject_harmful)["agent_accuracy"].iloc[0] == 1.0


def test_unhelpful_revisions_change_no_agent_correctness_but_can_move_a_vote(states):
    # wrong -> different wrong for two agents flips a 2-1 majority onto a different wrong answer
    ev = build_events(states([("q1", 0, 1, 4, 9), ("q1", 1, 1, 4, 9), ("q1", 2, 3, 3, 9)]))
    assert (ev[ev.revised == 1].label == UNHELPFUL).all()
    accepted = apply_policy(ev, np.ones(len(ev), dtype=bool))
    rejected = apply_policy(ev, np.zeros(len(ev), dtype=bool))
    assert accepted["agent_accuracy"].iloc[0] == rejected["agent_accuracy"].iloc[0] == 0.0
    assert accepted["panel_answer"].iloc[0] != rejected["panel_answer"].iloc[0]
