"""Revision-gate helpers: event kinds and exact panel expectations, checked by hand. CPU only."""

from __future__ import annotations

import math

from mad.analysis import revision_events as R


def test_kinds():
    assert R.kind(1, 1, 0) == "unchanged" and R.kind(1, 0, 0) == "helpful"
    assert R.kind(0, 1, 0) == "harmful" and R.kind(1, 2, 0) == "wrong_wrong"


def test_panel_expectation_accept_and_reject():
    before, after, gold = [1, 0, 2], [0, 0, 1], 0      # agent 0 helpful, agent 2 wrong->wrong
    assert R.panel_expectation(before, after, gold, [0, 0, 0]) == (1 / 3, 1 / 3)   # 1,0,2: three-way tie
    assert R.panel_expectation(before, after, gold, [1, 1, 1]) == (2 / 3, 1.0)     # 0,0,1
    agent, panel = R.panel_expectation(before, after, gold, [0.5, 0.5, 0.5])       # 4 equally likely finals
    # finals: reject both [1,0,2] 1/3; accept 0 [0,0,2] 1; accept 2 [1,0,1] 0; both [0,0,1] 1
    assert math.isclose(agent, 0.5) and math.isclose(panel, (1 / 3 + 1 + 0 + 1) / 4)


def test_perfect_gate_chooses_wrong_wrong_for_the_panel():
    # agent 0 helpful (accepted); agent 2 wrong->wrong: rejecting gives [0, 1, 2] (three-way tie,
    # 1/3), accepting gives [0, 1, 1] (1 wins, 0) -> reject
    before, after, gold = [1, 1, 2], [0, 1, 1], 0
    assert R.perfect_gate(before, after, gold) == (1 / 3, 1 / 3)


def test_random_gate_expected_value():
    before, after, gold = [1, 0, 0], [0, 1, 0], 0      # agent 0 helpful, agent 1 harmful
    agent, panel = R.random_gate(before, after, gold, 0.7)
    # P(accept helpful)=.7, P(accept harmful)=.3
    exp_agent = 0.7 * 0.7 * (3 / 3) + 0.7 * 0.3 * (2 / 3) + 0.3 * 0.7 * (2 / 3) + 0.3 * 0.3 * (1 / 3)
    assert math.isclose(agent, exp_agent)
    # finals: (acc,rej)=[0,0,0]:1  (acc,acc)=[0,1,0]:1  (rej,rej)=[1,0,0]:1  (rej,acc)=[1,1,0]:0
    assert math.isclose(panel, 0.7 * 0.7 + 0.7 * 0.3 + 0.3 * 0.7)


def test_plurality_ties():
    assert R._plurality([1, 1, 2]) == 1 and R._plurality([1, 2, 3]) == -1 and R._plurality([4]) == 4
