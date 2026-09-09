import numpy as np

from mad.analysis.labels import build_events
from mad.analysis.oracles import (
    evaluate,
    evaluate_all,
    headroom,
    panel_optimal_accept,
    policy_accept,
    random_accept,
)


def _ev(states, specs):
    return build_events(states(specs))


def test_per_agent_oracle_never_hurts_agent_accuracy(states):
    rng = np.random.default_rng(0)
    specs = []
    for q in range(80):
        for a in range(3):
            specs.append((f"q{q}", a, int(rng.integers(10)), int(rng.integers(10)), 9))
    ev = _ev(states, specs)
    mad = evaluate(ev, policy_accept(ev, "mad_standard"))
    full = evaluate(ev, policy_accept(ev, "oracle_full"))
    assert full["agent_accuracy"] >= mad["agent_accuracy"]


def test_per_agent_oracle_is_not_a_panel_upper_bound(states):
    # gold=9. Accepting everything leaves a 1-1-1 split that the tie-break resolves to gold;
    # rejecting the two unhelpful revisions restores a wrong plurality and loses the question.
    ev = _ev(states, [("q1", 0, 1, 9, 9), ("q1", 1, 0, 2, 9), ("q1", 2, 0, 3, 9)])
    mad = evaluate(ev, policy_accept(ev, "mad_standard"))
    full = evaluate(ev, policy_accept(ev, "oracle_full"))
    opt = evaluate(ev, panel_optimal_accept(ev))
    assert mad["panel_accuracy"] == 1.0
    assert full["panel_accuracy"] == 0.0, "this is exactly why G5 uses the panel-optimal oracle"
    assert opt["panel_accuracy"] == 1.0


def test_panel_optimal_dominates_every_fixed_policy(states):
    rng = np.random.default_rng(3)
    specs = [
        (f"q{q}", a, int(rng.integers(6)), int(rng.integers(6)), 5)
        for q in range(60)
        for a in range(3)
    ]
    ev = _ev(states, specs)
    opt = evaluate(ev, panel_optimal_accept(ev))["panel_accuracy"]
    for name in ("mad_standard", "round0_plurality", "oracle_full", "oracle_reject_harmful"):
        assert opt >= evaluate(ev, policy_accept(ev, name))["panel_accuracy"] - 1e-12


def test_endpoints_match_their_definitions(states):
    ev = _ev(states, [("q1", 0, 1, 5, 5), ("q1", 1, 5, 2, 5), ("q1", 2, 1, 3, 5)])
    assert policy_accept(ev, "mad_standard").all()
    assert not policy_accept(ev, "round0_plurality").any()
    table = evaluate_all(ev).set_index("policy")
    assert table.loc["mad_standard", "acceptance_rate"] == 1.0
    assert table.loc["round0_plurality", "acceptance_rate"] == 0.0


def test_random_gate_only_touches_revised_events(states):
    ev = _ev(states, [("q1", 0, 1, 5, 5), ("q1", 1, 7, 7, 5), ("q1", 2, 1, 3, 5)])
    acc = random_accept(ev, rate=0.0, seed=0)
    assert acc[(ev.revised == 0).to_numpy()].all()
    assert not acc[(ev.revised == 1).to_numpy()].any()


def test_headroom_reports_both_ceilings_and_the_flip_rate(states):
    rng = np.random.default_rng(1)
    specs = [
        (f"q{q}", a, int(rng.integers(8)), int(rng.integers(8)), 7)
        for q in range(60)
        for a in range(3)
    ]
    h = headroom(_ev(states, specs))
    assert h["panel_oracle_panel_optimal"] >= h["panel_oracle_full"] - 1e-12
    assert h["panel_headroom_optimal"] >= 0.0
    assert 0.0 <= h["panel_flip_rate_f"] <= 1.0
