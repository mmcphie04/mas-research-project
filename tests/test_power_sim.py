import numpy as np

from mad.analysis.labels import label_counts
from mad.analysis.power_sim import (
    attach_probe_scores,
    expected_labels,
    separation_for_auroc,
    simulate_events,
)
from mad.analysis.stats import auroc, required_discordant


def test_closed_form_label_counts_match_the_plan_table():
    # central scenario, per 1000 questions (docs/PLAN.md section 7)
    got = expected_labels(1000, p=0.42, r_C=0.14, r_I=0.28, s=0.45)
    assert abs(got["H"] - 176) < 2
    assert abs(got["B"] - 219) < 2
    assert abs(got["U"] - 268) < 2
    assert abs(got["revisions"] - 663) < 5


def test_simulated_events_reproduce_the_planted_rates():
    ev = simulate_events(4000, p=0.42, r_C=0.14, r_I=0.28, s=0.45, seed=0)
    assert abs(ev["pre_correct"].mean() - 0.42) < 0.02
    correct = ev[ev.pre_correct == 1]
    incorrect = ev[ev.pre_correct == 0]
    assert abs(correct["revised"].mean() - 0.14) < 0.02
    assert abs(incorrect["revised"].mean() - 0.28) < 0.02
    counts = label_counts(ev)
    expected = expected_labels(4000, 0.42, 0.14, 0.28, 0.45)
    for k in ("beneficial", "harmful", "unhelpful"):
        assert abs(counts[k] - expected[k[0].upper()]) / expected[k[0].upper()] < 0.10


def test_simulated_probe_hits_its_requested_auroc():
    ev = simulate_events(2000, seed=1)
    for target in (0.60, 0.70, 0.80):
        scores = attach_probe_scores(ev, target, seed=2)
        assert abs(auroc(ev["post_correct"], scores) - target) < 0.02


def test_separation_round_trips():
    for a in (0.55, 0.65, 0.75, 0.90):
        d = separation_for_auroc(a)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 20000)
        s = rng.normal(size=20000) + d * y
        assert abs(auroc(y, s) - a) < 0.01


def test_required_discordant_is_the_binding_constraint_at_realistic_flip_rates():
    # n_d ~ 0.5 * f * N_test, so f=0.20 and N_test=900 gives ~90 discordant pairs
    n_d = 0.5 * 0.20 * 900
    assert n_d >= required_discordant(0.65)
    assert n_d < required_discordant(0.60)  # a +2-point effect is NOT powered at this N
