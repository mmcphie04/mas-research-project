import numpy as np
from sklearn.metrics import roc_auc_score

from mad.analysis.stats import (
    auroc,
    bootstrap_auroc,
    conditional_auroc,
    delong,
    delong_paired_test,
    mcnemar,
    mcnemar_power,
    required_discordant,
)


def test_required_discordant_matches_the_planning_numbers():
    # the two figures the sample-size section is built on (docs/PLAN.md section 7)
    assert 80 <= required_discordant(0.65) <= 90
    assert 185 <= required_discordant(0.60) <= 200


def test_power_rises_with_discordant_pairs():
    assert mcnemar_power(20, 0.65) < mcnemar_power(85, 0.65) < mcnemar_power(300, 0.65)
    assert 0.75 < mcnemar_power(required_discordant(0.65), 0.65) < 0.90


def test_mcnemar_counts_only_discordant_pairs():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 1])
    b = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    res = mcnemar(a, b)
    assert res.n_discordant == 2 and res.b_favours_a == 2 and res.b_favours_b == 0
    assert res.delta == 0.25
    assert mcnemar(a, a).p_value == 1.0


def test_delong_auroc_agrees_with_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + y
    aucs, cov = delong(y, s[None, :])
    assert abs(aucs[0] - roc_auc_score(y, s)) < 1e-9
    assert cov.shape == (1, 1) and cov[0, 0] > 0


def test_delong_paired_test_detects_a_real_difference():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 600)
    strong = rng.normal(size=600) + 1.4 * y
    weak = rng.normal(size=600) + 0.2 * y
    a, b, p = delong_paired_test(y, strong, weak)
    assert a > b and p < 0.001
    _, _, p_same = delong_paired_test(y, strong, strong)
    assert p_same == 1.0


def test_conditional_auroc_is_chance_when_signal_is_group_level_only():
    # every within-group pair is uninformative because the score is constant inside a group
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(200), 3)
    y = rng.integers(0, 2, 600)
    score = np.repeat(rng.normal(size=200), 3)  # varies only across questions
    cond, n_pairs, n_groups = conditional_auroc(y, score, groups)
    assert n_pairs > 0 and n_groups > 0
    assert abs(cond - 0.5) < 0.02


def test_conditional_auroc_recovers_a_within_group_signal():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(300), 3)
    y = rng.integers(0, 2, 900)
    score = rng.normal(size=900) + 2.0 * y
    cond, _, _ = conditional_auroc(y, score, groups)
    assert cond > 0.85


def test_cluster_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(150), 3)
    y = rng.integers(0, 2, 450)
    score = rng.normal(size=450) + 0.8 * y
    ci = bootstrap_auroc(y, score, groups, n_boot=300, seed=0)
    assert ci.lo < ci.point < ci.hi
    assert abs(ci.point - auroc(y, score)) < 1e-12
    assert ci.n == 150
