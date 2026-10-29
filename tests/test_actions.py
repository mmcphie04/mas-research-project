"""Final-panel voting with exact random-tie expectations, and fractional-label policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mad.analysis import actions as AC


def test_expected_vote_exact_tie_expectation():
    assert AC.expected_vote([1, 1, 2], 1) == 1.0
    assert AC.expected_vote([1, 1, 2], 2) == 0.0
    assert AC.expected_vote([1, 2, 3], 2) == pytest.approx(1 / 3)  # three-way split
    assert AC.expected_vote([1, 1, 2, 2, 3, 3], 3) == pytest.approx(1 / 3)
    assert AC.expected_vote([1, 1, 1, 2, 2, 2], 1) == pytest.approx(0.5)  # 3-3 tie: no round-0 preference
    assert AC.expected_vote([-1, -1, 4], 4) == 1.0  # missing answers do not vote
    assert AC.expected_vote([-1, -1], 0) == 0.0


def test_unanimous_panel_can_change_under_final_panel_but_not_pooled():
    a0, ad, ai, gold = [0, 0, 0], [5, 5, 0], [5, 5, 5], 5
    fp = AC.outcomes(a0, ad, ai, gold, "final_panel")
    assert fp["stop"] == 0.0 and fp["debate"] == 1.0 and fp["independent"] == 0.5
    pooled = AC.outcomes(a0, ad, ai, gold, "pooled")
    assert pooled == {"stop": 0.0, "debate": 0.0, "independent": 0.0}  # the old rule's lock-in


def test_pooled_rule_matches_the_original_panel_vote():
    from mad.analysis.labels import panel_vote

    rng = np.random.default_rng(0)
    for _ in range(300):
        a0, ad = list(rng.integers(0, 4, 3)), list(rng.integers(0, 4, 3))
        gold = int(rng.integers(0, 4))
        old = int(panel_vote(a0 + ad, [0.0] * 6, a0, "round0_then_agent0") == gold)
        assert AC.pooled_vote(a0 + ad, a0, gold) == old


def test_fractional_logistic_accepts_fractions_and_matches_binary_fit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    yb = (X[:, 0] + rng.normal(size=200) > 0).astype(float)
    m = AC.fit_fractional_logistic(X, yb, C=0.5)
    ref = LogisticRegression(C=0.5, max_iter=5000).fit(StandardScaler().fit_transform(X), yb)
    np.testing.assert_allclose(m[-1].coef_, ref.coef_, atol=1e-3)
    yf = np.clip(yb * 0.5 + 0.25 * (X[:, 1] > 0), 0, 1)  # values 0, .25, .5, .75
    mf = AC.fit_fractional_logistic(X, yf)
    p = mf.predict_proba(X)[:, 1]
    assert np.all((p > 0) & (p < 1))
    with pytest.raises(ValueError):
        AC.fit_fractional_logistic(X, yb * 2)


def test_evaluate_keeps_fractional_outcomes():
    R = pd.DataFrame({"stop": [1 / 3, 1.0, 0.0], "independent": [0.5, 1.0, 1.0], "debate": [0.0, 0.0, 1.0]})
    e = AC.evaluate(R, ["independent", "debate", "stop"], "x", n_boot=100)
    assert e["expected_correct"] == pytest.approx(0.5 + 0.0 + 0.0)
    assert e["expected_gain_vs_stop"] == pytest.approx(0.5 - 1 / 3)
    assert e["expected_loss_vs_stop"] == pytest.approx(1.0)
    assert e["questions_partly_changed"] == 1 and e["questions_worsened"] == 1
    assert isinstance(e["expected_correct"], float)


def test_learned_choice_trains_on_fractional_labels():
    rng = np.random.default_rng(2)
    n = 120
    R = pd.DataFrame({"f": rng.normal(size=n)})
    R["stop"] = np.where(R.f > 0, 1.0, 1 / 3)
    R["independent"] = np.where(R.f > 0, 0.5, 1.0)
    R["debate"] = 0.5
    folds = np.arange(n) % 5
    ch = AC.learned_choice(R, ["f"], folds)
    got = AC.evaluate(R, ch, "learned", n_boot=50)["expected_correct"]
    assert got > R["stop"].sum() + 10  # learns to add answers only when f < 0


def test_hindsight_ties_go_to_stop():
    R = pd.DataFrame({"stop": [0.5], "independent": [0.5], "debate": [0.5]})
    assert list(AC.hindsight_best(R)) == ["stop"]
