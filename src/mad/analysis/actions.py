"""Choosing the next action after round 0: final-panel voting with fractional outcomes.

Rules (docs/NEXT_PHASE_PLAN.md, review decision 5):

  stop         plurality of the 3 initial answers
  debate       plurality of the 3 revised answers (each agent's final answer)
  independent  plurality of all 6 independent answers (3 initial + 3 fresh samples)

Ties are broken uniformly at random and scored EXACTLY by expectation: a vote whose top count is
shared by k options scores 1/k if the answer key is among them, else 0. Every downstream quantity
-- policy training labels, accuracy, gains and losses -- is therefore fractional, and nothing
here casts outcomes to integers.

The earlier rule (``pooled``: every available answer in one vote, ties to the round-0
plurality, then agent order) is kept for reproducing the 2026-10-28 analysis. Under it a
unanimous initial panel can never change: three new answers give another option at most a 3-3
tie, which goes back to round 0.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

ACTIONS = ("stop", "independent", "debate")


def expected_vote(answers, gold: int) -> float:
    """Exact expected correctness of a plurality vote with uniformly random tie-breaking.
    Negative entries (no answer) do not vote; no votes at all scores 0."""
    c = Counter(int(a) for a in answers if int(a) >= 0)
    if not c:
        return 0.0
    top = max(c.values())
    tied = [k for k, v in c.items() if v == top]
    return (1.0 / len(tied)) if int(gold) in tied else 0.0


def pooled_vote(answers, round0, gold: int) -> float:
    """The 2026-10-28 rule, via the original ``panel_vote``: plurality over all answers, ties to
    the round-0 plurality, then agent order. Deterministic, so 0 or 1. ``round0=None`` breaks
    ties by agent order only (the old "stop" vote)."""
    from mad.analysis.labels import panel_vote

    ans = [int(a) for a in answers]
    fb = None if round0 is None else [int(a) for a in round0]
    return float(panel_vote(ans, [0.0] * len(ans), fb, "round0_then_agent0") == int(gold))


def outcomes(a0, ad, ai, gold: int, rule: str = "final_panel") -> dict:
    """Expected correctness of each action for one question."""
    if rule == "final_panel":
        return {"stop": expected_vote(a0, gold), "debate": expected_vote(ad, gold),
                "independent": expected_vote(list(a0) + list(ai), gold)}
    if rule == "pooled":
        return {"stop": pooled_vote(a0, None, gold), "debate": pooled_vote(list(a0) + list(ad), a0, gold),
                "independent": pooled_vote(list(a0) + list(ai), a0, gold)}
    raise ValueError(rule)


def fit_fractional_logistic(X: np.ndarray, y: np.ndarray, C: float = 0.3):
    """Logistic regression on labels in [0, 1]: each row enters twice, as a positive with weight
    y and a negative with weight 1 - y (the binomial likelihood of a fractional outcome). With
    0/1 labels this is ordinary logistic regression. Returns a fitted scaler+model pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=float)
    if np.any((y < 0) | (y > 1)) or not np.all(np.isfinite(y)):
        raise ValueError("fractional labels must lie in [0, 1]")
    Xs = np.vstack([X, X])
    lab = np.r_[np.ones(len(y)), np.zeros(len(y))]
    w = np.r_[y, 1 - y]
    keep = w > 0
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    m.fit(Xs[keep], lab[keep], logisticregression__sample_weight=w[keep])
    return m


def predict_expected(model, X: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """P(correct) from a fitted fractional model; a constant when training labels had no spread."""
    if model is None:
        return np.full(len(X), float(np.mean(y_train)))
    return model.predict_proba(X)[:, 1]


def learned_choice(R, feats, folds, C: float = 0.3) -> np.ndarray:
    """Per action, a fractional logistic model of expected correctness given pre-action features,
    fitted on training folds; choose the action with the highest prediction (ties -> stop)."""
    X = R[feats].fillna(R[feats].median()).to_numpy(float)
    choice = np.empty(len(R), object)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        P = []
        for a in ACTIONS:
            y = R[a].to_numpy(float)[tr]
            model = fit_fractional_logistic(X[tr], y, C) if np.ptp(y) > 0 else None
            P.append(predict_expected(model, X[te], y))
        best = np.argmax(np.vstack(P) + np.array([[1e-9], [0], [0]]), 0)
        choice[te] = [ACTIONS[i] for i in best]
    return choice


def evaluate(R, choice, name: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Fractional evaluation against always-stop: expected correct, mean difference with a
    question-bootstrap interval, expected gains and losses (sums of positive / negative
    per-question differences), and how many questions moved fully or partly."""
    choice = np.asarray(choice)
    got = np.array([float(R[c].iloc[i]) for i, c in enumerate(choice)])
    stop = R["stop"].to_numpy(float)
    d = got - stop
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"policy": name, "expected_correct": float(got.sum()), "minus_stop": float(d.mean()),
            "ci95": [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))],
            "expected_gain_vs_stop": float(d[d > 0].sum()), "expected_loss_vs_stop": float(np.abs(d[d < 0]).sum()),
            "questions_improved": int((d > 0).sum()), "questions_worsened": int((d < 0).sum()),
            "questions_partly_changed": int(((d != 0) & (np.abs(d) < 1)).sum()),
            "extra_answers_generated": int(3 * (choice != "stop").sum()),
            **{f"share_{a}": float(np.mean(choice == a)) for a in ACTIONS}}


def hindsight_best(R) -> np.ndarray:
    A = R[list(ACTIONS)].to_numpy(float)
    return np.array([ACTIONS[int(np.argmax(row + np.array([1e-9, 0, 0])))] for row in A])


def sample_size(diff: np.ndarray, deltas=(0.02, 0.03, 0.05), z=1.96 + 0.84) -> dict:
    """Questions needed for a paired mean difference of size delta (two-sided 5%, 80% power),
    using the observed per-question SD of the difference. Planning only."""
    sd = float(np.std(diff, ddof=1))
    return {"per_question_sd": sd, **{f"n_for_delta_{d}": int(np.ceil((z * sd / d) ** 2)) for d in deltas}}
