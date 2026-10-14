"""mad.analysis.candidate_scoring: batch invariance, fold-safe revision metric, matched features,
training-only preprocessing (synthetic data; no activation archive needed)."""

import numpy as np
import pandas as pd
import pytest

from mad.analysis import candidate_scoring as CS


def synthetic(n_q=40, d=24, layers=(3, 6, 9), seed=0, missing_position=None):
    """Round 0 + one 'debate' condition, 3 agents; activations carry a weak correctness signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        gold = int(rng.integers(0, 4))
        a0 = [int(rng.integers(0, 4)) for _ in range(3)]
        for a in range(3):
            rows.append({"condition": "debate", "question_id": f"q{q}", "agent_id": a, "phase": "pre",
                         "answer": a0[a], "gold_display": gold, "correct": int(a0[a] == gold)})
        for a in range(3):
            b = a0[a] if rng.random() < 0.6 else int(rng.integers(0, 4))
            rows.append({"condition": "debate", "question_id": f"q{q}", "agent_id": a, "phase": "post",
                         "answer": b, "gold_display": gold, "correct": int(b == gold)})
    st = pd.DataFrame(rows)
    n = len(st)
    st["of_conf"] = st.correct * 0.5 + rng.normal(0, 1, n)
    st["of_len"] = rng.normal(0, 1, n)
    st["ol_readout_0"] = np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n))
    st["ot_len"] = rng.normal(0, 1, n)
    y = st.correct.to_numpy()
    arrays = {L: (rng.normal(0, 1, (n, d)) + 0.6 * y[:, None] * (np.arange(d) < 3)).astype(np.float32)
              for L in layers}
    present = np.ones(n, bool)
    if missing_position:
        present[rng.choice(n, 10, replace=False)] = False

    def layer_fn(L, pos):
        p = present if pos == missing_position else np.ones(n, bool)
        X = arrays[L].copy()
        X[~p] = 0.0
        return X, p

    pres = {pos: (present if pos == missing_position else np.ones(n, bool)) for pos in CS.POSITIONS}
    return CS.StateData(st, layer_fn, pres)


def fitted(D, train, recipe="readout_last:letter:meandiff:3-9"):
    act = CS.ActivationScorer(recipe, k_ref=3, seed=1).fit(D, train)
    out = CS.TabularScorer("out_lr[full]", ("of", "ol", "ot"), "lr", None, k_ref=3, seed=1).fit(D, train)
    comb = CS.CombinedScorer("comb", act, ("of", "ol", "ot"), None, k_ref=3, seed=1).fit(D, train)
    eq = CS.SumScorer("eq", act, out).fit(D, train)
    return {"act": act, "out": out, "comb": comb, "eq": eq}


def test_scores_do_not_depend_on_the_evaluation_batch():
    """A fitted scorer's score for one question's first answers is identical whether it is
    scored alone or together with unrelated questions and every post state."""
    D = synthetic()
    folds = CS.make_folds(D.groups, 4, 0)
    train = folds != 0
    models = fitted(D, train)
    q = D.qid[~train][0]
    alone = (D.qid == q) & (D.states.phase == "pre").to_numpy()
    batch = ~train  # every held-out question, pre and post
    for name, m in models.items():
        z_alone = m.score(D, alone)[1]
        z_batch = m.score(D, batch)[1]
        assert np.allclose(z_alone[alone], z_batch[alone]), name
    # the pre-only selector picks the same answer either way
    for name, m in models.items():
        pick_alone = int(np.nanargmax(m.score(D, alone)[1][alone]))
        pick_batch = int(np.nanargmax(m.score(D, batch)[1][alone]))
        assert pick_alone == pick_batch, name


def test_normalizers_ignore_evaluation_rows():
    D = synthetic()
    train = CS.make_folds(D.groups, 4, 0) != 0
    a = CS.ActivationScorer("readout_last:letter:meandiff:3-9", k_ref=3, seed=1).fit(D, train)
    before = (a.norm.mu, a.norm.sd, {L: (n.mu, n.sd) for L, n in a.layer_norm.items()})
    orig = D._layer_fn

    def shifted(L, pos):  # perturb only evaluation rows' activations
        X, p = orig(L, pos)
        X = X.copy()
        X[~train] += 50.0
        return X, p

    D._layer_fn, D._xcache = shifted, {}
    b = CS.ActivationScorer("readout_last:letter:meandiff:3-9", k_ref=3, seed=1).fit(D, train)
    after = (b.norm.mu, b.norm.sd, {L: (n.mu, n.sd) for L, n in b.layer_norm.items()})
    assert before == after


def test_revision_auc_is_invariant_to_rescaling_one_fold():
    """The brief's reproducer: per fold each helpful revision outranks each harmful one, so the
    metric is 1.0; pooling differences across folds gave 0.75, and 1.0 after rescaling fold 0."""
    rows = []
    for q, (kind, _fold) in enumerate([("helpful", 0), ("harmful", 0), ("helpful", 1), ("harmful", 1)]):
        pre_c, post_c = (0, 1) if kind == "helpful" else (1, 0)
        rows += [{"condition": "debate", "question_id": f"q{q}", "agent_id": 0, "phase": "pre", "answer": 0,
                  "gold_display": 0 if pre_c else 1, "correct": pre_c},
                 {"condition": "debate", "question_id": f"q{q}", "agent_id": 0, "phase": "post", "answer": 1,
                  "gold_display": 0 if pre_c else 1, "correct": post_c}]
    D = CS.StateData(pd.DataFrame(rows))
    folds = np.repeat([0, 0, 1, 1], 2)
    delta = {"q0": 2.0, "q1": 1.0, "q2": 0.2, "q3": 0.1}
    z = np.array([0.0 if ph == "pre" else delta[q] for q, ph in zip(D.qid, D.states.phase, strict=True)])
    W = CS.boot_weights(4, 0, 0)
    qindex = {q: i for i, q in enumerate(sorted(set(D.qid)))}
    r1 = CS.evaluate_scores(D, z, folds, "debate", np.array(sorted(set(D.qid))), W, qindex)
    z2 = np.where(folds == 0, 0.1 * z, z)
    r2 = CS.evaluate_scores(D, z2, folds, "debate", np.array(sorted(set(D.qid))), W, qindex)
    assert r1["metrics"]["revision_auc"]["value"] == 1.0
    assert r2["metrics"]["revision_auc"]["value"] == 1.0
    assert [c["pairs"] for c in r1["counts"]["revision_auc"]["cells"]] == [1, 1]
    # the old pooled-difference rule on the same numbers: 0.75 (helpful .2 < harmful 1)
    d_help, d_harm = np.array([2.0, 0.2]), np.array([1.0, 0.1])
    assert ((d_help[:, None] > d_harm[None, :]).mean()) == 0.75


def test_combined_model_matches_output_model_features_and_tuning():
    D = synthetic()
    train = CS.make_folds(D.groups, 4, 0) != 0
    m = fitted(D, train)
    assert m["comb"].names[:-1] == m["out"].names
    assert m["comb"].names[-1].startswith("act:")
    assert set(m["comb"].cv) == set(m["out"].cv) == set(CS.C_GRID)


def test_imputation_and_indicators_are_fitted_on_training_rows():
    M = np.array([[1.0, np.nan], [3.0, 2.0], [np.nan, 4.0], [5.0, 6.0]])
    imp = CS.Imputer().fit(M[:2])  # training rows: column 1 has a missing value, column 0 does not
    out = imp.transform(M)
    assert list(imp.ind_cols) == [1]
    assert out[2, 0] == 2.0  # evaluation-row gap in column 0 filled with the TRAINING mean
    assert out.shape[1] == 3


def test_duplicate_state_keys_are_rejected():
    st = pd.DataFrame([{"condition": "debate", "question_id": "q", "agent_id": 0, "phase": "post", "answer": 1,
                        "gold_display": 1, "correct": 1}] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        CS.StateData(st)


def test_missing_position_states_are_absent_not_imputed():
    D = synthetic(missing_position="before_stated_answer")
    train = CS.make_folds(D.groups, 4, 0) != 0
    a = CS.ActivationScorer("before_stated_answer:letter:meandiff:3-9", k_ref=3, seed=1).fit(D, train)
    z = a.score(D, ~train)[1]
    absent = ~D.present("before_stated_answer") & ~train
    assert np.isnan(z[absent]).all()
    assert np.isfinite(z[~train & D.present("before_stated_answer")]).all()


def test_tuning_preprocessing_is_fitted_inside_each_fold(monkeypatch):
    """Imputation/standardization for C tuning and references never see the held-out fold:
    only the final refit is fitted on all training rows."""
    D = synthetic()
    train = CS.make_folds(D.groups, 4, 0) != 0
    sizes = []
    orig = CS.TabularScorer._prep_fit

    def spy(self, M_rows):
        sizes.append(len(M_rows))
        return orig(self, M_rows)

    monkeypatch.setattr(CS.TabularScorer, "_prep_fit", spy)
    s = CS.TabularScorer("out", ("of", "ol", "ot"), "lr", None, k_ref=3, seed=1).fit(D, train)
    assert sizes.count(int(s.train.sum())) == 1 and len(sizes) == 3 + 1


def test_combined_reference_features_do_not_depend_on_the_held_out_fold_labels():
    """For reference fold r, the activation column of the OTHER rows comes from an activation
    scorer refitted without fold r: flipping fold r's labels must not change it."""
    D = synthetic()
    train = CS.make_folds(D.groups, 4, 0) != 0
    act = CS.ActivationScorer("readout_last:letter:meandiff:3-9", k_ref=3, seed=1).fit(D, train)
    comb = CS.CombinedScorer("comb", act, ("of", "ol", "ot"), None, k_ref=3, seed=1)
    T = train & np.isfinite(D.output(comb.sets)[0]).any(1)
    rf = CS._ref_folds(D, T, 3, 1)
    M0 = next(iter(comb._fold_matrices(D, T, rf)))[1]
    D.y = np.where(T & (rf == 0), 1 - D.y, D.y)  # flip fold 0's labels
    M1 = next(iter(comb._fold_matrices(D, T, rf)))[1]
    other = T & (rf != 0)
    assert np.allclose(M0[other, -1], M1[other, -1])


def test_aggregated_pair_bootstrap_equals_the_dense_computation():
    """The memory-light question-level aggregation gives the same values as weighting every
    pair by W[:, q_a] * W[:, q_b] (or W[:, q] within a question)."""
    rng = np.random.default_rng(5)
    nq, npairs = 30, 400
    W = CS.boot_weights(nq, 50, 0)
    qa, qb = rng.integers(0, nq, npairs), rng.integers(0, nq, npairs)
    c = rng.choice([0.0, 0.5, 1.0], npairs)
    assert np.allclose(CS.pair_rate_across(c, qa, qb, W), CS.weighted_rate(c, W[:, qa] * W[:, qb]), equal_nan=True)
    assert np.allclose(CS.pair_rate_within(c, qa, W), CS.weighted_rate(c, W[:, qa]), equal_nan=True)
