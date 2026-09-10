"""Readout-only collection, exercised against tier 0 so it needs no model and no network."""

import pandas as pd
import pytest
from conftest import make_cfg

from mad.collect import STATES_FILE, collect_states
from mad.data import Question, load_mc_dataset  # noqa: F401  (loader imported for the api check)
from mad.provenance import COMPLETION, MANIFEST, is_complete
from mad.store import ActivationStore


def _questions(n=25, n_options=4):
    return [
        Question(
            question_id=f"arc-{i}",
            subject="science",
            question=f"Question {i}?",
            options=tuple(f"opt{j}" for j in range(n_options)),
            gold_index=i % n_options,
        )
        for i in range(n)
    ]


def test_one_state_per_question_variant(tmp_path):
    cfg = make_cfg(tmp_path, run_id="collect")
    qs = _questions(25)
    res = collect_states(cfg, qs, n_variants=3)
    assert res.n_states == 75 == len(res.rows)
    assert set(res.rows.agent_id) == {0, 1, 2}
    assert res.rows.question_id.nunique() == 25


def test_variants_get_different_permutations(tmp_path):
    cfg = make_cfg(tmp_path, run_id="perm")
    res = collect_states(cfg, _questions(30), n_variants=3)
    per_q = res.rows.groupby("question_id")["permutation"].nunique()
    assert per_q.mean() > 1.5, "permutation variants should mostly differ within a question"


def test_activations_align_with_state_rows(tmp_path):
    cfg = make_cfg(tmp_path, run_id="align2")
    res = collect_states(cfg, _questions(20), n_variants=2)
    store = ActivationStore(res.run_dir)
    assert len(store) == len(res.rows)
    keys_store = set(zip(store.index.question_id, store.index.agent_id, strict=True))
    keys_rows = set(zip(res.rows.question_id, res.rows.agent_id, strict=True))
    assert keys_store == keys_rows
    assert store.load_layer(0).shape == (len(res.rows), cfg.stub.d_model)


def test_correctness_and_gold_are_consistent(tmp_path):
    cfg = make_cfg(tmp_path, run_id="gold")
    res = collect_states(cfg, _questions(30), n_variants=2)
    assert (res.rows["correct"] == (res.rows.readout_answer == res.rows.gold_display)).all()
    assert res.rows.gold_display.between(0, 3).all()
    assert res.rows.readout_answer.between(0, 3).all()


def test_provenance_written(tmp_path):
    cfg = make_cfg(tmp_path, run_id="prov2")
    res = collect_states(cfg, _questions(10), n_variants=2)
    assert (res.run_dir / MANIFEST).exists() and (res.run_dir / COMPLETION).exists()
    assert is_complete(res.run_dir)
    assert (res.run_dir / STATES_FILE).exists()
    assert pd.read_parquet(res.run_dir / STATES_FILE).shape[0] == 20


def test_real_backend_is_still_refused_an_oracle(tmp_path):
    cfg = make_cfg(tmp_path, run_id="noleak")
    cfg.model.backend = "hf"
    with pytest.raises(Exception):  # noqa: B017 - either build failure or the oracle guard
        collect_states(cfg, _questions(2), n_variants=1)
