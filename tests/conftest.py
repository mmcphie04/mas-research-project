from __future__ import annotations

import numpy as np
import pytest

from mad.config import load_config
from mad.data import stratified_sample, synthetic_questions
from mad.debate import run_debate


def make_cfg(tmp_dir, run_id="t", n_questions=60, **stub_over):
    cfg = load_config("configs/base.yaml", "configs/smoke_stub.yaml")
    cfg.run_id = run_id
    cfg.out_dir = str(tmp_dir)
    cfg.data.n_questions = n_questions
    for k, v in stub_over.items():
        setattr(cfg.stub, k, v)
    return cfg


def questions_for(cfg):
    pool = synthetic_questions(max(cfg.data.n_questions * 2, 50), seed=cfg.data.sample_seed)
    return stratified_sample(pool, cfg.data.n_questions, cfg.data.sample_seed, cfg.data.subjects)


@pytest.fixture(scope="session")
def stub_run(tmp_path_factory):
    """One tier-0 collection with a strong instance-level probe signal."""
    d = tmp_path_factory.mktemp("stub_run")
    cfg = make_cfg(d, run_id="signal", n_questions=120, probe_signal=1.4, difficulty_leak=0.0)
    result = run_debate(cfg, questions_for(cfg))
    return cfg, result


@pytest.fixture(scope="session")
def difficulty_run(tmp_path_factory):
    """One tier-0 collection where the ONLY signal is question-level difficulty."""
    d = tmp_path_factory.mktemp("difficulty_run")
    # a wide difficulty band is required for the shortcut to be visible at all: with
    # p_q confined to [0.27, 0.57] even a perfect read of difficulty tops out near AUROC 0.52
    cfg = make_cfg(
        d,
        run_id="difficulty",
        n_questions=300,
        probe_signal=0.0,
        difficulty_leak=4.0,
        base_accuracy=0.5,
        accuracy_spread=0.9,
    )
    result = run_debate(cfg, questions_for(cfg))
    return cfg, result


def build_states(specs):
    """Hand-built agent-state rows. specs: list of (qid, agent, pre_answer, post_answer, gold)."""
    import pandas as pd

    rows = []
    for qid, agent, pre, post, gold in specs:
        for phase, ans in (("pre", pre), ("post", post)):
            rows.append(
                {
                    "run_id": "r",
                    "condition": "debate",
                    "question_id": qid,
                    "agent_id": agent,
                    "phase": phase,
                    "subject": "s",
                    "n_options": 10,
                    "gold_display": gold,
                    "readout_answer": ans,
                    "correct": int(ans == gold),
                    "readout_confidence": 0.5,
                    "verbalized_confidence": 50.0,
                    "peer_majority": -1,
                    "n_peers_agree_pre": 0,
                    "peer_unanimous": 0,
                    "post_equals_peer_majority": -1,
                    "rationale_chars": 10,
                    "freetext_parse_ok": 1,
                    "freetext_agrees_readout": 1,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def states():
    return build_states


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(0)
