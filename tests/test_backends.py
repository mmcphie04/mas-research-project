"""Conformance suite: every backend must satisfy the same contract."""

import numpy as np
import pytest

from mad.backends.base import OracleEntry, build_backend
from mad.config import load_config
from mad.permute import identity_permutation
from mad.prompts import round0_messages


def _messages():
    return round0_messages(
        persona="You are careful.",
        subject="math",
        question="What is 2+2?",
        options=["3", "4", "5", "6"],
        perm=identity_permutation(4),
    )


def _stub():
    cfg = load_config("configs/base.yaml", "configs/smoke_stub.yaml")
    oracle = {"q1": OracleEntry(gold_display_index=1, n_options=4)}
    return build_backend(cfg, oracle=oracle), cfg


def test_stub_declares_shape():
    backend, cfg = _stub()
    assert backend.n_layers == cfg.stub.n_layers
    assert backend.d_model == cfg.stub.d_model


def test_render_is_a_string_and_includes_the_question():
    backend, _ = _stub()
    text = backend.render(_messages())
    assert isinstance(text, str) and "2+2" in text


def test_generate_and_readout_agree_on_shape():
    backend, cfg = _stub()
    meta = [
        {
            "question_id": "q1",
            "agent_id": 0,
            "phase": "pre",
            "condition": "debate",
            "planned_answer": 2,
        }
    ]
    prompt = backend.render(_messages())
    gens = backend.generate([prompt], seed=0, meta=meta)
    assert len(gens) == 1 and gens[0].n_prompt_tokens > 0

    reads = backend.readout([prompt + gens[0].text], [4], meta=meta)
    r = reads[0]
    assert r.hidden.shape == (backend.n_layers + 1, backend.d_model)
    assert r.hidden.dtype == np.float16
    assert r.letter_logprobs.shape == (4,)
    assert 0 <= r.answer_index < 4
    assert 0.0 < r.confidence <= 1.0
    assert abs(np.exp(r.letter_logprobs).sum() - 1.0) < 1e-5


def test_readout_commits_to_the_planted_answer():
    backend, _ = _stub()
    for planned in range(4):
        meta = [
            {
                "question_id": "q1",
                "agent_id": 0,
                "phase": "pre",
                "condition": "debate",
                "planned_answer": planned,
            }
        ]
        reads = backend.readout(["irrelevant text"], [4], meta=meta)
        assert reads[0].answer_index == planned


def test_deterministic_across_calls():
    backend, _ = _stub()
    meta = [
        {
            "question_id": "q1",
            "agent_id": 0,
            "phase": "pre",
            "condition": "debate",
            "planned_answer": 1,
        }
    ]
    a = backend.readout(["x"], [4], meta=meta)[0]
    b = backend.readout(["x"], [4], meta=meta)[0]
    assert np.array_equal(a.hidden, b.hidden)


def test_real_backends_are_refused_an_oracle():
    cfg = load_config("configs/base.yaml", "configs/smoke_tiny.yaml")
    with pytest.raises(ValueError, match="never be given an oracle"):
        build_backend(cfg, oracle={"q1": OracleEntry(0, 4)})


@pytest.mark.skipif(
    pytest.importorskip("importlib.util")
    and __import__("importlib.util", fromlist=["x"]).find_spec("transformers") is None,
    reason="transformers not installed (uv sync --extra hf)",
)
def test_hf_backend_conformance():
    cfg = load_config("configs/base.yaml", "configs/smoke_tiny.yaml")
    backend = build_backend(cfg)
    assert backend.n_layers > 0 and backend.d_model > 0
    prompt = backend.render(_messages())
    gens = backend.generate([prompt], seed=0)
    reads = backend.readout([prompt + gens[0].text], [4])
    r = reads[0]
    assert r.hidden.shape == (backend.n_layers + 1, backend.d_model)
    assert r.letter_logprobs.shape == (4,)
    assert 0 <= r.answer_index < 4
