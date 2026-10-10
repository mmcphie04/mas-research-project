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


def _tiny(positions, **kw):
    return build_backend(load_config("configs/base.yaml", "configs/smoke_tiny.yaml",
                                     **{"activations.positions": positions, **kw}))


HF_AVAILABLE = __import__("importlib.util", fromlist=["x"]).find_spec("transformers") is not None
POS = ["readout_last", "generation_last", "stated_answer", "before_stated_answer"]


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_positions_land_on_the_right_tokens_and_skip_peer_answers():
    backend = _tiny(POS)
    prompt = backend.render(_messages())
    # the prompt part holds a PEER's answer line; only the agent's own response may be used
    text = prompt + "Peer said:\nAnswer: (A)\n" + "My reasoning.\nAnswer: (C)\nConfidence: 70"
    gen_start = text.index("My reasoning")
    r = backend.readout([text], [4], meta=[{"gen_start_char": gen_start}])[0]
    tok = backend.tokenizer
    ids = (tok.encode(text[:gen_start], add_special_tokens=False)
           + tok.encode(text[gen_start:], add_special_tokens=False)
           + tok.encode("\nFinal answer: (", add_special_tokens=False))
    pos = r.alignment["positions"]
    assert all(ids[j] == t for j, t in pos.values()), "recorded token ids disagree with the input"
    assert tok.decode([ids[pos["stated_answer"][0]]]).strip() == "C"
    assert pos["before_stated_answer"][0] == pos["stated_answer"][0] - 1
    assert tok.decode([ids[pos["before_stated_answer"][0]]]).strip().endswith("(")
    assert tok.decode([ids[pos["generation_last"][0]]]).strip() == "0"  # Gemma splits digits
    assert set(r.extra) == {"generation_last", "stated_answer", "before_stated_answer"}
    assert r.extra["stated_answer"].shape == r.hidden.shape
    for lp in r.extra_logprobs.values():
        assert abs(np.exp(lp).sum() - 1) < 1e-4 and lp.shape == (4,)


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_missing_positions_are_reported_not_substituted():
    backend = _tiny(POS)
    prompt = backend.render(_messages())
    text = prompt + "Answer: (B) was my peer's view.\nI cannot decide."
    r = backend.readout([text], [4], meta=[{"gen_start_char": text.index("I cannot")}])[0]
    assert "stated_answer" not in r.extra and "before_stated_answer" not in r.extra
    assert r.alignment["missing"]["stated_answer"] == "answer_missing"
    r2 = backend.readout([prompt], [4], meta=[{"gen_start_char": None}])[0]
    assert r2.alignment["missing"]["stated_answer"] == "no_generated_response"


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_letter_probabilities_from_hidden_match_the_models_own_logits():
    """The per-position option probabilities use only the letter rows of the output projection;
    at the readout position they must equal what the model's full forward pass gives."""
    import torch

    backend = _tiny(["readout_last"])
    text = backend.render(_messages()) + "Answer: (B)\nConfidence: 60"
    r = backend.readout([text], [4])[0]
    enc = backend.tokenizer([text + "\nFinal answer: ("], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = backend.model(**enc, output_hidden_states=True, logits_to_keep=1, use_cache=False)
    mine = backend.letter_logprobs_from_hidden(out.hidden_states[-1][0, -1], 4)
    assert np.allclose(mine, r.letter_logprobs, atol=1e-4), (mine, r.letter_logprobs)


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_generation_records_and_replay_is_exact():
    backend = _tiny(POS, **{"model.max_new_tokens": 12})
    prompt = backend.render(_messages())
    g = backend.generate([prompt], seed=0)[0]
    assert g.stop_reason in ("stop_token", "length") and g.token_ids and len(g.token_ids) <= 12
    r = backend.readout([prompt + g.text], [4],
                        meta=[{"gen_start_char": len(prompt), "gen_token_ids": g.token_ids}])[0]
    assert r.alignment["replay_exact"] is True
    # the readout must feed exactly the generated ids (minus the final stop token)
    gen = [t for t in g.token_ids if t not in backend._stop_ids]
    P = r.alignment["n_prompt_tokens"]
    assert r.alignment["n_response_tokens"] == len(gen)
    seq_ids = backend.tokenizer.encode(prompt, add_special_tokens=False) + gen
    assert len(seq_ids) == P + len(gen)


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_readout_does_not_merge_the_response_end_with_the_suffix():
    """A response ending in a newline, re-tokenised with the suffix's leading newline, can become
    one merged token; building the input from ids keeps the generated tokens intact."""
    backend = _tiny(["readout_last", "generation_last"])
    tok = backend.tokenizer
    prompt = backend.render(_messages())
    gen = tok.encode("Answer: (B)\n", add_special_tokens=False)
    r = backend.readout([prompt + tok.decode(gen)], [4],
                        meta=[{"gen_start_char": len(prompt), "gen_token_ids": gen}])[0]
    j, tid = r.alignment["positions"]["generation_last"]
    assert tid == gen[-1] and r.alignment["replay_exact"] is True


@pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed (uv sync --extra hf)")
def test_hf_response_logprobs_match_full_logits():
    """The chunked response log-likelihood must equal what full-sequence logits give."""
    import torch

    backend = _tiny(["readout_last"], **{"model.max_new_tokens": 10})
    prompt = backend.render(_messages())
    g = backend.generate([prompt], seed=1)[0]
    r = backend.readout([prompt + g.text], [4],
                        meta=[{"gen_start_char": len(prompt), "gen_token_ids": g.token_ids}])[0]
    gen = [t for t in g.token_ids if t not in backend._stop_ids]
    pid = backend.tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([pid + gen])
    with torch.no_grad():
        logits = backend.model(input_ids=ids).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    ref = torch.stack([lp[len(pid) - 1 + k, t] for k, t in enumerate(gen)]).numpy()
    assert abs(r.alignment["response_logprob_mean"] - ref.mean()) < 1e-3
    assert abs(r.alignment["response_logprob_min"] - ref.min()) < 1e-3
