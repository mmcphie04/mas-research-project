"""Within-response curve inputs (mad.curves): answer-line removal, the paper's character cuts,
saved-token-id construction with validated boundaries, bridge / exact-replay / re-tokenized
variants, eligibility -- and, on a tiny real model, that exact replay and the f = 0 bridge
reproduce what the original readout computed."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mad import curves as C


class CharTok:
    """One token per character: every cut is a token boundary."""

    all_special_ids: list[int] = []

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


class PieceTok:
    """SentencePiece-like: a token is optional whitespace plus a run of non-space characters, so
    a space belongs to the FOLLOWING word. ``merge`` glues listed phrases into one token, which
    makes a cut inside them inexact."""

    all_special_ids = [0]

    def __init__(self, merge=()):
        self.vocab, self.inv, self.merge = {"<eos>": 0}, {0: "<eos>"}, merge

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab)
            self.inv[self.vocab[piece]] = piece
        return self.vocab[piece]

    def encode(self, s, add_special_tokens=False):
        pat = "|".join([*(re.escape(m) for m in self.merge), r"\s*\S+", r"\s+"])
        return [self._id(p) for p in re.findall(pat, s)]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inv[i] for i in ids if not (skip_special_tokens and i == 0))


PROMPT = "<user>question<model>\n"
TEXT = "First I compute the torque. Then the twist angle is small.\n\nAnswer: (C)\nConfidence: 80"


def test_reasoning_end_drops_answer_and_confidence_lines():
    end, status = C.reasoning_end(TEXT, 10)
    assert status == "ok" and TEXT[:end] == "First I compute the torque. Then the twist angle is small."
    t2 = "Reasoning.\n**Answer: (B)**\nConfidence: 7"
    assert t2[: C.reasoning_end(t2, 10)[0]] == "Reasoning."
    assert C.reasoning_end("No answer line here.", 10) == (None, "answer_missing")
    # the LAST answer field decides (as the parser's selection rule)
    t3 = "Answer: (A) was my first thought.\nMore reasoning.\nAnswer: (D)"
    assert t3[: C.reasoning_end(t3, 10)[0]].endswith("More reasoning.")


def test_cut_chars_is_the_papers_word_boundary_rule():
    s = "alpha beta gamma delta"
    assert C.cut_chars(s, 1.0) == len(s)
    assert C.cut_chars(s, 0.0) == 0
    n = int(len(s) * 0.5)  # 11 -> last space before 11 is at 10
    assert C.cut_chars(s, 0.5) == s.rfind(" ", 0, n) == 10
    assert C.cut_chars("nospaceshere", 0.5) == 0


@pytest.mark.parametrize("tok", [CharTok(), PieceTok()])
def test_standard_bridge_and_replay_inputs(tok):
    gen = tok.encode(TEXT)
    b = C.build_items(tokenizer=tok, prompt=PROMPT, text=TEXT, gen_ids=gen + ([0] if isinstance(tok, PieceTok) else []),
                      n_options=10)
    assert b.status == "ok" and b.replay_exact is True
    p_ids = tok.encode(PROMPT)
    reasoning = TEXT[: b.reasoning_chars]
    std = {it.fraction: it for it in b.items if it.variant == "standard"}
    assert set(std) == set(C.FRACS)
    assert std[0.0].ids == p_ids + tok.encode("Answer: (")  # nothing written: no separator
    assert std[1.0].ids == p_ids + gen[: std[1.0].body_tokens] + tok.encode("\n\nAnswer: (")
    for f, it in std.items():
        assert it.boundary_exact, f
        assert tok.decode(gen[: it.body_tokens]) == reasoning[: C.cut_chars(reasoning, f)]
    lens = [std[f].body_tokens for f in C.FRACS]
    assert lens == sorted(lens)
    br = {it.fraction: it for it in b.items if it.variant == "bridge"}
    assert br[0.0].ids == p_ids + tok.encode("\nFinal answer: (")  # == the saved exposure input
    assert br[1.0].ids == p_ids + gen[: br[1.0].body_tokens] + tok.encode("\nFinal answer: (")
    (ex,) = [it for it in b.items if it.variant == "exact_replay"]
    k = ex.body_tokens
    assert ex.ids == p_ids + gen[:k] and "C" in tok.decode(gen[k:k + 1])  # next token holds the letter
    assert tok.decode(gen[:k]).rstrip().endswith(("Answer:", "Answer: ("))  # stops before the letter


def test_inexact_boundary_is_detected_not_hidden():
    tok = PieceTok(merge=(" compute the",))  # one token spans a space a cut can land on
    text = "We compute the torque now.\nAnswer: (A)"
    gen = tok.encode(text)
    b = C.build_items(tokenizer=tok, prompt=PROMPT, text=text, gen_ids=gen, n_options=4)
    reasoning = text[: b.reasoning_chars]
    flagged = [it for it in b.items if it.variant == "standard" and it.boundary_exact is False]
    assert flagged, "a cut inside a merged token must be flagged"
    for it in flagged:  # the body never runs past the character cut
        assert len(tok.decode(gen[: it.body_tokens])) <= C.cut_chars(reasoning, it.fraction)


def test_saved_ids_are_preserved_even_when_reencoding_differs():
    tok = CharTok()
    saved = tok.encode(TEXT)
    b = C.build_items(tokenizer=tok, prompt=PROMPT, text=TEXT, gen_ids=saved, n_options=10,
                      variants=("standard", "retokenized"))
    std1 = next(it for it in b.items if it.variant == "standard" and it.fraction == 1.0)
    ret1 = next(it for it in b.items if it.variant == "retokenized" and it.fraction == 1.0)
    assert std1.ids[len(tok.encode(PROMPT)):][: std1.body_tokens] == saved[: std1.body_tokens]
    assert tok.decode(ret1.ids) == PROMPT + TEXT[: b.reasoning_chars] + "\n\nAnswer: ("


def test_replay_mismatch_is_recorded():
    tok = CharTok()
    b = C.build_items(tokenizer=tok, prompt=PROMPT, text=TEXT, gen_ids=tok.encode(TEXT + "x"), n_options=10)
    assert b.replay_exact is False


def test_letter_token_mismatch_skips_exact_replay():
    tok = CharTok()
    b = C.build_items(tokenizer=tok, prompt=PROMPT, text=TEXT, gen_ids=tok.encode(TEXT), n_options=10,
                      stated_letter_id=ord("Z"))
    (ex,) = [it for it in b.items if it.variant == "exact_replay"]
    assert ex.ids == [] and ex.note == "letter_token_mismatch"


def test_eligibility_keeps_capped_responses_with_stated_answers():
    assert C.eligibility("stated", "stated") == "eligible"
    assert C.eligibility("readout_fallback", "stated") == "own_answer_readout_fallback"
    assert C.eligibility("stated", "readout_fallback") == "previous_answer_readout_fallback"
    assert C.eligibility("stated", None, revision=False) == "eligible"
    # a capped response is built normally whenever its answer field parses
    capped = "Long reasoning that ran on and on.\nAnswer: (B)\nConfidence: 5"
    b = C.build_items(tokenizer=CharTok(), prompt=PROMPT, text=capped, gen_ids=CharTok().encode(capped),
                      n_options=4)
    assert b.status == "ok"


def test_outcome_kinds():
    assert C.outcome_kind(1, 1, 1) == "unchanged_correct"
    assert C.outcome_kind(1, 1, 2) == "unchanged_wrong"
    assert C.outcome_kind(1, 2, 2) == "helpful"
    assert C.outcome_kind(2, 1, 2) == "harmful"
    assert C.outcome_kind(1, 3, 2) == "wrong_to_wrong"


def test_stub_letter_readout_is_a_function_of_the_sequence():
    from mad.backends.stub import StubBackend
    from mad.config import load_config

    b = StubBackend(load_config("configs/base.yaml", "configs/smoke_stub.yaml"), {})
    out = b.letter_logprobs_ids([[1, 2, 3], [1, 2, 3], [1, 2, 4]], [4, 4, 4])
    np.testing.assert_array_equal(out[0], out[1])
    assert not np.allclose(out[0], out[2])
    assert all(abs(np.exp(o).sum() - 1) < 1e-5 for o in out)


HF = __import__("importlib.util", fromlist=["x"]).find_spec("transformers") is not None


@pytest.mark.skipif(not HF, reason="transformers not installed (uv sync --extra hf)")
def test_hf_exact_replay_and_bridge_reproduce_the_original_readout():
    from mad.backends.base import build_backend
    from mad.config import load_config

    backend = build_backend(load_config("configs/base.yaml", "configs/smoke_tiny.yaml",
                                        **{"activations.positions": ["readout_last", "before_stated_answer"]}))
    tok = backend.tokenizer
    prompt = backend.render([{"role": "user", "content": "Which option? (A) one (B) two (C) three (D) four"}])
    text = "Three is right because it is the only odd prime here.\n\nAnswer: (C)\nConfidence: 70"
    gen = tok.encode(text, add_special_tokens=False)
    post = backend.readout([prompt + text], [4], meta=[{"gen_start_char": len(prompt), "gen_token_ids": gen}])[0]
    expo = backend.readout([prompt], [4], meta=[{"gen_start_char": None}])[0]
    b = C.build_items(tokenizer=tok, prompt=prompt, text=text, gen_ids=gen, n_options=4,
                      stated_letter_id=backend._letter_ids[2])
    items = {(it.variant, it.fraction): it for it in b.items}
    ex, br0 = items[("exact_replay", None)], items[("bridge", 0.0)]
    lp = backend.letter_logprobs_ids([ex.ids, br0.ids], [4, 4], batch_size=2)
    np.testing.assert_allclose(lp[0], post.extra_logprobs["before_stated_answer"], atol=2e-4)
    np.testing.assert_allclose(lp[1], expo.letter_logprobs, atol=2e-4)
    # batched with a much shorter sequence (left padding) gives the same answer
    lp_b = backend.letter_logprobs_ids([items[("standard", 0.0)].ids, ex.ids], [4, 4], batch_size=2)
    np.testing.assert_allclose(lp_b[1], lp[0], atol=2e-4)


def test_forced_opening_builds_pre_and_post_opening_contexts_separately():
    tok = CharTok()
    prefill = "Decisive issue:"
    prompt = PROMPT + prefill
    text = " the torque units. Recomputed it.\n\nAnswer: (B)\nConfidence: 60"
    b = C.build_items(tokenizer=tok, prompt=prompt, text=text, gen_ids=tok.encode(text), n_options=4, prefill=prefill)
    v = {(it.variant, it.fraction): it for it in b.items}
    pre = v[("bridge_pre_opening", 0.0)]
    assert pre.ids == tok.encode(PROMPT) + tok.encode("\nFinal answer: (")  # the saved exposure context
    assert ("bridge", 0.0) not in v and ("standard", 0.0) not in v  # never pooled with ordinary curves
    assert v[("standard_after_opening", 0.0)].ids == tok.encode(prompt) + tok.encode("Answer: (")
    assert v[("bridge_after_opening", 0.0)].ids == tok.encode(prompt) + tok.encode("\nFinal answer: (")
    bad = C.build_items(tokenizer=tok, prompt=PROMPT, text=text, gen_ids=tok.encode(text), n_options=4, prefill=prefill)
    assert bad.status == "prefill_not_at_prompt_end"


@pytest.mark.skipif(not HF, reason="transformers not installed (uv sync --extra hf)")
def test_hf_fp32_head_matches_the_logits_path_in_float32_and_options_are_respected():
    from mad.backends.base import build_backend
    from mad.config import load_config

    backend = build_backend(load_config("configs/base.yaml", "configs/smoke_tiny.yaml",
                                        **{"activations.positions": ["readout_last", "before_stated_answer"]}))
    tok = backend.tokenizer
    prompt = backend.render([{"role": "user", "content": "Pick one: (A) x (B) y (C) z (D) w"}])
    seqs = [tok.encode(prompt + "I think it is the third.\n\nAnswer: (", add_special_tokens=False),
            tok.encode(prompt + "Answer: (", add_special_tokens=False)]
    a = backend.letter_logprobs_ids(seqs, [4, 4], batch_size=2, head="fp32_head")
    b = backend.letter_logprobs_ids(seqs, [4, 4], batch_size=2, head="bf16_logits")  # fp32 model on CPU
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, atol=1e-4)
        assert x.shape == (4,) and abs(np.exp(x).sum() - 1) < 1e-5
    c = backend.letter_logprobs_ids(seqs, [4, 4], batch_size=2, position_ids_from_mask=True)
    np.testing.assert_allclose(c[1], a[1], atol=1e-4)  # the unpadded (longest) sequence is unaffected
    with pytest.raises(ValueError):
        backend.letter_logprobs_ids(seqs, [4, 4], head="nope")


def test_option_letters_are_single_tokens_after_both_answer_prefixes():
    """With the cached Gemma-3-12B tokenizer: after "\\n\\nAnswer: (" and "\\nFinal answer: (" every
    option letter is exactly one token, the one the readout scores (no merge with "(")."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("google/gemma-3-12b-it", revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
                                            local_files_only=True)
    except Exception:
        pytest.skip("Gemma-3-12B tokenizer not cached")
    ids = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCDEFGHIJ"]
    for prefix in ("Some reasoning.\n\nAnswer: (", "Some reasoning.\nFinal answer: (", "<start_of_turn>model\nAnswer: ("):
        base = tok.encode(prefix, add_special_tokens=False)
        for letter, tid in zip("ABCDEFGHIJ", ids, strict=True):
            assert tok.encode(prefix + letter, add_special_tokens=False) == [*base, tid], (prefix, letter)
