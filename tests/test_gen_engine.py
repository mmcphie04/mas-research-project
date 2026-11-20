"""Generation core on the CPU stub: seeding, thinking-cap handling, context refusal, last-answer
parsing, token accounting, and hash-checked chunk archives. No GPU, no vLLM."""

from __future__ import annotations

import json

import pytest

from mad.gen import archive as A
from mad.gen import engine as E


def reqs(n, condition="bank/test", n_options=4, messages=None):
    return [E.Request(f"q{i}|a{i % 3}", f"q{i}", i % 3, condition, n_options,
                      messages=messages or [{"role": "system", "content": "S"}, {"role": "user", "content": f"Q{i}"}])
            for i in range(n)]


def test_seed_is_deterministic_and_distinct():
    assert E.seed_for("q1", 0, "c") == E.seed_for("q1", 0, "c")
    assert len({E.seed_for("q1", a, "c") for a in range(50)}) == 50
    assert E.seed_for("q1", 0, "c") != E.seed_for("q1", 0, "d") and 0 <= E.seed_for("x", "y", "z") < 2 ** 31


def test_nonthinking_rows_parse_the_last_answer_and_count_tokens():
    eng = E.StubEngine()
    rows, calls = E.run_requests(eng, E.MODES["qwen_nt"], reqs(12))
    assert len(rows) == 12 and len(calls) == 1 and calls[0]["phase"] == "answer"
    for r in rows:
        # the stub quotes another letter first ("A peer said Answer: (X)"), then states its own; the
        # answer phase is seeded with seed + 1
        assert r["parsed"] and r["answer"] == (r["seed"] + 1) % 4
        assert r["generated_tokens"] == r["answer_tokens"] == len(r["gen_ids"]) and r["thinking_tokens"] == 0
        assert r["prompt_tokens"] == len(r["prompt_ids"]) and r["answer_finish"] == "stop"
    assert calls[0]["generated_tokens"] == sum(r["answer_tokens"] for r in rows) + 12  # + the EOS of each
    assert calls[0]["prompt_tokens"] == sum(r["prompt_tokens"] for r in rows)
    # the non-thinking Qwen template carries the empty think block in the prompt
    assert "<think>\n\n</think>" in eng.tokenizer.decode(rows[0]["prompt_ids"])


def test_thinking_caps_are_recorded_and_closed_explicitly():
    eng = E.StubEngine(trace_len=(10, 9000), eos_every=7)
    mode = E.MODES["qwen_t"]
    rows, calls = E.run_requests(eng, mode, reqs(60))
    assert [c["phase"] for c in calls] == ["think", "answer"]
    capped = [r for r in rows if r["thinking_capped"]]
    closed = [r for r in rows if r["thinking_closed"]]
    eos = [r for r in rows if r["thinking_eos"]]
    assert capped and closed and eos and len(capped) + len(closed) + len(eos) == 60
    tok = eng.tokenizer
    for r in rows:
        assert r["thinking_tokens"] <= mode.think_cap
        gen = tok.decode(r["gen_ids"])
        assert gen.count("</think>") == 1, "every trace is closed exactly once"
        assert "Answer: (" not in r["thinking_text"] and r["parsed"]
    for r in capped:
        assert r["thinking_tokens"] == mode.think_cap and r["answer_tokens"] > 0


def test_context_overflow_is_refused_not_truncated():
    eng = E.StubEngine(max_model_len=400)
    long_msg = [{"role": "user", "content": "x" * 300}]
    rows, calls = E.run_requests(eng, E.MODES["qwen_t"], reqs(3) + reqs(2, messages=long_msg))
    skipped = [r for r in rows if r["skipped"]]
    # a thinking request needs prompt + 4096 + 400 + closing tokens: none fits in 400
    assert len(skipped) == 5 and all(r["skipped"] == "context" and r["answer"] == -1 for r in skipped)
    assert calls[0]["requests"] == 0
    eng2 = E.StubEngine(max_model_len=2000)
    longer = [{"role": "user", "content": "x" * 800}]
    rows2, _ = E.run_requests(eng2, E.MODES["qwen_nt"], reqs(3) + reqs(2, messages=longer))
    assert sum(r["skipped"] == "context" for r in rows2) == 2  # only the long prompts exceed 2000 - 1280


def test_capped_answer_is_flagged():
    eng = E.StubEngine(answer_len=2000)
    rows, _ = E.run_requests(eng, E.MODES["qwen_nt"], reqs(3))
    assert all(r["answer_finish"] == "length" and not r["parsed"] and r["answer"] == -1 for r in rows)


def test_trace_only_mode_for_judges():
    eng = E.StubEngine(trace_len=(10, 3000))
    rows, calls = E.run_requests(eng, E.MODES["qwen_t_judge"], reqs(10), answer=False)
    assert [c["phase"] for c in calls] == ["think"] and all(r["answer_finish"] is None for r in rows)
    assert all(r["thinking_tokens"] <= 2000 for r in rows) and any(r["thinking_capped"] for r in rows)


def test_chunks_are_hash_checked_and_incomplete_is_reported(tmp_path):
    eng = E.StubEngine()
    rows, calls = E.run_requests(eng, E.MODES["gemma"], reqs(5))
    A.write_job(tmp_path, "jb", {"job": "jb", "parts": {"gemma": 2}})
    d = A.write_chunk(tmp_path, "jb", "gemma", 0, rows, calls, {"question_ids": ["q0"]})
    assert A.verify_job(tmp_path, "jb")["complete"] == 1
    assert A.incomplete(tmp_path, "jb") == {"missing_chunks": ["jb-gemma-c001"]}
    back = A.read_rows(tmp_path, "jb")
    assert len(back) == 5 and list(back.gen_ids.iloc[0]) == rows[0]["gen_ids"]
    assert A.read_calls(tmp_path, "jb").seconds.notna().all()
    (d / "calls.json").write_text(json.dumps([]))
    with pytest.raises(A.JobMismatch):
        A.verify_job(tmp_path, "jb")
