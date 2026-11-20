"""Readout code paths on the CPU (a fake model on stub-generated rows): judge "1"/"2" readouts,
letter readouts after answers, Experiment B commitment readouts, activations at positions A and B,
answer-prediction curves, the letter-boundary finder, and the positive-control contexts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from mad.gen import archive as A
from mad.gen import readouts as RO
from mad.gen.engine import StubTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_letter_offset_finds_the_last_stated_letter():
    tok = StubTokenizer()
    text = "A peer said Answer: (B). I think otherwise.\nAnswer: (C)\nConfidence: 80"
    ids = tok.encode(text)
    k, aligned = RO.letter_offset(tok, ids, 4)
    assert aligned and tok.decode(ids[:k]).endswith("Answer: (") and tok.decode(ids[k:k + 1]) == "C"
    assert text.rfind("Answer: (C)") + len("Answer: (") == len(tok.decode(ids[:k]))
    assert RO.letter_offset(tok, tok.encode("no answer here"), 4) == (-1, False)


def test_mention_flag():
    assert RO.MENTION_RE.search("so option C looks right")
    assert RO.MENTION_RE.search("between (A) and (D)")
    assert not RO.MENTION_RE.search("compute the integral first")


def test_readouts_end_to_end_on_the_stub(tmp_path):
    G, H = _load("gen_run"), _load("hf_readouts")
    base = ["--out-root", str(tmp_path), "--job", "x", "--engine", "stub", "--split", "dev", "--limit", "6", "--chunk", "3"]
    for mode in ("qwen_nt", "qwen_t"):
        assert G.main(["--task", "bank", "--mode", mode, *base]) == 0
        assert G.main(["--task", "b", "--mode", mode, "--arms", "debate", "--n-panels", "2", *base]) == 0
    assert H.main(["--task", "b", "--out-root", str(tmp_path), "--job", "x", "--fake",
                   "--parts", "b_qwen_nt_debate__dev", "b_qwen_t_debate__dev"]) == 0
    ro_t = A.read_rows(tmp_path, "x", "b_qwen_t_debate__dev_ro")
    ro_nt = A.read_rows(tmp_path, "x", "b_qwen_nt_debate__dev_ro")
    gen = A.read_rows(tmp_path, "x", "b_qwen_t_debate__dev")
    assert len(ro_t) == len(gen) and set(ro_t.request_id) == set(gen.request_id)
    assert all(len(c) == 6 for c in ro_t.curve_logprobs) and "curve_logprobs" not in ro_nt.columns
    assert ro_t.b_letter_aligned.all() and (ro_t.b_letter_offset > 0).all()
    d = next(tmp_path.glob("x-b_qwen_t_debate__dev_ro-c*"))
    acts = np.load(d / "acts.npy")
    assert acts.shape == (len(pd.read_parquet(d / "rows.parquet")), 2, 3, 8)  # one per row: [A, B] x 3 layers
    assert acts.dtype == np.float16 and A.verify_job(tmp_path, "x")["missing"] == []
    # resume: nothing recomputed
    n = len(list(tmp_path.glob("x-b_qwen_t_debate__dev_ro-c*")))
    assert H.main(["--task", "b", "--out-root", str(tmp_path), "--job", "x", "--fake", "--parts", "b_qwen_t_debate__dev"]) == 0
    assert len(list(tmp_path.glob("x-b_qwen_t_debate__dev_ro-c*"))) == n
    assert H.main(["--task", "posctl", "--out-root", str(tmp_path), "--job", "x", "--fake", "--n-posctl", "5"]) == 0
    pc = A.read_rows(tmp_path, "x", "posctl_qwen_t")
    assert len(pc) == 5 and pc.instructed.between(0, 9).all() and "curve_logprobs" not in pc.columns


def test_judge_and_own_readouts_on_the_stub(tmp_path):
    G, H = _load("gen_run"), _load("hf_readouts")
    assert G.main(["--task", "judge", "--out-root", str(tmp_path), "--job", "j", "--engine", "stub", "--limit", "2"]) == 0
    assert H.main(["--task", "judge", "--out-root", str(tmp_path), "--job", "j", "--fake"]) == 0
    assert H.main(["--task", "own", "--out-root", str(tmp_path), "--job", "j", "--fake"]) == 0
    j = A.read_rows(tmp_path, "j", "judge_traces_ro")
    o = A.read_rows(tmp_path, "j", "judge_own_ro")
    assert len(j) == 4 and np.allclose(j.p1 + j.p2, 1) and len(o) == 6
    assert all(len(x) == y for x, y in zip(o.readout_logprobs, A.read_rows(tmp_path, "j", "judge_own").n_options, strict=True))


class Recorder:
    """Fake model that records every sequence it is given."""

    def __init__(self):
        self.seqs = []

    def forward(self, ids, positions, layers):
        self.seqs.append(list(ids))
        return {p: {layer: np.zeros(4) for layer in layers} for p in positions}, np.zeros(10)

    def letters(self, final, n):
        return np.full(n, -np.log(n))


def test_curves_and_commitments_are_well_formed():
    tok = StubTokenizer()
    prompt = tok.encode("<|user|>Q\n<|assistant|>")
    answer = tok.encode("Because.\nAnswer: (B)\nConfidence: 70")
    for generated_open in (True, False):  # the model generated "<think>\n" itself, or not
        trace = tok.encode(("<think>\n" if generated_open else "") + "step one. step two.")
        cont = trace + tok.encode("\n</think>\n\n")
        row = {"request_id": "r", "prompt_ids": prompt, "gen_ids": cont + answer, "answer_tokens": len(answer),
               "thinking_tokens": len(trace), "n_options": 4}
        R = Recorder()
        rec, _ = RO.b_row(R, tok, row, thinking=True)
        texts = [tok.decode(s) for s in R.seqs]
        curve = [t for t in texts if t.endswith("\n</think>\n\nFinal answer: (")]
        assert len(curve) == 6
        for t in curve:  # every point, including 0%, opens the think block exactly once and closes it
            body = t[len(tok.decode(prompt)):]
            assert body.startswith("<think>\n") and body.count("<think>") == 1 and body.count("</think>") == 1
        assert curve[0] == tok.decode(prompt) + "<think>\n" + "\n</think>\n\nFinal answer: ("
        assert curve[-1].endswith("step one. step two.\n</think>\n\nFinal answer: (")
        assert rec["trace_began_with_think"] is generated_open
        closed = [t for t in texts if t.endswith("<think>\n\n</think>\n\n\nFinal answer: (")]
        assert len(closed) == 1 and "commit_closed_logprobs" in rec
    # a template that already opens thinking: no second opening
    opened = prompt + tok.encode("<think>\n")
    assert RO.open_think(tok, opened) == opened and RO.open_think(tok, prompt) == opened


def test_nonthinking_commitments_coincide():
    tok = StubTokenizer()
    prompt = tok.encode("<|user|>Q\n<|assistant|><think>\n\n</think>\n\n")
    answer = tok.encode("Answer: (A)")
    R = Recorder()
    rec, _ = RO.b_row(R, tok, {"request_id": "r", "prompt_ids": prompt, "gen_ids": answer, "answer_tokens": len(answer),
                               "thinking_tokens": 0, "n_options": 4}, thinking=False)
    assert rec["commit_closed_logprobs"] == rec["commit_logprobs"] and "curve_logprobs" not in rec
