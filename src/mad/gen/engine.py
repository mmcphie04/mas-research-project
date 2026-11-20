"""Fast generation (vLLM) with explicit thinking-cap handling and full cost accounting.

Rules built in (docs/EXPAB_PLAN.md, section 0):

* **Thinking caps and context.** A thinking-mode request runs in two phases: the trace (stopped
  at ``</think>`` or the thinking cap), then the visible answer. A trace that hits the cap, or
  ends without ``</think>``, is closed explicitly (``\n</think>\n\n``) and flagged; nothing is
  silently truncated. A request whose prompt plus caps would exceed ``max_model_len`` is refused
  and recorded (``skipped = "context"``), never cut.
* **Answer extraction.** The LAST stated ``Answer: (X)`` of the visible answer
  (``mad.parsing.parse_answer``); a thinking trace is never parsed for the answer.
* **Cost accounting.** Every request records prompt, thinking, answer and generated tokens and
  the generate call(s) it was part of; every call records its wall-clock seconds and token
  totals. Methods are compared at matched tokens and seconds, never at matched call counts.
* **Seeds** are derived from (question, agent, condition) and recorded per request.

The engine is pluggable: ``VLLMEngine`` (imported lazily; it lives in its own environment on the
runtime, because vLLM pins its own torch) and ``StubEngine`` (deterministic, CPU, for tests).
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field, replace

from mad.parsing import parse_answer, parse_confidence

THINK_END = "</think>"


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float = 1.0


@dataclass(frozen=True)
class Mode:
    name: str
    repo_id: str
    thinking: bool
    template_kwargs: dict | None
    sampling: Sampling
    max_tokens: int = 1280          # non-thinking: the response cap
    think_cap: int = 0              # thinking: the trace cap
    answer_cap: int = 0             # thinking: the visible-answer cap
    answer_sampling: Sampling | None = None  # thinking: sampling of the answer phase (default: sampling)


NONTHINK = Sampling(temperature=0.7, top_p=0.95, top_k=64, repetition_penalty=1.0)
THINK = Sampling(temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0)
MODES = {
    "gemma": Mode("gemma", "google/gemma-3-12b-it", False, None, NONTHINK, max_tokens=1280),
    "qwen_nt": Mode("qwen_nt", "Qwen/Qwen3-8B", False, {"enable_thinking": False}, NONTHINK, max_tokens=1280),
    "qwen_t": Mode("qwen_t", "Qwen/Qwen3-8B", True, {"enable_thinking": True}, THINK, think_cap=4096, answer_cap=400),
    # the judge confirmation (docs/REVISION_JUDGE_PLAN.md): the judge's own protocol, unchanged
    "qwen_t_judge": Mode("qwen_t_judge", "Qwen/Qwen3-8B", True, {"enable_thinking": True}, THINK, think_cap=2000,
                         answer_cap=1280, answer_sampling=NONTHINK),
}


def seed_for(question_id: str, agent, condition: str) -> int:
    """Deterministic per-request seed from (question, agent, condition), < 2**31."""
    h = hashlib.sha256(f"{question_id}|{agent}|{condition}".encode()).hexdigest()
    return int(h[:8], 16) % (2 ** 31)


@dataclass
class GenParams:
    sampling: Sampling
    max_tokens: int
    seed: int
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class GenOut:
    token_ids: list[int]
    finish: str          # "stop" | "length"
    stop_token: int | None = None


@dataclass
class Request:
    request_id: str
    question_id: str
    agent: object
    condition: str
    n_options: int
    messages: list[dict] | None = None
    prompt_ids: list[int] | None = None   # instead of messages: an exact token prefix
    meta: dict = field(default_factory=dict)


class Engine:
    """``tokenizer`` (HF-compatible: apply_chat_template, encode, decode, convert_tokens_to_ids,
    eos ids) and ``generate``. ``max_model_len``: the context limit requests are checked against."""

    tokenizer = None
    max_model_len: int = 0
    eos_ids: list[int] = []

    def generate(self, prompts: list[list[int]], params: list[GenParams]) -> list[GenOut]:
        raise NotImplementedError


class VLLMEngine(Engine):
    def __init__(self, mode: Mode, max_model_len: int, gpu_memory_utilization: float = 0.90,
                 revision: str | None = None, enable_prefix_caching: bool = True):
        from vllm import LLM  # noqa: PLC0415 -- the runtime's separate vLLM environment only

        self.llm = LLM(model=mode.repo_id, revision=revision, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, seed=0,
                       enable_prefix_caching=enable_prefix_caching)
        self.tokenizer = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        from transformers import GenerationConfig  # noqa: PLC0415

        try:
            eos = GenerationConfig.from_pretrained(mode.repo_id, revision=revision).eos_token_id
        except OSError:
            eos = None
        eos = eos if isinstance(eos, list) else [eos]
        self.eos_ids = sorted({int(t) for t in [*eos, self.tokenizer.eos_token_id] if t is not None})

    def generate(self, prompts, params):
        from vllm import SamplingParams  # noqa: PLC0415
        from vllm.inputs import TokensPrompt  # noqa: PLC0415

        sps = [SamplingParams(n=1, temperature=p.sampling.temperature, top_p=p.sampling.top_p, top_k=p.sampling.top_k,
                              repetition_penalty=p.sampling.repetition_penalty, max_tokens=p.max_tokens, seed=p.seed,
                              stop_token_ids=list(p.stop_token_ids), skip_special_tokens=False, detokenize=False)
               for p in params]
        outs = self.llm.generate([TokensPrompt(prompt_token_ids=list(x)) for x in prompts], sps, use_tqdm=False)
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append(GenOut(list(c.token_ids), "length" if c.finish_reason == "length" else "stop",
                              c.stop_reason if isinstance(c.stop_reason, int) else None))
        return res


# ----------------------------------------------------------------------------- the stub (CPU tests)


class StubTokenizer:
    """Character-level tokenizer with a few special tokens; a minimal chat template."""

    SPECIAL = {"</think>": 1, "<think>": 2, "<eos>": 3}

    def __init__(self):
        self.inv = {v: k for k, v in self.SPECIAL.items()}
        self.eos_token_id = 3
        self.pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out = []
        for part in re.split(r"(</think>|<think>|<eos>)", text):
            if part in self.SPECIAL:
                out.append(self.SPECIAL[part])
            else:
                out.extend(ord(c) + 10 for c in part)
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        s = []
        for i in ids:
            if i in self.inv:
                if not skip_special_tokens:
                    s.append(self.inv[i])
            elif i >= 10:
                s.append(chr(i - 10))
        return "".join(s)

    def convert_tokens_to_ids(self, tok):
        return self.SPECIAL[tok]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        s = "".join(f"<|{m['role']}|>{m['content']}\n" for m in messages)
        if add_generation_prompt:
            s += "<|assistant|>"
            if enable_thinking is False:
                s += "<think>\n\n</think>\n\n"
        return s


class StubEngine(Engine):
    """Deterministic fake generation. A thinking-phase call (stop ids include </think>) writes a
    trace whose length is seeded: when it exceeds the call's max_tokens the output is cut at the
    cap (finish "length"), and every ``eos_every``-th seed ends with EOS instead of </think>. An
    answer-phase call writes reasoning that QUOTES a different letter first, then its own last
    line ``Answer: (X)`` -- the parser must take the last one. ``answer_len`` > max_tokens gives a
    capped answer."""

    def __init__(self, n_letters: int = 4, trace_len=(50, 6000), eos_every: int = 0, answer_len: int = 0,
                 max_model_len: int = 100000, seconds_per_token: float = 0.0):
        self.tokenizer = StubTokenizer()
        self.n_letters, self.trace_len, self.eos_every = n_letters, trace_len, eos_every
        self.answer_len, self.max_model_len, self.spt = answer_len, max_model_len, seconds_per_token
        self.eos_ids = [3]
        self.calls = 0

    def generate(self, prompts, params):
        self.calls += 1
        out = []
        for _p, prm in zip(prompts, params, strict=True):
            t = self.tokenizer
            if 1 in prm.stop_token_ids:  # thinking phase
                n = prm.seed % (self.trace_len[1] - self.trace_len[0]) + self.trace_len[0]
                body = t.encode("<think>\n" + "r" * n)
                if len(body) + 2 > prm.max_tokens:
                    out.append(GenOut(body[: prm.max_tokens], "length"))
                elif self.eos_every and prm.seed % self.eos_every == 0:
                    out.append(GenOut(body + [3], "stop", 3))
                else:
                    out.append(GenOut(body + t.encode("\n</think>"), "stop", 1))
                continue
            letter = "ABCDEFGHIJ"[prm.seed % self.n_letters]
            quoted = "ABCDEFGHIJ"[(prm.seed + 1) % self.n_letters]
            text = f"A peer said Answer: ({quoted}). I disagree.\nAnswer: ({letter})\nConfidence: {50 + prm.seed % 50}"
            if self.answer_len:
                text = "x" * self.answer_len + text
            ids = t.encode(text)
            if len(ids) > prm.max_tokens:
                out.append(GenOut(ids[: prm.max_tokens], "length"))
            else:
                out.append(GenOut(ids + [3], "stop", 3))
        return out


# ----------------------------------------------------------------------------- running requests


def render_ids(engine: Engine, mode: Mode, req: Request) -> list[int]:
    if req.prompt_ids is not None:
        return list(req.prompt_ids)
    text = engine.tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True,
                                                **(mode.template_kwargs or {}))
    return list(engine.tokenizer.encode(text, add_special_tokens=False))


def budget(mode: Mode, close_len: int = 8) -> int:
    """Tokens a request may generate beyond its prompt (both phases, plus the closing tokens)."""
    return mode.think_cap + close_len + mode.answer_cap if mode.thinking else mode.max_tokens


def _strip_eos(ids: list[int], eos: list[int]) -> list[int]:
    ids = list(ids)
    while ids and ids[-1] in eos:
        ids.pop()
    return ids


def run_requests(engine: Engine, mode: Mode, reqs: list[Request], answer: bool = True,
                 call_prefix: str = "") -> tuple[list[dict], list[dict]]:
    """Generate every request (one engine call per phase). Returns (rows, calls).

    ``answer=False`` (thinking mode only): stop after the closed trace (e.g. a judge whose verdict
    is read out by a forced-choice readout instead of being generated)."""
    tok = engine.tokenizer
    calls: list[dict] = []
    rows: list[dict] = []
    prompts = [render_ids(engine, mode, r) for r in reqs]
    need = budget(mode)
    ok = [i for i, p in enumerate(prompts) if len(p) + need <= engine.max_model_len]
    skipped = set(range(len(reqs))) - set(ok)

    def call(phase, idx, seqs, params):
        t0 = time.perf_counter()
        outs = engine.generate(seqs, params) if seqs else []
        secs = time.perf_counter() - t0
        cid = f"{call_prefix}{phase}-{len(calls)}"
        calls.append({"call_id": cid, "phase": phase, "requests": len(seqs), "seconds": round(secs, 4),
                      "prompt_tokens": int(sum(len(s) for s in seqs)),
                      "generated_tokens": int(sum(len(o.token_ids) for o in outs))})
        return cid, outs

    seeds = {i: seed_for(r.question_id, r.agent, r.condition) for i, r in enumerate(reqs)}
    think = {}
    if mode.thinking:
        end = tok.convert_tokens_to_ids(THINK_END)
        cid, outs = call("think", ok, [prompts[i] for i in ok],
                         [GenParams(mode.sampling, mode.think_cap, seeds[i], [end, *engine.eos_ids]) for i in ok])
        nl, close = tok.encode("\n\n"), tok.encode("\n" + THINK_END)
        for i, o in zip(ok, outs, strict=True):
            ids = list(o.token_ids)
            # vLLM may or may not return the stop token itself; either way it is a natural close
            closed = end in ids or o.stop_token == end
            trace = ids[: ids.index(end)] if end in ids else _strip_eos(ids, engine.eos_ids)
            eos_end = (not closed) and o.finish == "stop"
            capped = (not closed) and o.finish == "length"
            cont = trace + ([end] if closed else close) + nl
            think[i] = {"trace": trace, "cont": cont, "closed": closed, "capped": capped, "eos": eos_end, "call": cid}
    ans = {}
    if answer:
        seqs = [prompts[i] + (think[i]["cont"] if mode.thinking else []) for i in ok]
        cap = mode.answer_cap if mode.thinking else mode.max_tokens
        samp = (mode.answer_sampling or mode.sampling) if mode.thinking else mode.sampling
        cid, outs = call("answer", ok, seqs, [GenParams(samp, cap, seeds[i] + 1, list(engine.eos_ids)) for i in ok])
        for i, o in zip(ok, outs, strict=True):
            ans[i] = {"ids": _strip_eos(o.token_ids, engine.eos_ids), "finish": o.finish, "call": cid}
    for i, r in enumerate(reqs):
        th, an = think.get(i), ans.get(i)
        text = tok.decode(an["ids"], skip_special_tokens=True) if an else ""
        idx, parsed = parse_answer(text, r.n_options) if an else (-1, False)
        rows.append({
            "request_id": r.request_id, "question_id": r.question_id, "agent": str(r.agent), "condition": r.condition,
            "mode": mode.name, "seed": seeds[i], "n_options": r.n_options,
            "skipped": "context" if i in skipped else None,
            "prompt_tokens": len(prompts[i]),
            "thinking_tokens": len(th["trace"]) if th else 0,
            "answer_tokens": len(an["ids"]) if an else 0,
            "generated_tokens": (len(th["trace"]) if th else 0) + (len(an["ids"]) if an else 0),
            "thinking_capped": bool(th and th["capped"]), "thinking_closed": bool(th and th["closed"]),
            "thinking_eos": bool(th and th["eos"]),
            "answer_finish": an["finish"] if an else None,
            "answer": int(idx) if parsed else -1, "parsed": bool(parsed),
            "confidence": float(parse_confidence(text)) if an else float("nan"),
            "text": text, "thinking_text": tok.decode(th["trace"], skip_special_tokens=True) if th else "",
            "prompt_ids": prompts[i], "gen_ids": (th["cont"] if th else []) + (an["ids"] if an else []),
            "think_call": th["call"] if th else None, "answer_call": an["call"] if an else None,
            **{f"meta_{k}": v for k, v in r.meta.items()}})
    return rows, calls


def with_caps(mode: Mode, **kw) -> Mode:
    return replace(mode, **kw)
