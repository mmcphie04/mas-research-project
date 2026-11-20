"""Forced readouts and activations for generated rows (HF environment; one sequence per forward
pass, fp32 head -- G1 showed batched readouts drift). Backend-agnostic: anything with
``forward(ids, positions, layers) -> ({pos: {layer: vector}}, final_hidden)``, ``letters(final, n)``
and ``choices(final, token_ids)`` works (``HFReadouter``; a fake in the tests).

Fixed before any Experiment B data (docs/EXPAB_PLAN.md):
* LAYERS: Qwen3-8B hidden-state indices 14, 22, 31 (relative depth ~0.4, 0.6, 0.85 of 36 layers;
  index 0 is the embeddings).
* Position A ("just after reading peers"): the last token of the natural context before the agent
  writes -- the prompt, plus ``<think>\\n`` in thinking mode. The commitment readout appends
  READOUT_SUFFIX there; the same forward pass gives the activations at A.
* Position B ("end of thinking, before the answer letter"): the token immediately before the
  letter of the LAST stated ``Answer: (X)`` in the visible answer.
* Answer-prediction curves (thinking rows): the trace cut at 0/10/25/50/75/100% of its tokens,
  closed with ``\\n</think>\\n\\n``, then ``Final answer: (``; each point flags whether an option
  letter had already been mentioned in the kept trace.
"""

from __future__ import annotations

import re

import numpy as np

from mad.parsing import parse_answer_field
from mad.prompts import READOUT_SUFFIX

LAYERS = (14, 22, 31)
FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
MENTION_RE = re.compile(r"\(([A-J])\)|\b(?:option|answer|choice)\s*\(?([A-J])\b", re.IGNORECASE)
THINK_OPEN = "<think>\n"
CLOSED_EMPTY = "<think>\n\n</think>\n\n"  # the non-thinking template's empty block: a cross-mode commitment
CLOSE_THEN_ASK = "\n</think>\n\nFinal answer: ("


def ends_with(seq: list[int], tail: list[int]) -> bool:
    return len(tail) > 0 and list(seq[-len(tail):]) == list(tail)


def open_think(tok, prompt_ids) -> list[int]:
    """The prompt followed by an open think block -- unless the rendered prompt already ends with
    one (a template that opens thinking itself)."""
    p, op = list(prompt_ids), enc(tok, THINK_OPEN)
    return p if ends_with(p, op) else p + op


def trace_body(tok, trace: list[int]) -> list[int]:
    """The reasoning tokens of a trace without its leading ``<think>`` / ``<think>\n`` (if the
    model generated the opening itself)."""
    op = enc(tok, THINK_OPEN)
    if list(trace[: len(op)]) == op:
        return list(trace[len(op):])
    tid = tok.convert_tokens_to_ids("<think>")
    return list(trace[1:]) if trace and trace[0] == tid else list(trace)


def enc(tok, text: str) -> list[int]:
    return list(tok.encode(text, add_special_tokens=False))


def letter_offset(tok, answer_ids: list[int], n_options: int) -> tuple[int, bool]:
    """Index k into ``answer_ids`` of the token holding the letter of the last stated answer
    (so answer_ids[:k] ends just before it), and whether the boundary is exact. -1 if no answer."""
    text = tok.decode(answer_ids, skip_special_tokens=True)
    p = parse_answer_field(text, n_options)
    if not p.ok:
        return -1, False
    start, letter = p.letter_start, "ABCDEFGHIJ"[p.index]
    lo, hi = 0, len(answer_ids)
    while lo < hi:  # the largest k with len(decode(ids[:k])) <= start
        mid = (lo + hi + 1) // 2
        if len(tok.decode(answer_ids[:mid], skip_special_tokens=True)) <= start:
            lo = mid
        else:
            hi = mid - 1
    k = lo
    nxt = tok.decode(answer_ids[k:k + 1], skip_special_tokens=True)
    return k, nxt.strip().startswith(letter)


def judge(R, tok, row, prefill: str, choice_tokens: tuple[str, ...] = ("1", "2")) -> dict:
    ids = list(row["prompt_ids"]) + list(row["gen_ids"]) + enc(tok, prefill)
    _, final = R.forward(ids, [], [])
    lp = R.choices(final, [enc(tok, c)[0] for c in choice_tokens])
    p = np.exp(lp)
    return {"request_id": row["request_id"], "p1": float(p[0]), "p2": float(p[1]), "readout_tokens": len(ids)}


def letters_after_answer(R, tok, row) -> dict:
    ids = list(row["prompt_ids"]) + list(row["gen_ids"]) + enc(tok, READOUT_SUFFIX)
    _, final = R.forward(ids, [], [])
    return {"request_id": row["request_id"], "readout_logprobs": [round(float(x), 5) for x in R.letters(final, row["n_options"])],
            "readout_tokens": len(ids)}


def b_row(R, tok, row, thinking: bool, layers=LAYERS, curves: bool | None = None) -> tuple[dict, np.ndarray]:
    """Commitment readouts + activations at A and B, and (thinking) the answer-prediction curve.
    Returns (record, activations[2, len(layers), d]); missing B (no stated answer) is NaN.

    Commitment: ``commit_closed_logprobs`` -- the prompt + a CLOSED empty think block + READOUT_SUFFIX
    (for non-thinking rows the template already ends with that block) -- is the primary cross-mode
    measure; ``commit_logprobs`` (thinking rows: the OPEN think block, "at the start of the
    thinking"; non-thinking rows: identical to the closed one) is secondary."""
    prompt = open_think(tok, row["prompt_ids"]) if thinking else list(row["prompt_ids"])
    acts_a, final = R.forward(prompt + enc(tok, READOUT_SUFFIX), [len(prompt) - 1], layers)
    commit = R.letters(final, row["n_options"])
    if thinking:
        _, fin_c = R.forward(list(row["prompt_ids"]) + enc(tok, CLOSED_EMPTY) + enc(tok, READOUT_SUFFIX), [], [])
        commit_closed = R.letters(fin_c, row["n_options"])
    else:
        commit_closed = commit
    gen = list(row["gen_ids"])
    n_ans = int(row["answer_tokens"])
    ans_ids = gen[len(gen) - n_ans:] if n_ans else []
    k, aligned = letter_offset(tok, ans_ids, row["n_options"]) if n_ans else (-1, False)
    vec_a = np.stack([acts_a[len(prompt) - 1][layer] for layer in layers])
    vec_b = np.full_like(vec_a, np.nan)
    if k > 0:
        pre = list(row["prompt_ids"]) + gen[: len(gen) - n_ans + k]
        acts_b, _ = R.forward(pre, [len(pre) - 1], layers)
        vec_b = np.stack([acts_b[len(pre) - 1][layer] for layer in layers])
    rec = {"request_id": row["request_id"], "commit_logprobs": [round(float(x), 5) for x in commit],
           "commit_closed_logprobs": [round(float(x), 5) for x in commit_closed],
           "b_letter_offset": k, "b_letter_aligned": bool(aligned)}
    if thinking if curves is None else curves:
        # cut the reasoning BODY; the open marker is always present once (from the prompt or added),
        # so every point -- including 0% -- is a well-formed open-then-closed think block
        trace = gen[: int(row["thinking_tokens"])]
        body = trace_body(tok, trace)
        head = open_think(tok, row["prompt_ids"])
        curve, mentioned = [], []
        for f in FRACTIONS:
            kf = int(round(f * len(body)))
            seq = head + body[:kf] + enc(tok, CLOSE_THEN_ASK)
            _, fin = R.forward(seq, [], [])
            curve.append([round(float(x), 5) for x in R.letters(fin, row["n_options"])])
            mentioned.append(bool(MENTION_RE.search(tok.decode(body[:kf], skip_special_tokens=True))))
        rec.update({"curve_fractions": list(FRACTIONS), "curve_logprobs": curve, "curve_letter_mentioned": mentioned,
                    "trace_began_with_think": len(body) != len(trace)})
    return rec, np.stack([vec_a, vec_b]).astype(np.float16)


def positive_control_messages(base: list[dict], letter: str) -> list[dict]:
    """The round-0 prompt with an explicit instruction naming the letter to answer."""
    user = base[1]["content"] + (f"\n\nThis is a test of the answer interface. Whatever the content, your answer "
                                 f"must be option ({letter}).")
    return [base[0], {"role": "user", "content": user}]


class HFReadouter:
    """forward(ids, positions, layers) at batch size 1 on an HFBackend (fp32 head)."""

    def __init__(self, backend):
        self.b = backend

    def forward(self, ids, positions, layers):
        torch = self.b.torch
        x = torch.tensor([list(ids)], device=self.b.device)
        with torch.no_grad():
            res = self.b.model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True,
                               logits_to_keep=1, use_cache=False)
        hs = res.hidden_states
        acts = {p: {layer: hs[layer][0, p, :].float().cpu().numpy() for layer in layers} for p in positions}
        return acts, hs[-1][0, -1, :]

    def letters(self, final, n):
        return self.b.letter_logprobs_from_hidden(final, n)

    def choices(self, final, token_ids):
        torch = self.b.torch
        with torch.no_grad():
            W = self.b.model.get_output_embeddings().weight[list(token_ids)]
            z = W.float() @ final.float()
            return torch.log_softmax(z, dim=-1).cpu().numpy()
