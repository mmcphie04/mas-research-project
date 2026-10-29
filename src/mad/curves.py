"""Within-response prediction curves: readout inputs at fractions of a SAVED response.

Replicates the truncation of Batra et al. (arXiv:2610.35930, section 4.1; their
``code/interp/commit_lit.py``) on our archived transcripts, without regenerating anything:

  reasoning  = the agent's own response up to the line holding its final answer field,
               right-stripped (the ``Answer: (X)`` and ``Confidence: NN`` lines are removed)
  cut(f)     = the last space before int(len(reasoning) * f) characters; f = 1 keeps it all
  input(f)   = prompt + reasoning[:cut] + "\\n\\n" + "Answer: ("   (f > 0, non-empty body)
               prompt + "Answer: ("                               (f = 0 or an empty body)

and the next-token distribution over the option letters is read at the end.

Decisions (docs/NEXT_PHASE_PLAN.md, review decisions 1-4):

* **Standardized prefix** ``Answer: (`` (the agents' own answer format) after a blank line, as
  the paper appends its own format after "\\n\\n".
* **Bridge readouts** with our existing readout suffix ``\\nFinal answer: (`` at f = 0 and
  f = 1. At f = 0 this is byte-for-byte the input of the saved ``peer_exposure`` readout.
* **Exact replay**: the original token sequence (prompt ids + generated ids) ending
  immediately before the stated answer letter -- the input whose letter distribution was saved
  as ``before_stated_answer``. It is a reproduction check, NOT the standardized 100% point; the
  two can differ in formatting and tokenization. A readout that has seen the answer is never
  used as the 100% point.
* **Token construction** keeps the SAVED generated token ids. A character cut is mapped to the
  longest prefix of generated ids whose decoding does not exceed it; whether that decoding
  equals the character cut exactly is recorded for every item (``boundary_exact``). The
  paper's re-tokenization (encode the concatenated text) is built separately
  (``variant="retokenized"``) for comparison.
* **Eligibility**: the response's own answer and the answer it revised (for revisions) must be
  stated answers. Hitting the token cap is not itself a reason for exclusion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from mad.parsing import parse_answer_field
from mad.prompts import READOUT_SUFFIX

FRACS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
STANDARD_PREFIX = "Answer: ("
SEPARATOR = "\n\n"
BRIDGE_SUFFIX = READOUT_SUFFIX  # "\nFinal answer: ("
BRIDGE_FRACS = (0.0, 1.0)


# ----------------------------------------------------------------------------- text


def reasoning_end(text: str, n_options: int) -> tuple[int | None, str]:
    """Character length of the reasoning that precedes the final answer field.

    Returns (end, "ok"), or (None, reason) when the response has no valid stated answer. The
    reasoning is everything before the LINE holding the last answer field (the parser's
    selection rule), right-stripped; text after the field (the confidence line) is dropped too.
    """
    p = parse_answer_field(text, n_options)
    if not p.ok:
        return None, f"answer_{p.status}"
    line_start = text.rfind("\n", 0, p.letter_start) + 1
    return len(text[:line_start].rstrip()), "ok"


def cut_chars(reasoning: str, f: float) -> int:
    """The paper's ``cut_words``: the last space before ``int(len * f)``; f >= 1 keeps all."""
    if f >= 1.0:
        return len(reasoning)
    n = int(len(reasoning) * f)
    j = reasoning.rfind(" ", 0, n)
    return max(j, 0)


def mentions_letter(reasoning: str, letter: str) -> bool:
    """Whether the reasoning already names the final option, e.g. "(C)" or "option C"."""
    return bool(re.search(rf"\({letter}\)|\b(?:option|answer|choice)\s+\(?{letter}\b", reasoning,
                          re.IGNORECASE))


# ----------------------------------------------------------------------------- tokens


class PrefixDecoder:
    """Decoded length of ``ids[:k]``, cached; the generated ids are never re-encoded."""

    def __init__(self, tokenizer, ids: list[int]):
        self.tok, self.ids, self._cache = tokenizer, list(ids), {0: ""}

    def text(self, k: int) -> str:
        if k not in self._cache:
            self._cache[k] = self.tok.decode(self.ids[:k], skip_special_tokens=True)
        return self._cache[k]

    def cut(self, n_chars: int) -> int:
        """Largest k with len(decode(ids[:k])) <= n_chars (binary search; decoded length is
        non-decreasing in k for ordinary text)."""
        lo, hi = 0, len(self.ids)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if len(self.text(mid)) <= n_chars:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def token_of_char(self, char: int) -> int | None:
        """Index of the token whose decoding covers character ``char`` (as the readout's
        ``_token_of_char``), or None."""
        k = self.cut(char)  # decode(ids[:k]) ends at or before char; token k starts the char
        if k >= len(self.ids):
            return None
        return k if len(self.text(k + 1)) > char else None


@dataclass
class Item:
    """One readout input."""

    variant: str  # standard | bridge | exact_replay | retokenized
    fraction: float | None
    ids: list[int]
    body_tokens: int = 0
    char_cut: int = 0
    boundary_exact: bool | None = None  # decode(body ids) == reasoning[:char_cut]
    note: str = ""


@dataclass
class Built:
    items: list[Item] = field(default_factory=list)
    status: str = "ok"
    reasoning_chars: int = 0
    mentions_final_letter: bool | None = None
    replay_exact: bool | None = None


def build_items(*, tokenizer, prompt: str, text: str, gen_ids: list[int] | None, n_options: int,
                stated_letter_id: int | None = None, revision: bool = True,
                variants: tuple[str, ...] = ("standard", "bridge", "exact_replay"),
                stop_ids: set[int] | None = None) -> Built:
    """Readout inputs for one saved response.

    ``prompt`` is the exact templated string the response was generated from (including any
    forced prefill); ``text`` the response; ``gen_ids`` the saved generated ids (stop tokens
    are dropped as the original readout did). ``revision=False`` (a first answer) has no
    exposure readout to bridge to, but its f = 0 bridge is still built.
    """
    tok = tokenizer
    end, status = reasoning_end(text, n_options)
    if end is None:
        return Built(status=status)
    reasoning = text[:end]
    out = Built(reasoning_chars=end)
    p = parse_answer_field(text, n_options)
    out.mentions_final_letter = mentions_letter(reasoning, "ABCDEFGHIJ"[p.index])
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    if gen_ids is None:  # no saved ids: fall back to encoding the response (recorded)
        gen, out.replay_exact = tok.encode(text, add_special_tokens=False), None
    else:
        gen = list(gen_ids)
        drop = (stop_ids or set()) | set(getattr(tok, "all_special_ids", []))
        while gen and gen[-1] in drop:
            gen.pop()
        out.replay_exact = tok.decode(gen, skip_special_tokens=True) == text
    dec = PrefixDecoder(tok, gen)
    std = tok.encode(SEPARATOR + STANDARD_PREFIX, add_special_tokens=False)
    std0 = tok.encode(STANDARD_PREFIX, add_special_tokens=False)
    bridge = tok.encode(BRIDGE_SUFFIX, add_special_tokens=False)

    def body(f):
        c = cut_chars(reasoning, f)
        k = dec.cut(c)
        got = dec.text(k)
        exact = got == reasoning[:c]
        # a token holding two spaces can straddle the cut: the body then stops short by
        # whitespace only, which is recorded rather than treated as an alignment failure
        note = "" if exact else ("short_by_whitespace" if reasoning[:c].startswith(got)
                                 and not reasoning[len(got):c].strip() else "boundary_mismatch")
        return c, k, exact, note

    if "standard" in variants:
        for f in FRACS:
            c, k, exact, note = body(f)
            ids = prompt_ids + (gen[:k] + std if k > 0 else std0)
            out.items.append(Item("standard", f, ids, k, c, exact, note))
    if "bridge" in variants:
        for f in BRIDGE_FRACS:
            c, k, exact, note = body(f)
            out.items.append(Item("bridge", f, prompt_ids + gen[:k] + bridge, k, c, exact, note))
    if "retokenized" in variants:  # the paper's construction: encode the concatenated text
        for f in FRACS:
            c = cut_chars(reasoning, f)
            b = reasoning[:c]
            s = prompt + (b + SEPARATOR if b else "") + STANDARD_PREFIX
            out.items.append(Item("retokenized", f, tok.encode(s, add_special_tokens=False), 0, c,
                                  None, "text re-encoded"))
    if "exact_replay" in variants:
        k = dec.token_of_char(p.letter_start)
        if k is None:
            out.items.append(Item("exact_replay", None, [], note="letter_not_aligned"))
        elif stated_letter_id is not None and gen[k] != stated_letter_id:
            out.items.append(Item("exact_replay", None, [], note="letter_token_mismatch"))
        else:
            out.items.append(Item("exact_replay", None, prompt_ids + gen[:k], k, 0, None))
    return out


# ----------------------------------------------------------------------------- eligibility


def eligibility(post_source: str, prev_source: str | None, revision: bool = True) -> str:
    """'eligible', or the exclusion reason. Stated answers only; a capped response is eligible
    whenever it still carries a valid stated answer (``answer_source == 'stated'``)."""
    if post_source != "stated":
        return f"own_answer_{post_source}"
    if revision and prev_source != "stated":
        return f"previous_answer_{prev_source}"
    return "eligible"


def outcome_kind(prev: int, post: int, gold: int) -> str:
    if prev == post:
        return "unchanged_correct" if post == gold else "unchanged_wrong"
    if post == gold:
        return "helpful"
    if prev == gold:
        return "harmful"
    return "wrong_to_wrong"
