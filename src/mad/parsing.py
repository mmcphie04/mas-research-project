"""Answer and confidence parsing: a strict, line-anchored grammar.

The stated answer defines the primary correctness labels, so the parser must never turn prose
into a letter. The earlier regex had neither a line anchor nor a boundary after the letter, and
read "Answer: cannot determine." as C, "Answer: Because..." as B and "Answer: Analysis..." as A.

Grammar. An ANSWER FIELD is a whole line of the form

    [markdown]  Answer [markdown] :  [markdown]  ( X )  [markdown or "."]

where X is ONE UPPERCASE letter (any of A-Z: a letter beyond the question's options is still
recognised as an answer field, and then makes the response invalid). Leading list/quote/emphasis markup (``*``, ``_``, ``#``,
``>``, spaces) and trailing emphasis or a full stop are allowed, nothing else. This documented
form has status ``strict``. Two further forms are accepted with status ``permissive``, because
they are unambiguous:

    Answer: X            (no brackets; the letter ends the line)
    Answer: (X) text...  / Answer: X) text...   (a closing bracket follows the letter)

Rejected: a lowercase letter, a letter followed by more letters ("Because"), a letter followed by
a space and words ("A careful look shows"), anything not at the start of a line (a quoted
"... the format Answer: (B) ..." inside a sentence), and letters beyond the question's options.

Selection rule. The LAST answer field in the agent's OWN response decides. If that field names a
letter outside the question's options, the response has no valid answer (status
``out_of_range``) -- an earlier field is not used instead. Callers must pass only the agent's own
generated text: a round-1 transcript also contains its earlier answer and its peers' answers.

Confidence lines follow the same rules: ``Confidence: NN`` on its own line, last one wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

LETTERS = "ABCDEFGHIJ"

_PRE = r"^[ \t>*_#-]*"
_KEY = r"(?i:answer)[*_]*[ \t]*:[*_ \t]*"
# Any single UPPERCASE letter is syntactically an answer; whether it is a valid OPTION is decided
# afterwards. Matching only A-J let "Answer: (A)\nAnswer: (Z)" keep A, because the final (Z)
# field was invisible to the parser.
STRICT_RE = re.compile(_PRE + r"Answer[*_]*[ \t]*:[*_ \t]*\(([A-Z])\)[*_ \t.]*$", re.MULTILINE)
FIELD_RE = re.compile(
    _PRE + _KEY + r"(?:\(([A-Z])\)|([A-Z])\)|([A-Z])(?=[*_ \t.]*$))", re.MULTILINE
)
CONF_RE = re.compile(r"^[ \t>*_#-]*(?i:confidence)[*_]*[ \t]*:[*_ \t]*(\d{1,3})\b", re.MULTILINE)

STRICT, PERMISSIVE, MISSING, OUT_OF_RANGE = "strict", "permissive", "missing", "out_of_range"


@dataclass(frozen=True)
class AnswerParse:
    index: int  # display index, or -1 when there is no valid answer
    status: str  # strict | permissive | missing | out_of_range
    letter_start: int | None  # character offset of the letter in the parsed text
    n_fields: int  # how many answer fields the text contains

    @property
    def ok(self) -> bool:
        return self.status in (STRICT, PERMISSIVE)


def parse_answer_field(text: str, n_options: int) -> AnswerParse:
    """Parse the agent's own response under the grammar above."""
    fields = list(FIELD_RE.finditer(text or ""))
    if not fields:
        return AnswerParse(-1, MISSING, None, 0)
    last = fields[-1]
    group = next(g for g in (1, 2, 3) if last.group(g) is not None)
    letter, start = last.group(group), last.start(group)
    idx = ord(letter) - ord("A")
    if idx >= n_options:
        return AnswerParse(-1, OUT_OF_RANGE, start, len(fields))
    line_end = text.find("\n", last.start())
    line = text[last.start(): None if line_end == -1 else line_end]
    status = STRICT if STRICT_RE.fullmatch(line) else PERMISSIVE
    return AnswerParse(idx, status, start, len(fields))


def parse_answer(text: str, n_options: int) -> tuple[int, bool]:
    """(display index, parse_ok) -- the compact form used by row construction."""
    p = parse_answer_field(text, n_options)
    return p.index, p.ok


def parse_confidence(text: str) -> float:
    matches = CONF_RE.findall(text or "")
    if not matches:
        return float("nan")
    return float(np.clip(int(matches[-1]), 0, 100))
