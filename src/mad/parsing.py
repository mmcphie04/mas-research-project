"""Answer and confidence parsing.

Always takes the LAST match. A round-1 prompt quotes the agent's own round-0 response back to it,
so a first-match parser reads the previous answer and reports that nothing changed. That failure
is silent -- it looks like a zero revision rate, not like a crash.

Parse failures are tracked, never scored as wrong answers (docs/RISKS.md risk 19). The forced
readout pass means a parse failure never costs a row.
"""

from __future__ import annotations

import re

import numpy as np

ANSWER_RE = re.compile(r"Answer:\s*\(?([A-J])\)?", re.IGNORECASE)
CONF_RE = re.compile(r"Confidence:\s*(\d{1,3})", re.IGNORECASE)
LETTERS = "ABCDEFGHIJ"


def parse_answer(text: str, n_options: int) -> tuple[int, bool]:
    """Return (display index, parse_ok) from the last well-formed answer line."""
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return -1, False
    idx = LETTERS.index(matches[-1].upper())
    return (idx, True) if idx < n_options else (-1, False)


def parse_confidence(text: str) -> float:
    matches = CONF_RE.findall(text or "")
    if not matches:
        return float("nan")
    return float(np.clip(int(matches[-1]), 0, 100))
