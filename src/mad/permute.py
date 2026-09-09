"""Answer-option permutation.

One shared permutation per (question, run). All three agents see the same option ordering, so
peer messages can reference letters without translation. The permutation and its inverse are
recorded on every row (docs/RISKS.md risk 20).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

LETTERS = "ABCDEFGHIJ"


def letter_to_index(letter: str) -> int:
    return LETTERS.index(letter.upper())


def index_to_letter(idx: int) -> str:
    return LETTERS[idx]


@dataclass(frozen=True)
class Permutation:
    """``order[i]`` is the ORIGINAL index of the option now shown in display slot ``i``."""

    order: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.order)

    def display_options(self, options: list[str]) -> list[str]:
        return [options[i] for i in self.order]

    def original_to_display(self, original_idx: int) -> int:
        return self.order.index(original_idx)

    def display_to_original(self, display_idx: int) -> int:
        return self.order[display_idx]

    def as_list(self) -> list[int]:
        return list(self.order)


def make_permutation(n_options: int, question_id: str, seed: int) -> Permutation:
    """Deterministic per-(question, seed) permutation.

    Derived from a hash of the question id so that adding or reordering questions in the sample
    does not change any other question's permutation.
    """
    # hashlib, not builtin hash(): PYTHONHASHSEED randomises str hashing per process, which
    # would make permutations irreproducible across runs.
    digest = hashlib.sha256(f"{question_id}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return Permutation(tuple(int(i) for i in rng.permutation(n_options)))


def identity_permutation(n_options: int) -> Permutation:
    return Permutation(tuple(range(n_options)))
