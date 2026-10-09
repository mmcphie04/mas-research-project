"""Backend interface shared by all three tiers.

A conformance suite (tests/test_backends.py) runs against every implementation, so the tier-0
stub cannot quietly pass tests the real generation path would fail (docs/RISKS.md risk 17).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

Message = dict[str, str]


@dataclass
class Generation:
    text: str
    n_prompt_tokens: int
    n_completion_tokens: int
    token_ids: list[int] | None = None  # generated ids, padding removed; None where not applicable
    stop_reason: str | None = None  # "stop_token" | "length"; None where not applicable


@dataclass
class Readout:
    """One canonical activation state plus the forced-choice answer distribution."""

    hidden: np.ndarray  # (n_layers + 1, d_model), float16, at the readout position
    letter_logprobs: np.ndarray  # (n_options,), float32, log-softmax over option letters
    answer_index: int  # argmax over letter_logprobs -- never fails
    n_tokens: int
    # Hidden states at further token positions (mad.config.POSITIONS), when requested. A position
    # that does not exist for this text (no stated answer line) is absent from the dict.
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return float(np.exp(self.letter_logprobs.max()))


@dataclass
class OracleEntry:
    """Tier-0 only: ground truth handed to the stub at CONSTRUCTION, never per call."""

    gold_display_index: int
    n_options: int


class Backend(ABC):
    name: str
    n_layers: int
    d_model: int

    @abstractmethod
    def render(self, messages: list[Message], add_generation_prompt: bool = True) -> str:
        """Flatten a chat into the exact string the model will condition on."""

    @abstractmethod
    def generate(
        self, prompts: list[str], seed: int, meta: list[dict] | None = None
    ) -> list[Generation]:
        """Sample a continuation for each prompt. ``meta`` carries identifiers only."""

    @abstractmethod
    def readout(
        self, texts: list[str], n_options: list[int], meta: list[dict] | None = None
    ) -> list[Readout]:
        """Forced-format readout pass over completed transcripts (prefill only, no sampling)."""

    def plan_answer(self, **kwargs) -> int | None:
        """Tier-0 hook. Real backends return None and let sampling decide the answer."""
        return None

    def describe(self) -> dict:
        """Identity of the model and tokenizer, for the job manifest. Real backends extend this."""
        return {"backend": self.name, "n_layers": self.n_layers, "d_model": self.d_model}


def build_backend(cfg, oracle: dict[str, OracleEntry] | None = None) -> Backend:
    """Construct the backend named by ``cfg.model.backend``.

    ``oracle`` is accepted only by the stub; passing it to a real backend is an error, which is
    what keeps ground truth out of the real generation path.
    """
    kind = cfg.model.backend
    if kind == "stub":
        from mad.backends.stub import StubBackend

        return StubBackend(cfg, oracle or {})
    if oracle is not None:
        raise ValueError(f"backend {kind!r} must never be given an oracle table")
    if kind == "hf":
        from mad.backends.hf import HFBackend

        return HFBackend(cfg)
    if kind == "vllm":
        from mad.backends.vllm_backend import VLLMBackend

        return VLLMBackend(cfg)
    raise ValueError(f"unknown backend {kind!r}")
