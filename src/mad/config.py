"""Typed configuration with deterministic hashing.

Every run's fully-resolved config is hashed into its manifest, so a result can always be
traced back to the exact settings that produced it (see docs/RISKS.md Part IV).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

Condition = str  # "debate" | "self_reconsider"
Phase = str  # "pre" | "post"


@dataclass
class ModelConfig:
    backend: str = "stub"  # stub | hf | vllm
    repo_id: str = "stub-tiny"
    revision: str | None = None
    dtype: str = "float16"
    device: str = "cpu"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 8  # generation batch
    # Readout holds hidden states for every layer and position, so its safe batch differs from
    # generation's. None means "same as batch_size".
    readout_batch_size: int | None = None
    # How the stored forced-choice letter distribution is computed (versioned like
    # debate.brevity_words): "bf16_logits" -- the model's own logits tensor, bfloat16 on GPU and so
    # quantized to 0.0625 steps at this model's letter-logit scale (the legacy path, omitted from the
    # serialized config so every existing identity is unchanged); "fp32_head" -- the final hidden
    # state at the readout position projected onto the option-letter rows in float32.
    readout_head: str = "bf16_logits"
    # Token-budget generation batching (versioned; None = legacy fixed batches of batch_size,
    # omitted from the serialized config). When set, a generation batch holds consecutive prompts
    # while count x (longest prompt in the batch + max_new_tokens) <= gen_batch_tokens, up to
    # batch_size: short round-0 prompts batch fully, long 5-agent revision prompts in twos/threes.
    gen_batch_tokens: int | None = None
    # Sampling settings passed to generate EXPLICITLY (versioned; None = not passed, so the
    # model's own generation_config.json decides -- the legacy behaviour, omitted from the
    # serialized config). Gemma-3-12B's config sets top_k 64 and no repetition penalty, so a
    # cross-model protocol sets top_k: 64, repetition_penalty: 1.0 for every model.
    top_k: int | None = None
    repetition_penalty: float | None = None
    # Extra keyword arguments for the tokenizer's chat template (versioned; None = none), e.g.
    # {"enable_thinking": false} for Qwen3.
    chat_template_kwargs: dict | None = None


@dataclass
class StubConfig:
    """Planted ground truth for the tier-0 backend.

    These parameters define, in closed form, the B/H/U distribution and the probe signal that
    the analysis layer must recover. See docs/PLAN.md section 7 for the generative model.
    """

    base_accuracy: float = 0.42  # p
    accuracy_spread: float = 0.30  # per-question difficulty band around p
    revision_rate_correct: float = 0.14  # r_C
    revision_rate_incorrect: float = 0.28  # r_I
    switch_to_correct: float = 0.45  # s
    conformity: float = 0.60  # P(switch lands on peer majority | switch, majority differs)
    control_revision_scale: float = 0.5  # self-reconsideration revises less than peer debate
    n_layers: int = 8
    d_model: int = 64
    noise_scale: float = 1.0
    probe_signal: float = 1.2  # instance-level correctness signal (std units)
    difficulty_leak: float = 0.0  # question-level signal; drives the difficulty-detector shortcut
    signal_peak_layer: float = 0.6  # fractional depth at which the signal is strongest
    peer_exposure_commit: float = 0.6  # P(pre-revision state already holds the revised answer)
    freetext_noise: float = 0.03  # P(the forced readout commits to a different letter than the prose states)
    confidence_signal: float = 0.55  # readout-confidence separation; tuned for AUROC ~0.68
    confidence_noise: float = 1.1


@dataclass
class DebateConfig:
    n_agents: int = 3
    n_rounds: int = 1
    condition: Condition = "debate"
    tie_break: str = "round0_then_agent0"  # one of mad.analysis.labels.TIE_BREAKS; validated on load
    # Which answer counts. "stated": the letter the agent wrote in its final "Answer: (X)" line --
    # the answer its peers read and the panel would count. The forced readout is kept alongside
    # as an instrument. When no stated answer parses (e.g. truncation), the readout answer is
    # used and the row is flagged answer_source="readout_fallback"; that rate is reported.
    # "readout": the forced-choice readout answer for every row (the protocol of the 1B runs).
    canonical_answer: str = "stated"
    # One persona for every agent. Distinct personas made the agents the same weights but
    # different policies, which contradicts the homogeneous-agent framing and interacted badly
    # with a tie-break that falls back to agent 0. Diversity now comes from sampling alone.
    personas: list[str] = field(default_factory=lambda: ["You are a careful, methodical reasoner."])
    peer_shows_confidence: bool = False
    ask_verbalized_confidence: bool = True
    # Response-length protocol (versioned): the word guidance in every prompt. 150 is the legacy
    # protocol (pilot, validation; with max_new_tokens 768); 250 is the longer-response protocol
    # of the main collection (with max_new_tokens 1280). The legacy value is omitted from the
    # serialized config so that every existing job identity is unchanged.
    brevity_words: int = 150


@dataclass
class DataConfig:
    dataset: str = "TIGER-Lab/MMLU-Pro"
    dataset_revision: str | None = None
    split: str = "test"
    n_questions: int = 300
    subjects: list[str] | None = None
    stratify: bool = True
    permutation_seed: int = 12345
    split_fracs: tuple[float, float, float] = (0.5, 0.2, 0.3)
    split_seed: int = 20260909
    sample_seed: int = 7


@dataclass
class ActivationConfig:
    layers: str | list[int] = "all"
    positions: list[str] = field(default_factory=lambda: ["readout_last"])
    dtype: str = "bfloat16"  # storage format (fixed; float16 overflows on Gemma's massive activations)
    shard_size: int = 500
    # positions per chunk when computing the response likelihood (memory only, not values)
    likelihood_chunk: int = 256


@dataclass
class RunConfig:
    run_id: str = "dev"
    out_dir: str = "runs"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    stub: StubConfig = field(default_factory=StubConfig)
    debate: DebateConfig = field(default_factory=DebateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    activations: ActivationConfig = field(default_factory=ActivationConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["debate"].get("brevity_words") == LEGACY_BREVITY_WORDS:
            del d["debate"]["brevity_words"]  # legacy protocol: identical to pre-versioning identities
        if d["model"].get("readout_head") == LEGACY_READOUT_HEAD:
            del d["model"]["readout_head"]
        for key in ("gen_batch_tokens", "top_k", "repetition_penalty", "chat_template_kwargs"):
            if d["model"].get(key) is None:
                d["model"].pop(key, None)
        return d

    def hash(self) -> str:
        return config_hash(self.to_dict())


def config_hash(obj: Any) -> str:
    """Deterministic sha256 over a config dict. Key order and float repr are normalised."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a flat config dataclass from a dict, ignoring unknown keys loudly."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


_SECTIONS = {
    "model": ModelConfig,
    "stub": StubConfig,
    "debate": DebateConfig,
    "data": DataConfig,
    "activations": ActivationConfig,
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(*paths: str | Path, **overrides: Any) -> RunConfig:
    """Load and merge YAML config files left-to-right, then apply keyword overrides.

    Keyword overrides use dotted paths, e.g. ``load_config("base.yaml", **{"data.n_questions": 50})``.
    """
    raw: dict[str, Any] = {}
    for p in paths:
        with open(p) as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    for dotted, value in overrides.items():
        cursor = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _SECTIONS:
            kwargs[key] = _from_dict(_SECTIONS[key], value or {})
        else:
            kwargs[key] = value
    cfg = RunConfig(**kwargs)
    cfg.data.split_fracs = tuple(cfg.data.split_fracs)  # yaml gives a list
    validate(cfg)
    return cfg


def config_from_dict(data: dict[str, Any]) -> RunConfig:
    """Rebuild a RunConfig from its ``to_dict()`` form (e.g. a parent job's recorded identity)."""
    kwargs = {k: (_from_dict(_SECTIONS[k], v or {}) if k in _SECTIONS else v) for k, v in data.items()}
    cfg = RunConfig(**kwargs)
    cfg.data.split_fracs = tuple(cfg.data.split_fracs)
    validate(cfg)
    return cfg


CANONICAL_ANSWERS = ("stated", "readout")
LEGACY_BREVITY_WORDS = 150
LEGACY_READOUT_HEAD = "bf16_logits"
READOUT_HEADS = ("bf16_logits", "fp32_head")
BREVITY_WORDS = (150, 250)


def validate(cfg: RunConfig) -> None:
    """Reject settings that nothing downstream implements, instead of silently ignoring them."""
    from mad.analysis.labels import TIE_BREAKS

    if cfg.debate.tie_break not in TIE_BREAKS:
        raise ValueError(f"debate.tie_break={cfg.debate.tie_break!r}; expected one of {TIE_BREAKS}")
    if cfg.debate.canonical_answer not in CANONICAL_ANSWERS:
        raise ValueError(f"debate.canonical_answer={cfg.debate.canonical_answer!r}; "
                         f"expected one of {CANONICAL_ANSWERS}")
    if cfg.model.readout_head not in READOUT_HEADS:
        raise ValueError(f"model.readout_head={cfg.model.readout_head!r}; expected one of {READOUT_HEADS}")
    if cfg.debate.brevity_words not in BREVITY_WORDS:
        raise ValueError(f"debate.brevity_words={cfg.debate.brevity_words}; expected one of {BREVITY_WORDS}")
    if cfg.debate.n_rounds != 1:
        raise ValueError("only one revision round is implemented (debate.n_rounds must be 1)")
    unknown = set(cfg.activations.positions) - set(POSITIONS)
    if unknown:
        raise ValueError(f"unknown activation positions {sorted(unknown)}; expected from {POSITIONS}")


# Token positions at which hidden states can be kept (see mad.backends.hf.HFBackend.readout):
#   readout_last     the "(" of the appended "Final answer: (" -- the forced-choice position
#   generation_last  the last generated token, before anything is appended
#   stated_answer    the letter token inside the agent's own final "Answer: (X)" line
#   before_stated_answer  the token immediately before that letter, whose output predicts it
POSITIONS = ("readout_last", "generation_last", "stated_answer", "before_stated_answer")
