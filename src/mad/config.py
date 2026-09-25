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
    batch_size: int = 8


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
    freetext_noise: float = 0.03  # P(prose states a different letter than the readout commits to)
    confidence_signal: float = 0.55  # readout-confidence separation; tuned for AUROC ~0.68
    confidence_noise: float = 1.1


@dataclass
class DebateConfig:
    n_agents: int = 3
    n_rounds: int = 1
    condition: Condition = "debate"
    tie_break: str = "round0_then_agent0"  # round0_then_agent0 | round0_then_confidence | agent0
    # One persona for every agent. Distinct personas made the agents the same weights but
    # different policies, which contradicts the homogeneous-agent framing and interacted badly
    # with a tie-break that falls back to agent 0. Diversity now comes from sampling alone.
    personas: list[str] = field(default_factory=lambda: ["You are a careful, methodical reasoner."])
    peer_shows_confidence: bool = False
    ask_verbalized_confidence: bool = True


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
    dtype: str = "float16"
    shard_size: int = 500


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
        return asdict(self)

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
    return cfg
