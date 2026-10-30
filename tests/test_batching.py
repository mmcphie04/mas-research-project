"""Token-budget generation batching (configs/main_250.yaml after G2 attempt 1)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from mad.backends.hf import plan_generation_batches
from mad.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_fixed_batches_without_a_budget():
    assert plan_generation_batches([10] * 10, 4, 100, None) == [(0, 4), (4, 8), (8, 10)]


def test_budget_packs_short_prompts_fully_and_long_ones_in_small_batches():
    short = plan_generation_batches([500] * 16, 8, 1280, 24000)
    assert short == [(0, 8), (8, 16)]  # 8 x 1,780 = 14,240 <= 24,000
    long = plan_generation_batches([7119] * 5, 8, 1280, 24000)
    assert long == [(0, 2), (2, 4), (4, 5)]  # 2 x 8,399 fits, 3 x 8,399 does not
    mixed = plan_generation_batches([500, 500, 7119, 500], 8, 1280, 24000)
    assert mixed == [(0, 2), (2, 4)]  # the long prompt sets the batch's per-sequence cost
    assert plan_generation_batches([30000], 8, 1280, 24000) == [(0, 1)]  # one prompt always runs


def test_budget_is_versioned_and_drives_the_stress_batch():
    legacy = load_config("configs/base.yaml", "configs/colab_12b.yaml")
    assert "gen_batch_tokens" not in legacy.to_dict()["model"]
    new = load_config("configs/base.yaml", "configs/colab_12b.yaml", "configs/main_250.yaml")
    assert new.to_dict()["model"]["gen_batch_tokens"] == 24000
    spec = importlib.util.spec_from_file_location("run_experiment", ROOT / "scripts" / "run_experiment.py")
    re = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(re)
    assert re.worst_case_generation_batch(new, {"worst_round1_prompt_tokens": 7119}) == 2
    assert re.worst_case_generation_batch(legacy, {"worst_round1_prompt_tokens": 3000}) == legacy.model.batch_size


def test_hf_generate_under_a_budget_returns_every_prompt_in_order():
    import pytest

    pytest.importorskip("transformers")
    from mad.backends.base import build_backend

    cfg = load_config("configs/base.yaml", "configs/smoke_tiny.yaml",
                      **{"model.max_new_tokens": 8, "model.batch_size": 4, "model.gen_batch_tokens": 60})
    b = build_backend(cfg)
    prompts = [b.render([{"role": "user", "content": f"Say the number {i}."}]) for i in range(5)]
    lens = [len(b.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    plan = b.plan_batches(lens)
    assert sum(e - s for s, e in plan) == 5 and len(plan) > 2  # the budget split the batches
    out = b.generate(prompts, seed=0)
    assert len(out) == 5 and all(g.n_prompt_tokens == n for g, n in zip(out, lens, strict=True))
