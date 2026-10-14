"""End to end: stub archives -> cache -> freeze -> (training data gone) -> load -> apply.

The frozen scorers are fitted on a pilot-like archive (debate + self_reconsider, 12 questions)
and applied to a validation-like archive with a DIFFERENT number of questions and states, only
`debate` and `independent`, and some absent before-answer states; then to a complete-chunk
partial version of it. Prediction must not need the training rows.
"""

import importlib.util
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]
# the stub has 8 layers of width 64: scaled-down versions of the two fixed recipes
TEST_RECIPES = {"R1": "readout_last:letter:meandiff:2-6:2", "R2": "before_stated_answer:letter:pca4:2-6:2"}


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[script] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mods():
    return _load("run_experiment"), _load("probe_search"), _load("probe_validate")


def _archive(driver, root, name, conds, n, chunk=4):
    root.mkdir(parents=True, exist_ok=True)
    pos = root / "positions.yaml"  # all four positions, as in configs/colab_12b.yaml
    pos.write_text("activations:\n  positions: [readout_last, generation_last, stated_answer, before_stated_answer]\n")
    args = ["--config", *STUB, str(pos), "--out-root", str(root), "--name", name, "--dataset", "synthetic",
            "--n-questions", str(n), "--chunk", str(chunk), "--conditions", *conds]
    assert driver.main(args) == 0


@pytest.fixture(scope="module")
def built(mods, tmp_path_factory):
    driver, PS, PV = mods
    base = tmp_path_factory.mktemp("frozen")
    _archive(driver, base / "pilot", "pa", ["debate", "self_reconsider"], 12)
    _archive(driver, base / "val", "vb", ["debate", "independent"], 8)
    PS.build_cache(base / "pilot", "pa", base / "cache_pa")
    PS.build_cache(base / "val", "vb", base / "cache_vb")
    # absent before-answer states on the validation side (as for capped responses)
    k = pd.read_parquet(base / "cache_vb" / "keys.parquet")
    drop = k[(k.position == "before_stated_answer")].index[:5]
    k.drop(index=drop).reset_index(drop=True).to_parquet(base / "cache_vb" / "keys.parquet", index=False)
    frozen = base / "frozen.pkl"
    PV.freeze(str(base / "cache_pa"), str(frozen), str(base / "frozen_out"), recipes=TEST_RECIPES)
    shutil.rmtree(base / "cache_pa")  # the training data are gone before application
    return base, frozen


def test_cache_discovers_only_the_collected_conditions(built):
    base, _ = built
    meta = json.loads((base / "cache_vb" / "meta.json").read_text())
    assert meta["conditions"] == ["debate", "independent"]


def test_apply_to_a_different_sized_archive_with_missing_states(mods, built):
    _, _, PV = mods
    base, frozen = built
    F = pickle.load(open(frozen, "rb"))
    assert F["trained_on"]["questions"] == 12
    out = base / "applied"
    PV.apply_frozen(str(frozen), str(base / "cache_vb"), str(out), n_boot=50)
    rep = json.loads((out / "frozen_application.json").read_text())
    assert rep["n_questions"] == 8 and rep["conditions"] == ["debate", "independent"]
    states = pd.read_csv(out / "states.csv")
    assert len(states) == 8 * 9  # round 0 + two conditions, three agents
    scores = pd.read_csv(out / "scores_R2.csv.gz")
    assert len(scores) == len(states)
    absent = states["present:before_stated_answer"] == 0
    assert absent.sum() == 5
    assert scores.loc[absent, "act:z"].isna().all() and scores.loc[absent, "act_fb:z"].notna().all()
    r2 = rep["recipes"]["R2"]
    assert "debate|complete_case" in r2["conditions"] and "debate|all_fallback" in r2["conditions"]
    assert r2["conditions"]["debate|all_fallback"]["n_questions"] == 8
    assert r2["conditions"]["debate|complete_case"]["n_questions"] < 8
    assert set(r2["source_comparison"]) == {"complete_case", "all_fallback"}
    cc = r2["source_comparison"]["complete_case"]
    assert cc["n_questions"] == r2["conditions"]["debate|complete_case"]["n_questions"] or cc["n_questions"] <= 8
    assert (out / "source_comparison_R1_complete_case.csv").exists()
    dec = pd.read_csv(out / "decisions_R1.csv")
    assert set(dec.condition) == {"debate", "independent"}


def test_apply_to_a_complete_chunk_partial_archive(mods, built):
    """A budget-truncated job keeps whole chunks: apply to the first chunk's questions only."""
    _, _, PV = mods
    base, frozen = built
    part = base / "cache_partial"
    shutil.copytree(base / "cache_vb", part)
    job = json.loads((base / "val" / "vb-job.json").read_text())
    first = set(job["chunks"][0])
    st = pd.read_parquet(part / "states.parquet")
    st[st.question_id.astype(str).isin(first)].to_parquet(part / "states.parquet", index=False)
    PV.apply_frozen(str(frozen), str(part), str(base / "applied_partial"), n_boot=30)
    rep = json.loads((base / "applied_partial" / "frozen_application.json").read_text())
    assert rep["n_questions"] == len(first) == 4


def test_feature_schema_mismatch_is_refused(mods, built):
    _, _, PV = mods
    base, frozen = built
    bad = base / "cache_bad"
    shutil.copytree(base / "cache_vb", bad)
    st = pd.read_parquet(bad / "states.parquet")
    st.drop(columns=[c for c in st.columns if c.startswith("of_")][:1]).to_parquet(bad / "states.parquet", index=False)
    with pytest.raises(SystemExit, match="feature columns differ"):
        PV.apply_frozen(str(frozen), str(bad), str(base / "applied_bad"), n_boot=10)


def test_frozen_prediction_never_reads_training_arrays(mods, built):
    """freeze() strips the training-length reference arrays; scoring still works."""
    base, frozen = built
    F = pickle.load(open(frozen, "rb"))
    for ms in F["models"].values():
        for m in ms.values():
            assert getattr(m, "ref_z", None) is None
    assert np.isfinite(F["train_base_rate"])
