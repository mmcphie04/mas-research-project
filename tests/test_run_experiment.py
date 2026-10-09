"""The Colab driver, its resume behaviour, the smoke checks, the summary, and the notebook.

All against the tier-0 stub, so they run in seconds with no GPU, no network and no model.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(script: str):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver():
    return _load("run_experiment")


def _args(out, *extra, n=6, chunk=2, name="t"):
    return ["--config", *STUB, "--out-root", str(out), "--name", name, "--dataset", "synthetic",
            "--n-questions", str(n), "--chunk", str(chunk), *extra]


def test_full_run_writes_merged_conditions_and_status(driver, tmp_path):
    assert driver.main(_args(tmp_path)) == 0
    for c in ("self_reconsider", "debate", "misleading_one"):
        d = tmp_path / f"t-{c}"
        assert (d / "completion.json").exists(), f"{c} was not merged"
        assert len(json.loads((d / "completion.json").read_text())["merged_from"]) == 3
    status = json.loads((tmp_path / "t-status.json").read_text())
    assert status["chunks_done"] == status["chunks_total"] == 3
    assert status["timings"] and not status["failures"]


def test_rerun_skips_finished_chunks(driver, tmp_path, capsys):
    driver.main(_args(tmp_path))
    capsys.readouterr()
    driver.main(_args(tmp_path))
    out = capsys.readouterr().out
    assert "questions in" not in out, "a finished chunk was recomputed instead of skipped"


def test_a_lost_chunk_is_redone_and_remerged(driver, tmp_path, capsys):
    """What a Colab disconnect looks like: one chunk without its completion record."""
    driver.main(_args(tmp_path))
    (tmp_path / "t-debate-c001" / "completion.json").unlink()
    capsys.readouterr()
    assert driver.main(_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "c001:" in out, "the incomplete chunk was not redone"
    assert "c000:" not in out and "c002:" not in out, "complete chunks were redone"
    merged = json.loads((tmp_path / "t-debate" / "completion.json").read_text())["merged_from"]
    assert len(merged) == 3


def test_smoke_mode_checks_shapes_and_cleans_up(driver, tmp_path, capsys):
    assert driver.main(_args(tmp_path, "--smoke", n=60)) == 0
    out = capsys.readouterr().out
    assert "shape checks passed" in out and "SMOKE PASSED" in out
    assert not list(tmp_path.glob("t-*")), "smoke must not write into the real run's directories"
    capsys.readouterr()
    assert driver.main(_args(tmp_path, "--smoke", n=60)) == 0  # a second smoke run starts clean


def test_config_overrides_reach_the_run(driver, tmp_path):
    driver.main(_args(tmp_path, "--set", "model.max_new_tokens=77", n=2))
    m = json.loads((tmp_path / "t-debate-c000" / "manifest.json").read_text())
    assert m["config"]["model"]["max_new_tokens"] == 77


def test_summary_reports_intervals_and_generation_diagnostics(driver, tmp_path):
    driver.main(_args(tmp_path))
    summ = _load("summarize_run")
    assert summ.main(["--out-root", str(tmp_path), "--name", "t"]) == 0
    s = json.loads((tmp_path / "t-summary.json").read_text())
    assert set(s) == {"self_reconsider", "debate", "misleading_one"}
    d = s["debate"]
    for k in ("one_agent_accuracy", "vote_of_3", "after_round_1", "best_possible"):
        lo, hi = d[k]["ci95"]
        assert lo <= d[k]["mean"] <= hi, f"{k}: mean outside its own interval"
    assert d["best_possible"]["mean"] >= d["after_round_1"]["mean"] - 1e-9
    assert "hit_cap" in d["generation_pre"] and "parse_ok" in d["generation_post"]


# ----------------------------------------------------------------------------- the notebook

NB = ROOT / "notebooks" / "colab_runner.ipynb"


def _code_cells():
    nb = json.loads(NB.read_text())
    return nb, ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_is_valid_and_every_code_cell_parses():
    nb, cells = _code_cells()
    assert nb["nbformat"] == 4 and cells
    for i, src in enumerate(cells):
        ast.parse(src, filename=f"cell {i}")


def test_notebook_contains_no_secrets():
    text = NB.read_text()
    for pat in (r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}", r"hf_[A-Za-z0-9]{20,}"):
        assert not re.search(pat, text), f"a token-shaped string is hardcoded in the notebook: {pat}"
    _, cells = _code_cells()
    src = "\n".join(cells)  # decoded: the raw file JSON-escapes the quotes
    assert 'userdata.get("HF_TOKEN")' in src and 'userdata.get("GITHUB_TOKEN")' in src


def test_notebook_masks_the_token_and_strips_it_from_the_remote():
    _, cells = _code_cells()
    src = "\n".join(cells)
    assert 'replace(GITHUB_TOKEN, "***")' in src, "git errors could print the token"
    assert '"remote", "set-url", "origin"' in src, "the token would stay in .git/config"


def test_notebook_never_pushes_to_master_and_is_off_by_default():
    _, cells = _code_cells()
    push = next(c for c in cells if "PUSH_RESULTS" in c)
    assert "PUSH_RESULTS = False" in push
    assert '"colab-results"' in push and '"master"' not in push


def test_notebook_flags_match_the_driver():
    """The notebook passes CLI flags by name; a rename in the driver must not silently break it."""
    _, cells = _code_cells()
    driver_cells = [c for c in cells if "run_experiment.py" in c or "BASE_CMD" in c]
    used = set(re.findall(r'"(--[a-z-]+)"', "\n".join(driver_cells)))
    driver_src = (ROOT / "scripts" / "run_experiment.py").read_text()
    defined = set(re.findall(r'add_argument\("(--[a-z-]+)"', driver_src))
    assert used, "found no flags in the notebook -- the regex is wrong"
    assert used <= defined, f"notebook uses flags the driver does not define: {used - defined}"
