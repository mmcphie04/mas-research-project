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


def test_full_run_writes_views_and_status(driver, tmp_path):
    assert driver.main(_args(tmp_path)) == 0
    for c in ("self_reconsider", "debate", "misleading_one"):
        d = tmp_path / f"t-{c}"
        view = json.loads((d / "view.json").read_text())
        assert view["complete"] and view["chunks_complete"] == 3 and view["questions"] == 6
        assert not list(d.glob("acts_*.npy")), "a view must reference shards, not copy them"
    status = json.loads((tmp_path / "t-status.json").read_text())
    assert status["state"] == "finished"
    assert status["chunks_done"] == status["chunks_total"] == 3
    assert status["timings"] and not status["failures"]


def test_view_loads_the_same_activations_as_the_chunks(driver, tmp_path):
    import numpy as np

    from mad.store import ActivationStore

    driver.main(_args(tmp_path))
    view = ActivationStore(tmp_path / "t-debate")
    parts = [ActivationStore(tmp_path / f"t-debate-c00{i}").load_layer(3) for i in range(3)]
    assert np.array_equal(view.load_layer(3), np.concatenate(parts))


def test_resolved_config_records_the_cli_dataset_and_size(driver, tmp_path):
    driver.main(_args(tmp_path, n=4))
    job = json.loads((tmp_path / "t-job.json").read_text())
    assert job["identity"]["config"]["data"]["dataset"] == "synthetic"
    assert job["identity"]["dataset"]["key"] == "synthetic" and len(job["questions"]) == 4
    m = json.loads((tmp_path / "t-debate-c000" / "manifest.json").read_text())
    assert m["config"]["data"]["n_questions"] == 4 and m["config"]["data"]["dataset"] == "synthetic"
    assert m["job_identity_sha256"] == job["identity_sha256"]


def test_rerun_skips_finished_chunks(driver, tmp_path, capsys):
    driver.main(_args(tmp_path))
    capsys.readouterr()
    driver.main(_args(tmp_path))
    out = capsys.readouterr().out
    assert "questions in" not in out, "a finished chunk was recomputed instead of skipped"


def test_a_lost_chunk_is_redone_and_the_view_rebuilt(driver, tmp_path, capsys):
    """What a Colab disconnect looks like: one chunk without its completion record."""
    driver.main(_args(tmp_path))
    (tmp_path / "t-debate-c001" / "completion.json").unlink()
    capsys.readouterr()
    assert driver.main(_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "c001:" in out, "the incomplete chunk was not redone"
    assert "c000:" not in out and "c002:" not in out, "complete chunks were redone"
    assert json.loads((tmp_path / "t-debate" / "view.json").read_text())["chunks_complete"] == 3


def test_extension_appends_to_the_same_question_order(driver, tmp_path, capsys):
    driver.main(_args(tmp_path, n=4))
    first = json.loads((tmp_path / "t-job.json").read_text())["questions"]
    capsys.readouterr()
    assert driver.main(_args(tmp_path, n=6)) == 0
    out = capsys.readouterr().out
    job = json.loads((tmp_path / "t-job.json").read_text())
    assert job["questions"][:4] == first, "extension resampled instead of appending"
    assert "c002:" in out and "c000:" not in out and "c001:" not in out
    assert [h["event"] for h in job["history"]] == ["created", "extended"]


def test_extension_that_would_change_a_started_chunk_is_refused(driver, tmp_path):
    from mad.job import JobMismatch

    driver.main(_args(tmp_path, n=5))  # last chunk holds one question
    with pytest.raises(JobMismatch, match="membership"):
        driver.main(_args(tmp_path, n=6))


@pytest.mark.parametrize("change", [["--chunk", "3"], ["--set", "model.max_new_tokens=77"],
                                    ["--conditions", "debate"], ["--pool", "pilot"]])
def test_resume_with_different_settings_is_refused(driver, tmp_path, change):
    from mad.job import JobMismatch

    driver.main(_args(tmp_path, n=4))
    args = _args(tmp_path, n=4)
    if change[0] in args:  # replace an existing flag's value
        i = args.index(change[0])
        args[i + 1] = change[1]
    else:
        args += change
    with pytest.raises(JobMismatch, match="different settings"):
        driver.main(args)


def test_execution_settings_may_change_on_resume(driver, tmp_path):
    driver.main(_args(tmp_path, n=4))
    assert driver.main(_args(tmp_path, "--set", "model.batch_size=2", n=4)) == 0


def test_resume_at_a_different_commit_needs_explicit_permission(driver, tmp_path):
    from mad.job import JobMismatch

    driver.main(_args(tmp_path, n=4))
    p = tmp_path / "t-job.json"
    job = json.loads(p.read_text())
    job["code"]["git_sha"] = "0" * 40
    p.write_text(json.dumps(job))
    with pytest.raises(JobMismatch, match="allow-code-change"):
        driver.main(_args(tmp_path, n=4))
    assert driver.main(_args(tmp_path, "--allow-code-change", n=4)) == 0
    assert json.loads(p.read_text())["history"][-1]["event"] == "resumed+code_change"


def test_a_chunk_from_another_job_is_refused(driver, tmp_path):
    from mad.job import JobMismatch

    driver.main(_args(tmp_path, n=4))
    m = tmp_path / "t-debate-c001" / "manifest.json"
    man = json.loads(m.read_text())
    man["job_identity_sha256"] = "other"
    m.write_text(json.dumps(man))
    with pytest.raises(JobMismatch, match="different job"):
        driver.main(_args(tmp_path, n=4))


def test_a_corrupted_shard_fails_the_view(driver, tmp_path):
    from mad.job import JobMismatch

    driver.main(_args(tmp_path, n=4))
    (tmp_path / "t-debate" / "view.json").unlink()
    shard = tmp_path / "t-debate-c000" / "acts_0000.npy"
    shard.write_bytes(shard.read_bytes()[:-8] + b"\x00" * 8)
    with pytest.raises(JobMismatch, match="checksum"):
        driver.main(_args(tmp_path, n=4))


def test_round0_states_are_referenced_not_copied(driver, tmp_path):
    import pandas as pd

    driver.main(_args(tmp_path, n=2))
    ix = pd.read_parquet(tmp_path / "t-debate-c000" / "activation_index.parquet")
    pre = ix[ix.phase == "pre"]
    assert (pre.layout == "row_major").all() and pre.shard_path.str.startswith("../t-round0-c000/").all()
    assert not (ix[ix.phase != "pre"].shard_path.str.startswith("..")).any()


def test_full_verification_catches_a_flipped_byte_that_the_view_cache_misses(driver, tmp_path):
    from mad.job import JobMismatch, build_view, verify_archive

    driver.main(_args(tmp_path, n=4))
    assert verify_archive(tmp_path, "t")["views"]["debate"]["questions"] == 4
    shard = tmp_path / "t-debate-c001" / "acts_0001.npy"
    if not shard.exists():
        shard = next((tmp_path / "t-debate-c001").glob("acts_*.npy"))
    b = bytearray(shard.read_bytes())
    b[-3] ^= 0xFF  # one byte; no metadata touched
    shard.write_bytes(bytes(b))
    job = json.loads((tmp_path / "t-job.json").read_text())
    assert build_view(tmp_path, job, "debate") is not None  # the cached view does not notice
    with pytest.raises(JobMismatch, match="checksum"):
        verify_archive(tmp_path, "t")


def test_a_corrupted_shared_round0_file_is_caught(driver, tmp_path):
    from mad.job import JobMismatch, verify_archive

    driver.main(_args(tmp_path, n=2))
    f = tmp_path / "t-round0-c000" / "round0_hidden_readout_last.npy"
    b = bytearray(f.read_bytes())
    b[-5] ^= 0x01
    f.write_bytes(bytes(b))
    with pytest.raises(JobMismatch, match="checksum"):
        verify_archive(tmp_path, "t")


def test_an_archive_copied_elsewhere_verifies_and_loads_identically(driver, tmp_path):
    out, copy = tmp_path / "a", tmp_path / "elsewhere" / "b"
    driver.main(_args(out, n=4))
    va = _load("verify_archive")
    assert va.main(["--out-root", str(out), "--name", "t", "--copy-to", str(copy)]) == 0
    rep = json.loads((out / "t-verify.json").read_text())
    assert rep["ok"] and rep["copy_loads_identically"]
    assert "pre/readout_last" in rep["views"]["debate"]["by_phase_and_position"]


def test_smoke_measures_lengths_over_the_intended_pool_not_two_questions(driver, tmp_path, capsys):
    driver.main(_args(tmp_path, "--smoke", n=40))
    out = capsys.readouterr().out
    assert "token coverage over all 40 intended questions" in out


def test_ledger_records_every_session_and_chunk(driver, tmp_path):
    driver.main(_args(tmp_path, n=4))
    driver.main(_args(tmp_path, n=4))
    events = [json.loads(x)["event"] for x in (tmp_path / "t-ledger.jsonl").read_text().splitlines()]
    assert events.count("session_start") == 2 and events.count("session_end") == 2
    assert events.count("chunk_done") == 2
    status = json.loads((tmp_path / "t-status.json").read_text())
    assert "debate/generation_s" in status["timings"][0]["stages"]


def test_child_job_reuses_the_parents_round0_exactly(driver, tmp_path):
    import pandas as pd

    from mad.job import verify_archive

    driver.main(_args(tmp_path, n=6, chunk=2))
    child = ["--config", *STUB, "--out-root", str(tmp_path), "--name", "kid", "--parent", "t",
             "--subset-n", "3", "--conditions", "debate_answer_only", "independent"]
    assert driver.main(child) == 0
    assert not list(tmp_path.glob("kid-round0-*")), "a child job generated its own round 0"
    job = json.loads((tmp_path / "kid-job.json").read_text())
    assert job["identity"]["parent"]["identity_sha256"] == json.loads(
        (tmp_path / "t-job.json").read_text())["identity_sha256"]
    assert [len(c) for c in job["chunks"]] == [2, 1, 0]
    k = ["question_id", "agent_id"]
    parent = pd.read_parquet(tmp_path / "t-debate" / "agent_states.parquet").query("phase == 'pre'")
    kid = pd.read_parquet(tmp_path / "kid-debate_answer_only" / "agent_states.parquet").query("phase == 'pre'")
    merged = kid.merge(parent, on=k, suffixes=("_kid", "_parent"))
    assert len(merged) == 9 and (merged.answer_kid == merged.answer_parent).all()
    assert (merged.readout_logprobs_kid == merged.readout_logprobs_parent).all()
    rep = verify_archive(tmp_path, "kid")
    assert rep["views"]["independent"]["questions"] == 3


def test_child_job_refuses_without_the_parents_round0(driver, tmp_path):
    import shutil

    driver.main(_args(tmp_path, n=4, chunk=2))
    shutil.rmtree(tmp_path / "t-round0-c001")
    child = ["--config", *STUB, "--out-root", str(tmp_path), "--name", "kid", "--parent", "t",
             "--conditions", "debate_answer_only"]
    assert driver.main(child) == 1
    status = json.loads((tmp_path / "kid-status.json").read_text())
    assert any("never generates its own round 0" in f["message"] for f in status["failures"])


def test_child_job_refuses_a_different_config(driver, tmp_path):
    driver.main(_args(tmp_path, n=4, chunk=2))
    child = ["--config", *STUB, "--out-root", str(tmp_path), "--name", "kid", "--parent", "t",
             "--conditions", "debate_answer_only", "--set", "model.max_new_tokens=99"]
    with pytest.raises(SystemExit, match="parent's config"):
        driver.main(child)


def test_pilot_and_main_pools_are_disjoint(driver, tmp_path):
    driver.main(_args(tmp_path, "--pool", "pilot", "--pilot-size", "8", n=8, name="p"))
    driver.main(_args(tmp_path, "--pilot-size", "8", n=8, name="m"))
    ids = {n: {q["id"] for q in json.loads((tmp_path / f"{n}-job.json").read_text())["questions"]}
           for n in ("p", "m")}
    assert not ids["p"] & ids["m"]


def test_time_cap_persists_across_sessions(driver, tmp_path):
    assert driver.main(_args(tmp_path, "--max-hours", "0")) == 0
    status = json.loads((tmp_path / "t-status.json").read_text())
    assert status["state"] == "stopped_time_cap" and status["chunks_done"] == 0
    s = tmp_path / "t-status.json"
    st = json.loads(s.read_text())
    st["elapsed_hours_total"] = 5.0
    s.write_text(json.dumps(st))
    driver.main(_args(tmp_path, "--max-hours", "4"))
    assert json.loads(s.read_text())["state"] == "stopped_time_cap", "earlier sessions' hours were forgotten"


def test_repeated_failures_stop_the_run(driver, tmp_path, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("something broke")

    monkeypatch.setattr(driver, "run_chunk", boom)
    assert driver.main(_args(tmp_path, n=10)) == 1
    assert len(calls) == 2, "the driver kept failing chunk after chunk"
    assert json.loads((tmp_path / "t-status.json").read_text())["state"] == "failed"


def test_out_of_memory_stops_immediately(driver, tmp_path, monkeypatch):
    calls = []

    def oom(*a, **k):
        calls.append(1)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(driver, "run_chunk", oom)
    assert driver.main(_args(tmp_path, n=10)) == 1
    assert len(calls) == 1


def test_smoke_mode_checks_values_and_cleans_up(driver, tmp_path, capsys):
    assert driver.main(_args(tmp_path, "--smoke", n=60)) == 0
    out = capsys.readouterr().out
    assert "smoke checks passed" in out and "SMOKE PASSED" in out
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
    assert set(s["conditions"]) == {"self_reconsider", "debate", "misleading_one"}
    d = s["conditions"]["debate"]
    for k in ("one_agent_accuracy", "vote_of_3", "after_round_1", "panel_optimal"):
        lo, hi = d[k]["ci95"]
        assert lo <= d[k]["mean"] <= hi, f"{k}: mean outside its own interval"
    assert d["panel_optimal"]["mean"] >= d["after_round_1"]["mean"] - 1e-9
    assert "hit_token_cap" in d["generation"]["pre"]
    assert d["completeness"]["complete"]


def test_summary_ignores_directories_outside_the_job(driver, tmp_path):
    """The old name filter ("-c0" not in name) broke at chunk c100 and picked up stray dirs."""
    import shutil

    driver.main(_args(tmp_path))
    shutil.copytree(tmp_path / "t-debate-c000", tmp_path / "t-debate-c100")
    shutil.copytree(tmp_path / "t-debate", tmp_path / "t-stray")
    summ = _load("summarize_run")
    assert summ.main(["--out-root", str(tmp_path), "--name", "t"]) == 0
    s = json.loads((tmp_path / "t-summary.json").read_text())
    assert set(s["conditions"]) == {"self_reconsider", "debate", "misleading_one"}


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


def test_notebook_never_pushes_to_master_never_forces_and_is_off_by_default():
    _, cells = _code_cells()
    push = next(c for c in cells if "PUSH_RESULTS" in c)
    assert "PUSH_RESULTS = False" in push
    assert "colab-results/" in push and '"master"' not in push
    assert '"-f", authed' not in push and "--force" not in push, "never force-push"
    assert '"-c0"' not in push and "view_dirs" in push, "results must be found through the job file"


def test_notebook_runs_a_pinned_commit():
    _, cells = _code_cells()
    src = "\n".join(cells)
    assert 'COMMIT      = "SET-ME"' in src and "len(COMMIT) == 40" in src
    assert '"checkout", "--quiet", "--detach", COMMIT' in src
    assert '"pull"' not in src and "BRANCH" not in src, "the notebook must not follow a moving branch"


def test_notebook_flags_match_the_driver():
    """The notebook passes CLI flags by name; a rename in the driver must not silently break it."""
    _, cells = _code_cells()
    driver_cells = [c for c in cells if "run_experiment.py" in c or "BASE_CMD" in c]
    used = set(re.findall(r'"(--[a-z-]+)"', "\n".join(driver_cells)))
    driver_src = (ROOT / "scripts" / "run_experiment.py").read_text()
    defined = set(re.findall(r'add_argument\("(--[a-z-]+)"', driver_src))
    assert used, "found no flags in the notebook -- the regex is wrong"
    assert used <= defined, f"notebook uses flags the driver does not define: {used - defined}"


def test_pilot_report_runs_on_a_four_condition_pilot(driver, tmp_path):
    assert driver.main(_args(tmp_path, "--pool", "pilot", "--pilot-size", "40", "--conditions",
                             "self_reconsider", "debate", "debate_answer_only", "misleading_one",
                             n=40, chunk=20)) == 0
    pr = _load("pilot_report")
    out = tmp_path / "report"
    assert pr.main(["--out-root", str(tmp_path), "--name", "t", "--out", str(out)]) == 0
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["conditions"]) == {"self_reconsider", "debate", "debate_answer_only", "misleading_one"}
    assert rep["power"]["table"] and rep["disk"]["projected_gb"] and "runtime" in rep
    assert (out / "transcripts.md").read_text().count("###") >= 12
    assert pr.required_test_questions(0.03, 0.10) > pr.required_test_questions(0.04, 0.10)
