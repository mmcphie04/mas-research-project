import json

import pytest
from conftest import make_cfg, questions_for

from mad.analysis.labels import build_events, revision_rates
from mad.analysis.report import pilot_report
from mad.debate import run_debate
from mad.provenance import COMPLETION, MANIFEST, is_complete, require_complete
from mad.store import ActivationStore


def test_full_tier0_pipeline_recovers_the_planted_distribution(tmp_path):
    cfg = make_cfg(tmp_path, run_id="e2e", n_questions=400)
    questions = questions_for(cfg)
    result = run_debate(cfg, questions)

    n_q, n_agents = len(questions), cfg.debate.n_agents
    assert len(result.rows) == n_q * n_agents * 2
    assert result.n_states == n_q * n_agents * 2

    ev = build_events(result.rows)
    r = revision_rates(ev)
    s = cfg.stub
    assert abs(r["base_accuracy"] - s.base_accuracy) < 0.03
    assert abs(r["r_C"] - s.revision_rate_correct) < 0.04
    assert abs(r["r_I"] - s.revision_rate_incorrect) < 0.04
    assert abs(r["s_switch_to_correct"] - s.switch_to_correct) < 0.06


def test_provenance_is_written_before_and_after(tmp_path):
    cfg = make_cfg(tmp_path, run_id="prov", n_questions=20)
    result = run_debate(cfg, questions_for(cfg))
    manifest = json.loads((result.run_dir / MANIFEST).read_text())
    completion = json.loads((result.run_dir / COMPLETION).read_text())

    for key in (
        "run_id",
        "config",
        "config_hash",
        "packages",
        "git_sha",
        "git_dirty",
        "prompt_template_hashes",
        "question_id_list_hash",
        "timestamp_start",
    ):
        assert key in manifest, f"manifest is missing {key}"
    assert completion["row_counts"]["agent_states"] == 120
    assert completion["shard_sha256"]
    assert is_complete(result.run_dir)


def test_a_crashed_run_is_refused(tmp_path):
    cfg = make_cfg(tmp_path, run_id="crashed", n_questions=20)
    result = run_debate(cfg, questions_for(cfg))
    (result.run_dir / COMPLETION).unlink()
    assert not is_complete(result.run_dir)
    with pytest.raises(RuntimeError, match="crashed"):
        require_complete(result.run_dir)


def test_activations_align_with_rows(tmp_path):
    cfg = make_cfg(tmp_path, run_id="align", n_questions=30)
    result = run_debate(cfg, questions_for(cfg))
    store = ActivationStore(result.run_dir)
    assert len(store) == len(result.rows)
    assert store.n_layers == cfg.stub.n_layers
    assert store.d_model == cfg.stub.d_model
    keys_store = set(
        zip(store.index.question_id, store.index.agent_id, store.index.phase, strict=True)
    )
    keys_rows = set(
        zip(result.rows.question_id, result.rows.agent_id, result.rows.phase, strict=True)
    )
    assert keys_store == keys_rows
    assert store.load_layer(0).shape == (len(result.rows), cfg.stub.d_model)


def test_control_condition_revises_less_than_debate(tmp_path):
    base = make_cfg(tmp_path, run_id="cond-debate", n_questions=300)
    debate = run_debate(base, questions_for(base))
    ctrl_cfg = make_cfg(tmp_path, run_id="cond-self", n_questions=300)
    ctrl_cfg.debate.condition = "self_reconsider"
    control = run_debate(ctrl_cfg, questions_for(ctrl_cfg))

    r_debate = build_events(debate.rows)["revised"].mean()
    r_control = build_events(control.rows)["revised"].mean()
    assert r_control < r_debate, "the self-reconsideration control must isolate peer influence"
    assert abs(r_control / r_debate - ctrl_cfg.stub.control_revision_scale) < 0.15


def test_pilot_report_runs_and_evaluates_every_gate(tmp_path):
    cfg = make_cfg(tmp_path, run_id="rep", n_questions=120)
    result = run_debate(cfg, questions_for(cfg))
    report = pilot_report(result.rows, probe_auroc=0.71)
    assert report["n_questions"] == 120
    assert set(report["gates"]) >= {"G1_base_accuracy", "G5_panel_headroom", "G7_probe_auroc"}
    assert report["gates"]["G7_probe_auroc"]["pass"] is True
    assert all(g["value"] is not None for g in report["gates"].values())
    assert report["parsing"]["freetext_parse_failure_rate"] == 0.0
