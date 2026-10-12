"""Gate evaluation: training-only choices, one score scale, the new schema, information times."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mad.analysis import gates as G
from mad.analysis.features import load_events, output_features

ROOT = Path(__file__).resolve().parents[1]
STUB = ["configs/base.yaml", "configs/smoke_stub.yaml"]


def _load(script):
    spec = importlib.util.spec_from_file_location(script, ROOT / "scripts" / f"{script}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _events(n_q=60, seed=0, p_rev=0.4):
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_q):
        gold = int(rng.integers(4))
        for a in range(3):
            pre = gold if rng.random() < 0.45 else int(rng.integers(4))
            post = pre if rng.random() > p_rev else int(rng.integers(4))
            rows.append({"run_id": "r", "condition": "debate", "question_id": f"q{q}", "agent_id": a,
                         "subject": "s", "n_options": 4, "gold_display": gold, "pre_answer": pre,
                         "post_answer": post, "pre_correct": int(pre == gold), "post_correct": int(post == gold),
                         "pre_confidence": 0.5, "post_confidence": 0.5})
    ev = pd.DataFrame(rows)
    ev["revised"] = (ev.pre_answer != ev.post_answer).astype(int)
    ev["label"] = np.select([ev.revised == 0, ev.post_correct == 1, ev.pre_correct == 1],
                            ["none", "beneficial", "harmful"], "unhelpful")
    X = rng.normal(size=(len(ev), 5))
    X[:, 0] += ev.pre_correct * 0.8
    X[:, 1] += ev.post_correct * 0.8
    return ev, X


def test_held_out_labels_cannot_influence_any_choice():
    ev, X = _events()
    te = ev.question_id.isin([f"q{i}" for i in range(12)]).to_numpy()
    for kind in ("output_linear", "output_hgb"):
        F = G.Features(kind, output=X)
        a = G.fit_crossfit(F, ev, ~te, "round0_then_agent0")
        ev2 = ev.copy()
        ev2.loc[te, "pre_correct"] = 1 - ev2.loc[te, "pre_correct"]
        ev2.loc[te, "post_correct"] = 1 - ev2.loc[te, "post_correct"]
        b = G.fit_crossfit(F, ev2, ~te, "round0_then_agent0")
        assert a["tau"] == b["tau"] and a["params"] == b["params"]
        sa, sb = G.score(a, F, te), G.score(b, F, te)
        for k in sa:
            assert np.allclose(sa[k], sb[k]), f"{kind}: held-out {k} moved with held-out labels"


def test_threshold_is_chosen_on_the_calibrated_scale_used_at_evaluation():
    """A strongly non-identity calibrator: the chosen tau must be optimal for the CALIBRATED
    out-of-fold scores, and evaluation must use calibrated scores."""
    ev, X = _events(n_q=90, seed=3)
    X = X * 25  # nearly separable raw logits -> raw probabilities near 0/1, far from calibrated
    F = G.Features("output_linear", output=X)
    tr = np.ones(len(ev), bool)
    f = G.fit_crossfit(F, ev, tr, "round0_then_agent0")
    for key in G.GATE_KEYS:
        rule, o = G.split_key(key)
        tau, best = G.choose_tau(ev, f["oof_cal"]["new"], f["oof_cal"]["old"], o, "round0_then_agent0", rule)
        assert f["tau"][key] == tau
    s = G.score(f, F, tr)
    assert np.allclose(s["new_cal"], f["calibrators"]["new"].transform(s["new_raw"]))
    assert not np.allclose(s["new_cal"], s["new_raw"]), "the test needs a non-identity calibrator"


def test_accept_all_and_reject_all_are_attainable():
    ev, _ = _events(n_q=40, seed=5, p_rev=0.6)
    rev = (ev.revised == 1).to_numpy()
    good = ev.label.eq("beneficial").to_numpy() | ~rev
    # scores say "accept" exactly when the revision is good; still, extreme cases must be reachable
    hi, lo = np.full(len(ev), 0.9), np.full(len(ev), 0.1)
    tau_all, _ = G.choose_tau(ev.assign(post_correct=1), hi, lo, "agent", "round0_then_agent0")
    assert tau_all == -np.inf or tau_all < 0.8
    tau_none, _ = G.choose_tau(ev.assign(post_correct=0, pre_correct=1), hi, lo, "agent", "round0_then_agent0")
    assert tau_none == np.inf
    assert good.any()


@pytest.fixture(scope="module")
def stub_view(tmp_path_factory):
    out = tmp_path_factory.mktemp("stubrun")
    drv = _load("run_experiment")
    assert drv.main(["--config", *STUB, "--out-root", str(out), "--name", "s", "--dataset", "synthetic",
                     "--n-questions", "60", "--chunk", "20", "--conditions", "debate", "self_reconsider",
                     "--set", 'activations.positions=["readout_last","generation_last","stated_answer"]']) == 0
    return out


def test_new_schema_is_read_without_a_sidecar(stub_view):
    d = stub_view / "s-debate"
    assert not (d / "readout_full.parquet").exists()
    ev, _ = load_events(d)
    assert ev.attrs["lp_source"] == "agent_states.readout_logprobs" and ev.attrs["has_exposure"]


def test_features_respect_information_time(stub_view):
    ev, _ = load_events(stub_view / "s-debate")
    _, pre = output_features(ev, "pre")
    _, exp = output_features(ev, "exposure")
    _, post = output_features(ev, "post")
    assert not any("@exposure" in n or "@post" in n or "new" in n or "shown" in n for n in pre)
    assert any("@exposure" in n for n in exp) and not any("@post" in n or "new" in n for n in exp)
    assert any("@post" in n for n in post) and "p_new@exposure" in post
    for names in (pre, exp, post):
        assert not any(k in n for n in names for k in ("gold", "correct", "fabricated", "target"))


def test_cv_route_end_to_end_exports_everything(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--n-boot", "200", "--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--time", "exposure", "--families", "output_linear", "activation", "stacked"]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["activation_phases"] == ["pre", "peer_exposure"]
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert {"old_raw", "old_cal", "new_cal", "tau_difference_panel", "accept_difference_panel",
            "accept_protect_panel", "pre_answer",
            "post_answer", "gold_display"} <= set(pred.columns)
    assert set(pred.family) == {"output_linear", "activation", "stacked"}
    panel = pd.read_csv(tmp_path / "panel.csv")
    assert {"panel_answer", "panel_correct"} <= set(panel.columns)
    assert panel.question_id.nunique() == 60
    ch = json.loads((tmp_path / "choices.json").read_text())
    assert all("calibrators" in c and "tau" in c for c in ch)


def test_position_is_respected_and_missing_states_drop_whole_questions(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--n-boot", "200", "--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--position", "stated_answer", "--families", "activation"]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["position"] == "stated_answer" and res["questions_used"] + len(
        res["questions_dropped_missing_states"]) == 60


def test_split_route_scores_only_test_questions(stub_view, tmp_path):
    ms = _load("make_split")
    split = tmp_path / "split.json"
    assert ms.main(["--out-root", str(stub_view), "--name", "s", "--fracs", "0.5", "0.2", "0.3",
                    "--seed", "1", "--out", str(split)]) == 0
    assign = json.loads(split.read_text())["assignment"]
    R = _load("reanalysis")
    out = tmp_path / "o"
    args = ["--run", str(stub_view / "s-debate"), "--out", str(out), "--mode", "split",
            "--split-manifest", str(split), "--families", "output_linear", "activation"]
    with pytest.raises(SystemExit, match="collected with split"):
        R.main(args)  # made AFTER collection: not the preregistered route
    assert R.main(["--n-boot", "200", *args, "--allow-unregistered-split"]) == 0
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred.question_id.map(assign)) == {"test"}


def test_registered_split_route_end_to_end(tmp_path):
    """make_split BEFORE generation -> the driver binds it -> reanalysis accepts only that split."""
    ms, drv, R = _load("make_split"), _load("run_experiment"), _load("reanalysis")
    split = tmp_path / "m-split.json"
    assert ms.main(["--dataset", "synthetic", "--pool", "main", "--n-questions", "60", "--fracs", "0.5",
                    "0.2", "0.3", "--seed", "3", "--out", str(split)]) == 0
    assert drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "m", "--dataset", "synthetic",
                     "--n-questions", "60", "--chunk", "20", "--conditions", "debate",
                     "--split-manifest", str(split)]) == 0
    sha = json.loads(split.read_text())["meta"]["assignment_sha256"]
    assert json.loads((tmp_path / "m-job.json").read_text())["identity"]["split"]["assignment_sha256"] == sha
    assert json.loads((tmp_path / "m-debate" / "manifest.json").read_text())["split_assignment_sha256"] == sha
    out = tmp_path / "o"
    assert R.main(["--n-boot", "200", "--run", str(tmp_path / "m-debate"), "--out", str(out), "--mode", "split",
                   "--split-manifest", str(split), "--families", "output_linear"]) == 0
    assert json.loads((out / "results.json").read_text())["split"]["registered_with_run"] == sha


def test_split_integrity_and_membership_are_enforced(tmp_path):
    ms, drv = _load("make_split"), _load("run_experiment")
    split = tmp_path / "s.json"
    ms.main(["--dataset", "synthetic", "--n-questions", "40", "--seed", "1", "--out", str(split)])
    with pytest.raises(SystemExit, match="covers 40 questions"):  # split made for a different selection
        drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "x", "--dataset", "synthetic",
                  "--n-questions", "30", "--chunk", "10", "--conditions", "debate", "--split-manifest", str(split)])
    data = json.loads(split.read_text())
    q = next(iter(data["assignment"]))
    data["assignment"][q] = "train" if data["assignment"][q] != "train" else "test"  # tampered
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="recorded hash"):
        ms.load_split(bad)
    data2 = json.loads(split.read_text())
    data2["assignment"][q] = "holdout"
    data2["meta"]["assignment_sha256"] = ms.assignment_sha256(data2["assignment"])
    bad.write_text(json.dumps(data2))
    with pytest.raises(SystemExit, match="invalid split names"):
        ms.load_split(bad)


def test_child_job_keeps_its_parents_split(tmp_path):
    ms, drv = _load("make_split"), _load("run_experiment")
    split = tmp_path / "p-split.json"
    ms.main(["--dataset", "synthetic", "--n-questions", "40", "--seed", "1", "--out", str(split)])
    drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "p", "--dataset", "synthetic",
              "--n-questions", "40", "--chunk", "20", "--conditions", "debate", "--split-manifest", str(split)])
    assert drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "k", "--parent", "p",
                     "--subset-n", "20", "--conditions", "independent"]) == 0
    p_split = json.loads((tmp_path / "p-job.json").read_text())["identity"]["split"]
    assert json.loads((tmp_path / "k-job.json").read_text())["identity"]["split"] == p_split


def test_candidate_analysis_rejects_unmapped_questions(tmp_path):
    drv, rc, ac = _load("run_experiment"), _load("run_candidates"), _load("analyze_candidates")
    drv.main(["--config", *STUB, "--out-root", str(tmp_path), "--name", "p", "--dataset", "synthetic",
              "--n-questions", "30", "--chunk", "15", "--conditions", "debate"])
    rc.main(["--config", *STUB, "--out-root", str(tmp_path), "--parent", "p", "--n", "30"])
    ms = _load("make_split")
    split = tmp_path / "s.json"
    ms.main(["--out-root", str(tmp_path), "--name", "p", "--seed", "2", "--out", str(split)])
    data = json.loads(split.read_text())
    data["assignment"].pop(next(iter(data["assignment"])))  # one question unmapped
    data["meta"]["assignment_sha256"] = ms.assignment_sha256(data["assignment"])
    split.write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="not in the split"):
        ac.main(["--run", str(tmp_path / "p-candidates"), "--out", str(tmp_path / "a"),
                 "--split-manifest", str(split)])
    with pytest.raises(SystemExit, match="already complete"):
        rc.main(["--config", *STUB, "--out-root", str(tmp_path), "--parent", "p", "--n", "30"])



def test_transfer_applies_frozen_gates_to_another_condition(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--n-boot", "200", "--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--families", "output_linear", "--transfer-runs", str(stub_view / "s-self_reconsider")]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    tr = res["results"][0]["transfer"]["s-self_reconsider"]
    assert "output_linear" in tr["families"] and tr["reference"]["accept_all"]["panel_accuracy"] >= 0


def test_protect_gate_uses_only_the_old_answer_probability():
    ev, _ = _events(n_q=40, seed=7, p_rev=0.6)
    rev = (ev.revised == 1).to_numpy()
    p_old = np.linspace(0.05, 0.95, len(ev))
    a = G.gate_accept(np.zeros(len(ev)), p_old, rev, 0.5, "protect")
    b = G.gate_accept(np.ones(len(ev)), p_old, rev, 0.5, "protect")
    assert (a == b).all(), "the protect rule must not depend on P(new)"
    assert G.gate_accept(np.zeros(len(ev)), p_old, rev, np.inf, "protect").all()
    assert (G.gate_accept(np.zeros(len(ev)), p_old, rev, -np.inf, "protect") == ~rev).all()


def test_future_response_fields_cannot_change_pre_or_exposure_features(stub_view):
    ev, _ = load_events(stub_view / "s-debate")
    base = {t: output_features(ev, t, ["generation_last", "stated_answer"])[0] for t in ("pre", "exposure")}
    fut = ev.copy()
    fut.attrs.update(ev.attrs)
    fut["post_answer"] = (fut.post_answer + 1) % fut.n_options
    fut["post_rationale_chars"] = fut.post_rationale_chars * 3 + 7
    fut["post_verbalized"] = 1.0
    for c in [c for c in fut.columns if c.startswith("lp_post")]:
        fut[c] = fut[c].map(lambda v: v[::-1] if isinstance(v, np.ndarray) else v)
    for c in ("post_answer_source", "post_stop_reason", "post_freetext_parse_ok",
              "post_freetext_agrees_readout", "post_response_logprob_mean", "post_response_logprob_min"):
        fut[c] = "x" if fut[c].dtype == object else -1
    for t in ("pre", "exposure"):
        assert np.array_equal(output_features(fut, t, ["generation_last", "stated_answer"])[0], base[t]), t


def test_position_matched_output_features_have_explicit_missingness(stub_view):
    ev, _ = load_events(stub_view / "s-debate")
    X, names = output_features(ev, "post", ["stated_answer"])
    assert "pos:missing@pre@stated_answer" in names and "pos:p_new@post@stated_answer" in names
    assert not any("exposure@stated_answer" in n for n in names), "no stated answer at exposure"
    miss = X[:, names.index("pos:missing@post@stated_answer")]
    pnew = X[:, names.index("pos:p_new@post@stated_answer")]
    assert ((miss == 1) <= (pnew == 0)).all(), "missing positions must contribute zeros"


def test_readout_only_ablation_and_constant_baseline_are_exported(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--n-boot", "200", "--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--position", "stated_answer", "--families", "output_linear", "output_readout_only",
                   "activation", "--transfer-runs", str(stub_view / "s-self_reconsider")]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["output_positions"] == ["generation_last", "stated_answer"]  # strong set + activation position
    assert len(res["output_features"]) > len(res["output_features_readout_only"])
    r0 = res["results"][0]
    assert "output_readout_only" in r0["families"] and "constant_training_rate" in r0
    assert "gate_protect_panel" in r0["families"]["activation"]
    assert (tmp_path / "transfer_predictions.csv").exists() and (tmp_path / "transfer_panel.csv").exists()


def test_strong_output_baseline_is_the_default_where_saved():
    R = _load("reanalysis")
    all4 = ["readout_last", "generation_last", "stated_answer", "before_stated_answer"]
    assert R.resolve_output_positions(all4, "readout_last") == ["before_stated_answer", "generation_last"]
    assert R.resolve_output_positions(all4, "stated_answer") == ["before_stated_answer", "generation_last",
                                                                 "stated_answer"]
    assert R.resolve_output_positions([], "readout_last") == [], "old runs fall back to the readout"
    assert R.resolve_output_positions(all4, "readout_last", []) == [], "explicit empty means none"


def test_selection_by_logloss_rejects_near_constant_models():
    """Regression (12B pilot): mean per-fold AUROC preferred C=1e-4, whose near-constant
    predictions rank at chance or worse once folds are pooled. Log-loss does not."""
    ev, X = _events(n_q=120, seed=11)
    y, g = ev.pre_correct.to_numpy(), ev.question_id.to_numpy()
    tiny = G.cv_logloss(lambda: G.linear(1e-4), X, y, g)
    fair = G.cv_logloss(lambda: G.linear(1e-1), X, y, g)
    assert fair < tiny
    F = G.Features("output_linear", output=X)
    _, prm = G.select(F, np.ones(len(y), bool), y, g)
    assert prm["C"] != 1e-4


def test_calibration_never_reverses_a_ranking():
    from mad.analysis.calibrate import PlattScaler

    s = np.linspace(0, 1, 200)
    y = (s < 0.5).astype(int)  # scores anti-correlated with the outcome
    cal = PlattScaler().fit(s, y)
    out = cal.transform(np.array([0.1, 0.9]))
    assert cal.constant is not None and out[0] == out[1] == y.mean()
    good = PlattScaler().fit(s, 1 - y)
    a, b = good.transform(np.array([0.1, 0.9]))
    assert good.constant is None and b > a
