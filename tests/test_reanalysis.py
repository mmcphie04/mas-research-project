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
    for o in G.OBJECTIVES:
        tau, best = G.choose_tau(ev, f["oof_cal"]["new"], f["oof_cal"]["old"], o, "round0_then_agent0")
        assert f["tau"][o] == tau
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
    assert R.main(["--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--time", "exposure", "--families", "output_linear", "activation", "stacked"]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["activation_phases"] == ["pre", "peer_exposure"]
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert {"old_raw", "old_cal", "new_cal", "tau_panel", "accept_panel", "pre_answer",
            "post_answer", "gold_display"} <= set(pred.columns)
    assert set(pred.family) == {"output_linear", "activation", "stacked"}
    panel = pd.read_csv(tmp_path / "panel.csv")
    assert {"panel_answer", "panel_correct"} <= set(panel.columns)
    assert panel.question_id.nunique() == 60
    ch = json.loads((tmp_path / "choices.json").read_text())
    assert all("calibrators" in c and "tau" in c for c in ch)


def test_position_is_respected_and_missing_states_drop_whole_questions(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
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
    assert R.main(["--run", str(stub_view / "s-debate"), "--out", str(out), "--mode", "split",
                   "--split-manifest", str(split), "--families", "output_linear", "activation"]) == 0
    pred = pd.read_csv(out / "predictions.csv")
    assert set(pred.question_id.map(assign)) == {"test"}


def test_transfer_applies_frozen_gates_to_another_condition(stub_view, tmp_path):
    R = _load("reanalysis")
    assert R.main(["--run", str(stub_view / "s-debate"), "--out", str(tmp_path), "--seeds", "0",
                   "--families", "output_linear", "--transfer-runs", str(stub_view / "s-self_reconsider")]) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    tr = res["results"][0]["transfer"]["s-self_reconsider"]
    assert "output_linear" in tr["families"] and tr["reference"]["accept_all"]["panel_accuracy"] >= 0
