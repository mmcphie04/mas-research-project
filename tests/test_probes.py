import numpy as np
import pandas as pd

from mad.analysis.labels import build_events
from mad.analysis.probes import FeatureBank, fit_probe, layer_sweep, representation_target_grid
from mad.analysis.stats import conditional_auroc
from mad.debate import ROWS_FILE
from mad.store import ActivationStore

PEAK_LAYER = 5  # signal_peak_layer 0.6 of 8 layers


def _bank(cfg, result):
    rows = pd.read_parquet(result.run_dir / ROWS_FILE)
    events = build_events(rows)
    return FeatureBank(ActivationStore(result.run_dir), events), events


def test_probe_recovers_a_planted_instance_level_signal(stub_run):
    cfg, result = stub_run
    bank, ev = _bank(cfg, result)
    X = bank.features(PEAK_LAYER, "post")
    probe = fit_probe(X, ev["post_correct"].to_numpy(), ev["question_id"].to_numpy())
    assert probe.cv_auroc > 0.75, (
        f"planted probe_signal=1.4 not recovered (got {probe.cv_auroc:.3f})"
    )


def test_shuffled_labels_give_chance_performance(stub_run):
    cfg, result = stub_run
    bank, ev = _bank(cfg, result)
    X = bank.features(PEAK_LAYER, "post")
    rng = np.random.default_rng(0)
    y = rng.permutation(ev["post_correct"].to_numpy())
    probe = fit_probe(X, y, ev["question_id"].to_numpy())
    assert 0.35 < probe.cv_auroc < 0.65, (
        f"permutation control is not at chance: {probe.cv_auroc:.3f}"
    )


def test_layer_sweep_peaks_where_the_signal_was_planted(stub_run):
    cfg, result = stub_run
    bank, ev = _bank(cfg, result)
    sweep = layer_sweep(
        bank, ev["post_correct"].to_numpy(), ev["question_id"].to_numpy(), representation="post"
    )
    best = int(sweep.loc[sweep["cv_auroc"].idxmax(), "layer"])
    assert abs(best - PEAK_LAYER) <= 1, f"peak at layer {best}, planted at {PEAK_LAYER}"
    assert sweep.loc[sweep.layer == 0, "cv_auroc"].iloc[0] < sweep["cv_auroc"].max()


def test_representation_target_grid_is_filled(stub_run):
    cfg, result = stub_run
    bank, ev = _bank(cfg, result)
    grid = representation_target_grid(
        bank, ev, layer=PEAK_LAYER, representations=("pre", "post", "paired")
    )
    assert len(grid) == 6
    cell = grid.set_index(["representation", "target"])["cv_auroc"]
    # (a) pre -> y_pre and (d) post -> y_post are the single-state correctness probes
    assert cell[("pre", "y_pre")] > 0.7
    assert cell[("post", "y_post")] > 0.7
    # the pre state carries no information about which revision will be proposed
    assert cell[("pre", "y_post")] < cell[("post", "y_post")]


def test_difficulty_only_signal_is_caught_by_conditional_auroc(difficulty_run):
    """The primary shortcut diagnostic, tested against a shortcut planted on purpose.

    With probe_signal=0 and difficulty_leak=3 the activations encode nothing but item difficulty.
    A marginal AUROC well above chance with a conditional AUROC at chance is exactly the
    difficulty-detector signature of docs/RISKS.md risk 1.
    """
    cfg, result = difficulty_run
    bank, ev = _bank(cfg, result)
    X = bank.features(PEAK_LAYER, "post")
    y = ev["post_correct"].to_numpy()
    groups = ev["question_id"].to_numpy()
    probe = fit_probe(X, y, groups)
    marginal = probe.cv_auroc
    cond, n_pairs, n_groups = conditional_auroc(
        y[~np.isnan(probe.oof)], probe.oof[~np.isnan(probe.oof)], groups[~np.isnan(probe.oof)]
    )
    # a perfect read of item difficulty tops out near 0.72 marginal / exactly 0.50 conditional
    assert marginal > 0.62, (
        f"planted difficulty signal should be visible marginally ({marginal:.3f})"
    )
    assert n_pairs > 100 and n_groups > 30
    assert abs(cond - 0.5) < 0.07, f"conditional AUROC {cond:.3f} should be near chance"
    assert marginal - cond > 0.10, "the marginal/conditional gap is the diagnostic"


def test_real_signal_survives_the_conditional_check(stub_run):
    cfg, result = stub_run
    bank, ev = _bank(cfg, result)
    X = bank.features(PEAK_LAYER, "post")
    y = ev["post_correct"].to_numpy()
    groups = ev["question_id"].to_numpy()
    probe = fit_probe(X, y, groups)
    ok = ~np.isnan(probe.oof)
    cond, _, _ = conditional_auroc(y[ok], probe.oof[ok], groups[ok])
    assert cond > 0.68, f"instance-level signal should survive conditioning (got {cond:.3f})"
