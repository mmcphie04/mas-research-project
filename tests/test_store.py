import numpy as np

from mad.store import ActivationStore, ActivationWriter, storage_bytes


def test_round_trip_is_bitwise_and_layer_major(tmp_path):
    n_layers, d_model, n = 5, 16, 23
    rng = np.random.default_rng(0)
    originals = [rng.normal(size=(n_layers + 1, d_model)).astype(np.float16) for _ in range(n)]

    w = ActivationWriter(tmp_path, n_layers, d_model, shard_size=7)  # forces multiple shards
    for i, h in enumerate(originals):
        w.add(
            {
                "run_id": "r",
                "condition": "debate",
                "question_id": f"q{i}",
                "agent_id": 0,
                "phase": "pre",
                "position": "readout_last",
            },
            h,
        )
    index = w.close()
    assert len(index) == n
    assert index["shard"].nunique() == 4

    store = ActivationStore(tmp_path)
    for layer in store.layer_range():
        got = store.load_layer(layer)
        want = np.stack([o[layer] for o in originals]).astype(np.float32)
        assert np.array_equal(got, want), f"layer {layer} did not round-trip bitwise"


def test_row_selection(tmp_path):
    w = ActivationWriter(tmp_path, 2, 4, shard_size=2)
    for i in range(6):
        w.add(
            {
                "run_id": "r",
                "condition": "debate",
                "question_id": f"q{i}",
                "agent_id": 0,
                "phase": "pre",
                "position": "readout_last",
            },
            np.full((3, 4), i, dtype=np.float16),
        )
    w.close()
    store = ActivationStore(tmp_path)
    picked = store.load_layer(0, rows=np.array([0, 3, 5]))
    assert picked.shape == (3, 4)
    assert [int(p[0]) for p in picked] == [0, 3, 5]


def test_shape_mismatch_is_rejected(tmp_path):
    w = ActivationWriter(tmp_path, 3, 8)
    try:
        w.add({"run_id": "r"}, np.zeros((3, 8), dtype=np.float16))
    except ValueError as exc:
        assert "expected" in str(exc)
    else:
        raise AssertionError("expected a shape error: 3 layers means 4 vectors, not 3")


def test_storage_arithmetic_matches_the_plan():
    # 18,000 states, 48 layers, d=3840, one position -> ~6.8 GB (docs/RISKS.md risk 16)
    gb = storage_bytes(18_000, 48, 3840, n_positions=1) / 1e9
    assert 6.5 < gb < 7.0
    # the same states with every token position stored would be three orders of magnitude worse
    tb = storage_bytes(18_000, 48, 3840, n_positions=800) / 1e12
    assert tb > 5.0
