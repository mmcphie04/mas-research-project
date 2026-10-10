import numpy as np

from mad.store import ActivationStore, ActivationWriter, from_bf16_bits, storage_bytes, to_bf16_bits


def test_round_trip_is_bitwise_and_layer_major(tmp_path):
    n_layers, d_model, n = 5, 16, 23
    rng = np.random.default_rng(0)
    # values exactly representable in bfloat16 (the storage format) round-trip bitwise
    originals = [from_bf16_bits(to_bf16_bits(rng.normal(size=(n_layers + 1, d_model)))) for _ in range(n)]

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


def test_merge_runs_concatenates_chunks(tmp_path):
    """Chunked overnight collection is only survivable if the chunks recombine exactly."""
    import pandas as pd
    from conftest import make_cfg

    from mad.collect import STATES_FILE, collect_states
    from mad.data import Question
    from mad.store import merge_runs

    def qs(lo, hi):
        return [
            Question(f"q{i}", "s", f"Q{i}?", tuple(f"o{j}" for j in range(4)), i % 4)
            for i in range(lo, hi)
        ]

    dirs = []
    for i, (lo, hi) in enumerate([(0, 10), (10, 20), (20, 26)]):
        cfg = make_cfg(tmp_path, run_id=f"chunk{i}")
        dirs.append(collect_states(cfg, qs(lo, hi), n_variants=2).run_dir)

    merged = merge_runs(dirs, tmp_path / "merged", STATES_FILE)
    assert len(merged) == 26 * 2
    assert merged.question_id.nunique() == 26

    store = ActivationStore(tmp_path / "merged")
    assert len(store) == 52
    # every chunk's activations must survive the shard renumbering bit-exactly
    per_chunk = [ActivationStore(d).load_layer(3) for d in dirs]
    expected = np.concatenate(per_chunk, axis=0)
    assert np.array_equal(store.load_layer(3), expected)
    assert pd.read_parquet(tmp_path / "merged" / STATES_FILE).shape[0] == 52


def test_merge_refuses_mismatched_shapes(tmp_path):
    from conftest import make_cfg

    from mad.collect import STATES_FILE, collect_states
    from mad.data import Question
    from mad.store import merge_runs

    q = [Question("q0", "s", "Q?", ("a", "b", "c", "d"), 0)]
    a = collect_states(make_cfg(tmp_path, run_id="a"), q, n_variants=1).run_dir
    cfg_b = make_cfg(tmp_path, run_id="b")
    cfg_b.stub.d_model = 32  # different width
    b = collect_states(cfg_b, q, n_variants=1).run_dir
    try:
        merge_runs([a, b], tmp_path / "bad", STATES_FILE)
    except ValueError as exc:
        assert "differing shapes" in str(exc)
    else:
        raise AssertionError("merging runs with different d_model must fail loudly")


def test_massive_activations_survive_storage(tmp_path):
    """Regression: gemma-3-12b reaches |x| = 163,840 at the readout position (layers 40-47) and
    over 800,000 elsewhere; float16 (max 65,504) stored those as infinity."""
    big = np.array([[1.5, -163840.0, 815104.0, 3.0e5]], dtype=np.float32)
    bits = to_bf16_bits(big)
    back = from_bf16_bits(bits)
    assert np.isfinite(back).all() and np.allclose(back, big, rtol=1e-2)
    assert np.isnan(from_bf16_bits(to_bf16_bits(np.array([np.nan], np.float32)))).all()
    w = ActivationWriter(tmp_path, 1, 4)
    w.add({"run_id": "r", "condition": "c", "question_id": "q", "agent_id": 0, "phase": "pre",
           "position": "readout_last"}, np.vstack([big, big]))
    w.close()
    x = ActivationStore(tmp_path).load_layer(1)
    assert np.isfinite(x).all() and np.allclose(x[0], big[0], rtol=1e-2)


def test_old_float16_shards_still_load(tmp_path):
    w = ActivationWriter(tmp_path, 1, 2)
    w.add({"run_id": "r", "condition": "c", "question_id": "q", "agent_id": 0, "phase": "pre",
           "position": "readout_last"}, np.ones((2, 2)))
    w.close()
    shard = tmp_path / "acts_0000.npy"
    np.save(shard, np.full((2, 1, 2), 0.5, dtype=np.float16))  # what an old run wrote
    assert np.allclose(ActivationStore(tmp_path).load_layer(0), 0.5)
