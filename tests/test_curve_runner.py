"""Curve readout runner (mad.curve_runner): bound outputs, validated rows, safe stopping and
resume, coverage-checked analysis, and the smoke report. Synthetic builds, fake backends."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
import pytest

from mad import curve_runner as CR


class CharTok:
    all_special_ids: list[int] = []

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


LETTER_IDS = [ord(c) for c in "ABCDEFGHIJ"]


def _lp(key: str, n: int) -> np.ndarray:
    z = np.random.default_rng(int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)).normal(size=n)
    return z - np.log(np.exp(z).sum())


class Fake:
    """Deterministic in the input sequence; optional delay, interruption and a lookup table."""

    def __init__(self, name="fake", delay=0.0, interrupt_on=None, table=None):
        self.name, self.delay, self.interrupt_on, self.table, self.calls = name, delay, interrupt_on, table or {}, 0

    def describe(self):
        return {"backend": self.name, "repo_id": "test", "model_commit": "x", "chat_template_sha256": "y", "dtype": "f32"}

    def letter_logprobs_ids(self, seqs, n_options, batch_size=None):
        self.calls += 1
        if self.interrupt_on is not None and self.calls == self.interrupt_on:
            raise KeyboardInterrupt
        time.sleep(self.delay)
        out = []
        for s, n in zip(seqs, n_options, strict=True):
            h = CR.seq_hash(s)
            out.append(np.asarray(self.table[h], dtype=np.float32) if h in self.table else _lp(h, n).astype(np.float32))
        return out


def _records(tweak=""):
    recs = []
    for pool in ("pilot", "val"):
        for cond in ("debate", "self_reconsider"):
            for q in range(4):
                for a in range(3):
                    prev, ans = (q + a) % 4, (q + a + (a == 1)) % 4
                    text = f"Step one {tweak}compute {q}. Step two check {a} carefully now.\n\nAnswer: ({'ABCD'[ans]})\nConfidence: 70"
                    recs.append({"pool": pool, "condition": cond, "question_id": f"q{q}", "agent_id": a, "n_options": 4,
                                 "gold": q % 4, "answer": ans, "prev_answer": prev, "eligibility": "eligible",
                                 "kind": "helpful" if ans == q % 4 and prev != ans else "unchanged_wrong",
                                 "changed": int(prev != ans), "stop_reason": "stop_token",
                                 "prompt": f"<user>{pool}{cond}{q}{a}<model>\n", "text": text,
                                 "token_ids": [ord(c) for c in text], "prefill": "",
                                 "lp_exposure": _lp(f"e{pool}{cond}{q}{a}", 4).tolist(),
                                 "lp_before_stated": _lp(f"b{pool}{cond}{q}{a}", 4).tolist()})
    return recs


def _build(d, tweak="", source_sets=("core",)):
    R, M, flat, off = CR.build_tables(_records(tweak), CharTok(), LETTER_IDS,
                                      ("standard", "bridge", "exact_replay", "retokenized"))
    CR.write_build(d, R, M, flat, off, {"source_sets": list(source_sets), "variants": ["standard"]})
    return CR.load_build(d)


def _ident(b):
    return CR.backend_identity(b)


def test_complete_run_and_identical_inputs_are_read_once(tmp_path):
    B = _build(tmp_path / "b")
    f = Fake()
    st = CR.run(B, tmp_path / "o", f, _ident(f), batch_size=4)
    assert st["state"] == "complete" and st["modes"]["run"]["missing"] == 0
    rows = CR.load_rows(tmp_path / "o", B)
    assert set(rows.item) == set(CR.select(B, "all").item)
    assert st["copied_from_identical_inputs"] > 0  # standardized f=1 == exact replay here


def test_time_limit_flushes_then_resume_matches_an_uninterrupted_run(tmp_path):
    B = _build(tmp_path / "b")
    slow = Fake(delay=0.05)
    st = CR.run(B, tmp_path / "o", slow, _ident(slow), batch_size=2, max_minutes=0.15 / 60)
    assert st["state"] == "time_limit" and st["modes"]["run"]["missing"] > 0
    assert st["modes"]["run"]["done"] > 0  # completed batches were persisted
    status = json.loads((tmp_path / "o" / "run_status.json").read_text())
    assert status["state"] == "time_limit" and status["modes"]["run"]["missing_sample"]
    st2 = CR.run(B, tmp_path / "o", Fake(), _ident(slow), batch_size=8)
    assert st2["state"] == "complete"
    ref = CR.run(B, tmp_path / "ref", Fake(), _ident(slow), batch_size=8)
    assert ref["state"] == "complete"
    a = CR.load_rows(tmp_path / "o", B).drop_duplicates("item").set_index("item").logprobs
    b = CR.load_rows(tmp_path / "ref", B).drop_duplicates("item").set_index("item").logprobs
    assert all(np.allclose(a[i], b[i]) for i in b.index)


def test_interruption_keeps_completed_batches(tmp_path):
    B = _build(tmp_path / "b")
    f = Fake(interrupt_on=3)
    st = CR.run(B, tmp_path / "o", f, _ident(f), batch_size=4)
    assert st["state"] == "interrupted"
    assert len(CR.load_rows(tmp_path / "o", B)) >= 8  # two full batches survived
    assert CR.run(B, tmp_path / "o", Fake(), _ident(f), batch_size=4)["state"] == "complete"


def test_changed_inputs_are_refused(tmp_path):
    B = _build(tmp_path / "b")
    f = Fake()
    CR.run(B, tmp_path / "o", f, _ident(f), batch_size=4)
    B2 = _build(tmp_path / "b", tweak="x ")  # rebuilt in place: different inputs, new build_id
    assert B2.build_id != B.build_id
    with pytest.raises(CR.RunMismatch, match="build_id"):
        CR.run(B2, tmp_path / "o", f, _ident(f), batch_size=4)


def test_backend_change_including_stub_outputs_is_refused(tmp_path):
    B = _build(tmp_path / "b")
    stub = Fake(name="stub")
    CR.run(B, tmp_path / "o", stub, _ident(stub), batch_size=4)
    model = Fake(name="hf")
    with pytest.raises(CR.RunMismatch, match="identity"):
        CR.run(B, tmp_path / "o", model, _ident(model), batch_size=4)
    with pytest.raises(CR.RunMismatch, match="stub"):
        CR.analyze(B, tmp_path / "o", tmp_path / "a")


def test_rows_are_validated_by_hash_and_option_count(tmp_path):
    B = _build(tmp_path / "b")
    f = Fake()
    CR.run(B, tmp_path / "o", f, _ident(f), batch_size=4)
    part = sorted((tmp_path / "o").glob("rows_*.parquet"))[0]
    df = pd.read_parquet(part)
    df.loc[0, "seq_hash"] = "0" * 40
    df.to_parquet(part, index=False)
    with pytest.raises(CR.InvalidRows, match="sequence hash"):
        CR.load_rows(tmp_path / "o", B)
    df = pd.read_parquet(part)
    df.loc[0, "seq_hash"] = B.M.set_index("item").loc[int(df.loc[0, "item"]), "seq_hash"]
    df["n_options"] = 5
    df.to_parquet(part, index=False)
    with pytest.raises(CR.InvalidRows, match="option count"):
        CR.load_rows(tmp_path / "o", B)


def test_altered_build_files_are_detected(tmp_path):
    _build(tmp_path / "b")
    M = pd.read_parquet(tmp_path / "b" / "items_meta.parquet")
    M.loc[0, "n_options"] = 9
    M.to_parquet(tmp_path / "b" / "items_meta.parquet", index=False)
    with pytest.raises(CR.RunMismatch, match="altered"):
        CR.load_build(tmp_path / "b")


def test_core_run_refuses_a_build_with_other_sources(tmp_path):
    B = _build(tmp_path / "b", source_sets=("core", "extras"))
    f = Fake()
    with pytest.raises(CR.RunMismatch, match="source sets"):
        CR.run(B, tmp_path / "o", f, _ident(f), expect_sources=["core"])


def test_analysis_requires_complete_coverage_unless_labelled_partial(tmp_path):
    B = _build(tmp_path / "b")
    f = Fake()
    CR.run(B, tmp_path / "o", f, _ident(f), batch_size=8)
    full = CR.analyze(B, tmp_path / "o", tmp_path / "a", n_boot=50)
    assert full["label"] == "FINAL" and full["missing_items"] == 0
    # remove ONE response's f = 0.5 reading: final analysis must refuse; the labelled partial
    # analysis must drop exactly that response at every fraction
    rows = CR.load_rows(tmp_path / "o", B)
    drop = B.M[(B.M.variant == "standard") & (B.M.fraction == 0.5)].iloc[0]
    rows = rows[rows.item != drop["item"]]
    for p in (tmp_path / "o").glob("rows_*.parquet"):
        p.unlink()
    rows.to_parquet(tmp_path / "o" / "rows_run_merged.parquet", index=False)
    with pytest.raises(CR.IncompleteRun, match="1 of"):
        CR.analyze(B, tmp_path / "o", tmp_path / "a")
    res = CR.analyze(B, tmp_path / "o", tmp_path / "a", partial_smoke=True, n_boot=50)
    assert res["label"] == "PARTIAL_SMOKE" and res["responses"] == full["responses"] - 1
    cur = pd.read_csv(tmp_path / "a" / "curves_PARTIAL_SMOKE.csv")
    sub = cur[(cur.metric == "is_eventual") & (cur.subset == "all") & (cur.pool == "pooled")]
    for _, g in sub.groupby("condition"):
        assert g.n.nunique() == 1  # the same responses at every fraction
    assert (tmp_path / "a" / "analysis.json").exists()  # the FINAL analysis is kept apart


def test_smoke_time_limit_stops_every_remaining_mode(tmp_path):
    B = _build(tmp_path / "b")
    slow = Fake(delay=0.05)
    rep = CR.smoke(B, tmp_path / "s", slow, _ident(slow), batch_sizes=(2, 4), max_minutes=0.1 / 60)
    assert rep["overall"] == "FAIL" and rep["state"] == "time_limit"
    rows = CR.load_rows(tmp_path / "s", B)
    assert not (rows["mode"] == "batch4").any()  # the outer loop stopped too


def test_smoke_report_passes_only_when_reproduction_holds(tmp_path):
    B = _build(tmp_path / "b")
    # a backend that returns the saved distributions for the bridge f0 and exact-replay inputs
    table = {}
    R = B.R.set_index(CR.KEY)
    for r in B.M.itertuples():
        k = (r.pool, r.condition, r.question_id, r.agent_id)
        if r.variant == "bridge" and r.fraction == 0.0:
            table[r.seq_hash] = json.loads(R.loc[k, "lp_exposure"])
        elif r.variant == "exact_replay":
            table[r.seq_hash] = json.loads(R.loc[k, "lp_before_stated"])
    good = Fake(table=table)
    rep = CR.smoke(B, tmp_path / "s1", good, _ident(good), batch_sizes=(2, 4))
    assert rep["state"] == "complete"
    assert rep["overall"] == "PASS", {k: v for k, v in rep["checks"].items() if v["gated"] and not v["pass"]}
    bad = Fake()
    rep2 = CR.smoke(B, tmp_path / "s2", bad, _ident(bad), batch_sizes=(2,))
    failed = {k for k, v in rep2["checks"].items() if v["gated"] and not v["pass"]}
    assert rep2["overall"] == "FAIL" and failed == {"bridge_f0_vs_saved_exposure",
                                                    "exact_replay_vs_saved_before_stated_answer"}
    sel = pd.read_csv(tmp_path / "s1" / "smoke_selection.csv")
    assert {"stratified_response", "longest_input"} <= set(sel.why)
