"""Judge analysis mechanics on SYNTHETIC judge/own outputs (random probabilities and answers) over
the committed event sample: every section computes, panels are complete, the decision bar is
evaluated. Says nothing about real judges."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("judge_analysis", ROOT / "scripts" / "judge_analysis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_analysis_on_synthetic_outputs(tmp_path, monkeypatch):
    A = _load()
    monkeypatch.setattr(A, "N_BOOT", 50)
    import revision_gate as RG
    monkeypatch.setattr(RG, "N_BOOT", 50)
    rng = np.random.default_rng(1)
    events = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/events.jsonl")]
    qs = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/questions.jsonl")]
    reuse = {q["question_id"] for q in json.loads((ROOT / "artifacts/revision_gate/judge/gemma_reuse.json").read_text())["questions"]}
    for m in A.JUDGES:
        d = tmp_path / m
        d.mkdir()
        ev = [e for e in events if e["thinking_subsample"]] if m == "qwen3_think" else events
        rows = [{"event_id": e["event_id"], "order": o, "p1": (p := float(rng.random())), "p2": 1 - p}
                for e in ev for o in ("original_first", "revised_first")]
        (d / f"{m}_judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        qq = [q for q in qs if (q["thinking_subsample"] or m != "qwen3_think") and not (m == "gemma" and q["question_id"] in reuse)]
        own = [{"question_id": q["question_id"], "answer": int(rng.integers(-1, q["n_options"]))}
               for q in qq for _ in range(1 if m == "qwen3_think" else 3)]
        (d / f"{m}_own.jsonl").write_text("".join(json.dumps(r) + "\n" for r in own))
    r = A.analyze(tmp_path, ROOT)
    assert set(r["judges"]) == set(A.JUDGES) and not r["missing"]
    assert r["judges"]["gemma"]["helpful_harmful"] == 391 and r["judges"]["qwen3_think"]["events"] == 157
    assert r["judges"]["gemma"]["own_answers_cover"] == 1.0  # fresh + reused pool answers cover every question
    assert r["panels"]["gemma"]["panels"] > 0 and r["panels"]["qwen3_think"]["panels"] <= r["panels"]["gemma"]["panels"]
    assert set(r["judges"]["llama"]["decision_bar"]) == {"a_accuracy_ge_0.75", "b_clearly_beats_both_oof_gates",
                                                         "c_clearly_beats_own_answer_rule", "met"}
    assert "GQL" in r["cross_model"]["mixed_panels"]
    assert "Decision bar" in A.markdown(r)
