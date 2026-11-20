"""Experiment A/B and judge-confirmation request builders, against the committed question set and
the CPU stub engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mad import prompts as P
from mad.gen import engine as E
from mad.gen import prompts_ab as PA
from mad.gen import tasks as T

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "artifacts/expab/plan"


@pytest.fixture(scope="module")
def qs():
    q, assign, m = T.load_questions(PLAN)
    return q[:12], assign


def bank_rows(qs, mode, n, letters=None):
    """Synthetic bank rows (answers chosen by the caller) in the engine's row format."""
    rows = []
    for i, q in enumerate(qs):
        for a in range(n):
            ans = letters(i, a) if letters else a % 2
            rows.append({"mode": mode, "question_id": q.question_id, "agent": str(a), "parsed": ans >= 0, "answer": ans,
                         "text": f"Reasoning {mode} {a}.\nAnswer: ({'ABCDEFGHIJ'[max(ans, 0)]})\nConfidence: 70"})
    return rows


def test_question_set_is_checked_and_split(qs):
    q, assign = qs
    allq, assignment, m = T.load_questions(PLAN)
    assert len(allq) == 1500 and sum(v == "dev" for v in assignment.values()) == 600
    assert all(0 <= T.gold_display(x) < x.n_options for x in allq[:50])


def test_bank_requests_use_the_main_protocol(qs):
    q, _ = qs
    r = T.bank_requests(q[:2], "qwen_t")
    assert len(r) == 10 and r[0].messages == T.round0(q[0]) and "250 words" in r[0].messages[0]["content"]
    rec = next(json.loads(x) for x in open(ROOT / "runs/main250/main250-debate3/transcripts.jsonl")
               if json.loads(x)["phase"] == "pre")
    assert r[0].messages[0]["content"] == next(m["content"] for m in rec["messages"] if m["role"] == "system")
    assert len({x.request_id for x in T.bank_requests(q, "gemma")}) == 120


def test_chooser_candidates_orders_and_parsing(qs):
    q, _ = qs
    bank = T.bank_index(bank_rows(q, "qwen_nt", 10, letters=lambda i, a: [0, 0, 0, 1, 1, 2, -1, 0, 3, 1][a]))
    pool = T.pool_rows(bank, q[0].question_id, T.ARMS["A1_qnt10"][1])
    assert T.disagreeing(pool)
    c = T.candidates(pool, q[0].question_id, "A1_qnt10")
    assert [x["answer"] for x in c] == [0, 1, 2, 3] and [x["votes"] for x in c] == [4, 3, 1, 1]
    assert all("Confidence" not in x["text"] for x in c)
    reqs = T.chooser_requests(q[0], "A1_qnt10", c)
    assert len(reqs) == 3
    orders = [json.loads(r.meta["order_answers"]) for r in reqs]
    assert all(sorted(o) == [0, 1, 2, 3] for o in orders) and len({tuple(o) for o in orders}) > 1
    user = reqs[0].messages[1]["content"]
    assert "votes" not in user.lower() and user.count("=== Candidate") == 4 and "Best candidate: N" in user
    assert T.parse_pick("I think Best candidate: 2 ... no wait.\nBest candidate: 3", 4) == 3
    assert T.parse_pick("Best candidate: 9", 4) == -1 and T.parse_pick("Best candidate: (4)", 4) == 4
    # a cross-family pool: 5 Gemma + 5 Qwen
    bank.update(T.bank_index(bank_rows(q, "gemma", 10)))
    assert len(T.pool_rows(bank, q[0].question_id, T.ARMS["A3"][1])) == 10


def test_unanimous_pool_is_not_chosen(qs):
    q, _ = qs
    bank = T.bank_index(bank_rows(q, "qwen_nt", 10, letters=lambda i, a: 2))
    assert not T.disagreeing(T.pool_rows(bank, q[0].question_id, [("qwen_nt", 10)]))


def test_b_panels_and_arms(qs):
    q, assign = qs
    bank = T.bank_index(bank_rows(q, "qwen_nt", 3, letters=lambda i, a: [0, 0, 1][a] if i % 2 else 0))
    assert T.panel(bank, "qwen_nt", q[0].question_id) is None          # unanimous
    rows = T.panel(bank, "qwen_nt", q[1].question_id)
    assert rows is not None
    out = {arm: T.b_requests(q[1], "qwen_nt", rows, arm) for arm in T.B_ARMS}
    for arm, rs in out.items():
        assert len(rs) == 3 and rs[0].messages[2]["role"] == "assistant"
        last = rs[0].messages[-1]["content"]
        assert ("Panel member" in last) == (arm in ("debate", "dissent"))
        assert ("--- Response A" in last) == (arm == "anon")
        assert (PA.DISSENT in last) == (arm == "dissent")
        assert ("Confidence: 70" not in last)  # peers' confidence lines removed
    assert json.loads(out["alone"][0].meta["shown_sources"]) == []
    assert json.loads(out["debate"][0].meta["shown_sources"]) == [1, 2]
    assert out["debate"][0].messages[-1]["content"].startswith(P.PEER_BLOCK.split("{peers}")[0])
    assert out["anon"][0].messages[-1]["content"].startswith(P.PEER_BLOCK.split("{peers}")[0])
    sel = T.select_panels(bank, q, {x.question_id: "dev" for x in q}, "dev", "qwen_nt", 3, seed=1)
    assert len(sel) == 3 and all(T.panel(bank, "qwen_nt", s) for s in sel)


def test_stub_end_to_end_bank_then_chooser(qs):
    q, _ = qs
    eng = E.StubEngine(n_letters=4, trace_len=(10, 300))
    rows, calls = E.run_requests(eng, E.MODES["qwen_t"], T.bank_requests(q[:4], "qwen_t"))
    assert all(r["meta_gold"] == T.gold_display(next(x for x in q if x.question_id == r["question_id"])) for r in rows)
    bank = T.bank_index(rows)
    arm_rows = []
    for x in q[:4]:
        pool = T.pool_rows(bank, x.question_id, T.ARMS["A1_qt5"][1])
        if T.disagreeing(pool):
            arm_rows += T.chooser_requests(x, "A1_qt5", T.candidates(pool, x.question_id, "A1_qt5"))
    assert arm_rows
    out, _ = E.run_requests(eng, E.MODES["qwen_t"], arm_rows)
    assert all(r["meta_arm"] == "A1_qt5" and r["thinking_tokens"] > 0 for r in out)


def test_judge_confirm_requests():
    ev = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/events.jsonl")][:3]
    qq = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/questions.jsonl")][:2]
    r = T.judge_confirm_requests(ev, qq)
    assert len(r) == 3 * 2 + 2 * 3
    assert {x.condition for x in r} == {"judge_confirm/judge", "judge_confirm/own"}
