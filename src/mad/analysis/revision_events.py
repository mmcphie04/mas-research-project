"""Revision events across every Gemma-3-12B dataset, for the revision-gate work (exploratory).

One row per (dataset, condition, question, agent): the answer before and after the revision, its
kind (helpful = wrong -> right, harmful = right -> wrong, wrong_wrong = changed between wrong
answers, unchanged), the gate features fixed on 2026-10-30, and the panel context needed to
simulate a gate on the panel decision.

main250's TEST questions (``main250-split.json``, 98) are reserved: ``load_events`` returns only
their counts (``test_counts``); none of their rows enter the event table.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.actions import expected_vote

# (dataset, view prefix relative to the repo, job name, protocol words, conditions)
# condition -> (peers shown?, structured prompt?, note)
SOURCES = [
    ("pilot12b", "runs/colab", "pilot12b", 150, {
        "debate": (True, False, ""), "debate_answer_only": (True, False, "peers' answers only"),
        "misleading_one": (True, False, "one fabricated peer"), "self_reconsider": (False, False, "no peers")}),
    ("val12b", "runs/val12b", "val12b", 150, {
        "debate": (True, False, ""), "independent": (False, False, "independent re-answer, not a revision")}),
    ("val12b", "runs/val12b", "val12b-self", 150, {"self_reconsider": (False, False, "no peers; 100 questions")}),
    ("pilot250", "runs/pilot250/out", "pilot250", 250, {
        "debate3": (True, False, ""), "structured3": (True, True, ""), "debate5": (True, False, ""),
        "structured5": (True, True, "")}),
    ("pilot250_r2", "runs/pilot250/out", "pilot250_r2", 250, {
        "debate3_r2": (True, False, "second round"), "structured3_r2": (True, True, "second round"),
        "debate5_r2": (True, False, "second round"), "structured5_r2": (True, True, "second round"),
        "self_structured5": (False, True, "no peers")}),
    ("main250", "runs/main250", "main250", 250, {
        "debate3": (True, False, ""), "structured3": (True, True, ""), "debate5": (True, False, ""),
        "structured5": (True, True, "")}),
]
TRAIN_SOURCES = {("pilot12b", "debate"), ("pilot12b", "debate_answer_only"), ("pilot12b", "misleading_one"),
                 ("val12b", "debate"), ("pilot250", "debate3"), ("pilot250", "structured3"),
                 ("pilot250", "debate5"), ("pilot250", "structured5"), ("main250", "debate3"),
                 ("main250", "structured3"), ("main250", "debate5"), ("main250", "structured5")}
FEATURES = ["conf_before", "conf_after", "conf_change", "p_old_exposure", "p_new_exposure", "p_diff_exposure",
            "n_peers_new", "n_peers_old", "new_is_peer_majority", "panel_unanimous", "len_before", "len_after",
            "lp_before", "lp_after", "panel_k", "structured", "protocol_250"]
COLS = ["question_id", "agent_id", "phase", "answer", "gold_display", "n_options", "verbalized_confidence",
        "readout_logprobs", "n_completion_tokens", "response_logprob_mean", "shown_peer_answers",
        "shown_peer_majority", "previous_round_answer", "round", "subject"]


def kind(before: int, after: int, gold: int) -> str:
    if before == after:
        return "unchanged"
    if after == gold:
        return "helpful"
    if before == gold:
        return "harmful"
    return "wrong_wrong"


def _ints(s) -> list[int]:
    if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
        return []
    return [int(float(x)) for x in str(s).split(",")]


def _probs(s) -> np.ndarray | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
        return None
    return np.exp(np.array([float(v) for v in str(s).split(",")]))


def _plurality(answers) -> int:
    """The unique plurality answer, or -1 when the top count is tied."""
    c = Counter(int(a) for a in answers).most_common()
    return c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else -1


def _num(x) -> float:
    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else np.nan


def load_condition(root: Path, prefix: str, job: str, cond: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    path = root / prefix / f"{job}-{cond}" / "agent_states.parquet"
    have = set(pq.read_schema(path).names)
    s = pd.read_parquet(path, columns=[c for c in COLS if c in have])
    for c in COLS:
        if c not in have:
            s[c] = None
    return s.assign(question_id=s.question_id.astype(str))


def condition_events(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = {ph: g.set_index(["question_id", "agent_id"]) for ph, g in s.groupby("phase")}
    pre, post = idx["pre"], idx["post"]
    expo = idx.get("peer_exposure")
    for q, g in post.groupby(level=0):
        agents = sorted(a for _, a in g.index)
        gold = int(g.gold_display.iloc[0])
        before = {}
        for a in agents:
            r = post.loc[(q, a)]
            prev = r.previous_round_answer
            second = r["round"] is not None and not pd.isna(r["round"]) and int(r["round"]) >= 2
            before[a] = int(prev) if second and prev is not None and not pd.isna(prev) and int(prev) >= 0 \
                else int(pre.loc[(q, a)].answer)
        panel_before = [before[a] for a in agents]
        prev_majority = _plurality(panel_before)
        for a in agents:
            r, r0 = post.loc[(q, a)], pre.loc[(q, a)]
            b, af = before[a], int(r.answer)
            shown = _ints(r.shown_peer_answers)
            p = _probs(expo.loc[(q, a)].readout_logprobs) if expo is not None and (q, a) in expo.index else None
            maj = _num(r.shown_peer_majority)
            cb, ca = _num(r0.verbalized_confidence), _num(r.verbalized_confidence)
            rows.append({
                "question_id": q, "agent_id": int(a), "gold": gold, "n_options": int(r.n_options),
                "subject": r.subject, "before": b, "after": af, "kind": kind(b, af, gold),
                "changed": int(b != af), "panel_k": len(agents),
                "panel_before": panel_before, "prev_round_majority": prev_majority,
                "panel_unanimous": float(len(set(panel_before)) == 1),
                "conf_before": cb, "conf_after": ca, "conf_change": ca - cb,
                "p_old_exposure": float(p[b]) if p is not None else np.nan,
                "p_new_exposure": float(p[af]) if p is not None else np.nan,
                "p_diff_exposure": float(p[af] - p[b]) if p is not None else np.nan,
                "n_peers_new": float(sum(x == af for x in shown)), "n_peers_old": float(sum(x == b for x in shown)),
                "new_is_peer_majority": float(not np.isnan(maj) and int(maj) >= 0 and int(maj) == af),
                "peer_majority": int(maj) if not np.isnan(maj) else -1,
                "len_before": _num(r0.n_completion_tokens), "len_after": _num(r.n_completion_tokens),
                "lp_before": _num(r0.response_logprob_mean), "lp_after": _num(r.response_logprob_mean)})
    return pd.DataFrame(rows)


def load_events(root=".", split_manifest="artifacts/main250/plan/main250-split.json") -> dict:
    """{"events": all non-test rows, "test_counts": {condition: counts}, "sources": [...]}."""
    root = Path(root)
    assign = json.loads((root / split_manifest).read_text())["assignment"]
    frames, test_counts, sources = [], {}, []
    for dataset, prefix, job, words, conds in SOURCES:
        for cond, (peers, structured, note) in conds.items():
            ev = condition_events(load_condition(root, prefix, job, cond))
            ev = ev.assign(dataset=dataset, condition=cond, protocol=words, peers=peers, structured=float(structured),
                           protocol_250=float(words == 250), note=note,
                           split="non-main250" if dataset != "main250" else ev.question_id.map(assign))
            if dataset == "main250":
                if ev.split.isna().any():
                    raise SystemExit("a main250 question is missing from the split manifest")
                t = ev[ev.split == "test"]
                test_counts[cond] = _counts(t)  # counts only; the rows go no further
                ev = ev[ev.split != "test"]
            sources.append({"dataset": dataset, "condition": cond, "protocol": words, "peers": peers,
                            "structured": structured, "note": note})
            frames.append(ev)
    return {"events": pd.concat(frames, ignore_index=True), "test_counts": test_counts, "sources": sources}


def _counts(ev: pd.DataFrame) -> dict:
    k = ev.kind.value_counts()
    hh = ev[ev.kind.isin(["helpful", "harmful"])]
    return {"questions": int(ev.question_id.nunique()), "answers": int(len(ev)),
            "panel_k": int(ev.panel_k.iloc[0]) if len(ev) else None, "changed": int(ev.changed.sum()),
            "helpful": int(k.get("helpful", 0)), "harmful": int(k.get("harmful", 0)),
            "wrong_wrong": int(k.get("wrong_wrong", 0)),
            "questions_with_helpful_or_harmful": int(hh.question_id.nunique())}


def inventory(loaded: dict) -> list[dict]:
    ev, out = loaded["events"], []
    for src in loaded["sources"]:
        sub = ev[(ev.dataset == src["dataset"]) & (ev.condition == src["condition"])]
        if src["dataset"] == "main250":
            out.append({**src, "part": "train+validation", **_counts(sub)})
            out.append({**src, "part": "test (counts only)", **loaded["test_counts"][src["condition"]]})
        else:
            out.append({**src, "part": "all", **_counts(sub)})
    return out


# ----------------------------------------------------------------------------- gate simulation


def panel_expectation(before, after, gold: int, p_accept) -> tuple[float, float]:
    """Expected (agent accuracy, panel score) when each CHANGED agent's revision is accepted
    independently with probability p_accept[i] (unchanged agents keep their answer). Exact
    enumeration over the changed agents; panel score = expected plurality accuracy under random
    tie-breaking."""
    before, after = list(before), list(after)
    ch = [i for i in range(len(before)) if before[i] != after[i]]
    agent, panel = 0.0, 0.0
    for choice in product((0, 1), repeat=len(ch)):
        pr = 1.0
        final = before.copy()
        for i, c in zip(ch, choice, strict=True):
            pr *= p_accept[i] if c else 1 - p_accept[i]
            if c:
                final[i] = after[i]
        if pr == 0:
            continue
        agent += pr * np.mean([f == gold for f in final])
        panel += pr * expected_vote(final, gold)
    return agent, panel


def perfect_gate(before, after, gold: int) -> tuple[float, float]:
    """Accept helpful, reject harmful; for wrong->wrong revisions, the accept/reject combination
    giving the best panel score."""
    p = [1.0 if (b != a and a == gold) else 0.0 for b, a in zip(before, after, strict=True)]
    ww = [i for i, (b, a) in enumerate(zip(before, after, strict=True)) if b != a and b != gold and a != gold]
    best = None
    for choice in product((0.0, 1.0), repeat=len(ww)):
        q = p.copy()
        for i, c in zip(ww, choice, strict=True):
            q[i] = c
        r = panel_expectation(before, after, gold, q)
        if best is None or r[1] > best[1]:
            best = r
    return best


def random_gate(before, after, gold: int, acc: float, ww_accept: float = 0.5) -> tuple[float, float]:
    """A gate right with probability ``acc`` on each helpful/harmful revision (accepts helpful,
    rejects harmful), accepting wrong->wrong revisions with probability ``ww_accept``."""
    p = []
    for b, a in zip(before, after, strict=True):
        p.append(acc if (b != a and a == gold) else (1 - acc) if (b != a and b == gold) else ww_accept)
    return panel_expectation(before, after, gold, p)
