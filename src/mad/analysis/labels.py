"""Revision labelling and panel voting.

An agent-state table (one row per question x agent x phase) is folded into an EVENT table
(one row per question x agent) carrying the pre/post answers and the revision label:

    beneficial  wrong -> right      harmful  right -> wrong      unhelpful  wrong -> wrong

Note the structural fact these labels obey (docs/PLAN.md section 3.2): among revised events,
beneficial == post_correct and harmful == pre_correct, so no binary partition of the three is
anything other than a single-state correctness label. The labels are for description; the
learning target is correctness.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

BENEFICIAL, HARMFUL, UNHELPFUL, NONE = "beneficial", "harmful", "unhelpful", "none"
LABELS = (BENEFICIAL, HARMFUL, UNHELPFUL)
EVENT_KEYS = ["run_id", "condition", "question_id", "agent_id"]


def build_events(rows: pd.DataFrame) -> pd.DataFrame:
    """Pivot the agent-state table into one row per (condition, question, agent)."""
    pre = rows[rows["phase"] == "pre"].set_index(EVENT_KEYS)
    post = rows[rows["phase"] == "post"].set_index(EVENT_KEYS)
    if not pre.index.equals(post.index.sortlevel()[0]) and set(pre.index) != set(post.index):
        raise ValueError("pre and post states do not cover the same (question, agent) keys")
    post = post.reindex(pre.index)

    ev = pd.DataFrame(index=pre.index)
    ev["subject"] = pre["subject"]
    ev["n_options"] = pre["n_options"]
    ev["gold_display"] = pre["gold_display"]
    # "answer" is the canonical answer (mad.debate.canonical_answer); runs collected before it
    # existed used the readout answer, which is what they get here
    col = "answer" if "answer" in pre.columns else "readout_answer"
    ev["pre_answer"] = pre[col]
    ev["post_answer"] = post[col]
    ev["pre_correct"] = pre["correct"]
    ev["post_correct"] = post["correct"]
    ev["pre_confidence"] = pre["readout_confidence"]
    ev["post_confidence"] = post["readout_confidence"]
    ev["pre_verbalized"] = pre["verbalized_confidence"]
    ev["post_verbalized"] = post["verbalized_confidence"]
    ev["peer_majority"] = post["peer_majority"]
    ev["n_peers_agree_pre"] = post["n_peers_agree_pre"]
    ev["peer_unanimous"] = post["peer_unanimous"]
    ev["post_equals_peer_majority"] = post["post_equals_peer_majority"]
    ev["pre_rationale_chars"] = pre["rationale_chars"]
    ev["post_rationale_chars"] = post["rationale_chars"]
    ev["pre_parse_ok"] = pre["freetext_parse_ok"]
    ev["post_parse_ok"] = post["freetext_parse_ok"]
    ev["revised"] = (ev["pre_answer"] != ev["post_answer"]).astype(int)
    ev["label"] = np.select(
        [
            ev["revised"] == 0,
            (ev["revised"] == 1) & (ev["post_correct"] == 1),
            (ev["revised"] == 1) & (ev["pre_correct"] == 1),
        ],
        [NONE, BENEFICIAL, HARMFUL],
        default=UNHELPFUL,
    )
    return ev.reset_index()


def label_counts(events: pd.DataFrame) -> dict[str, int]:
    counts = events["label"].value_counts().to_dict()
    return {k: int(counts.get(k, 0)) for k in (*LABELS, NONE)}


def revision_rates(events: pd.DataFrame) -> dict[str, float]:
    """Per-agent revision rates, split by whether the agent started correct (r_C and r_I)."""
    total = len(events)
    correct = events[events["pre_correct"] == 1]
    incorrect = events[events["pre_correct"] == 0]
    return {
        "revision_rate": float(events["revised"].mean()) if total else float("nan"),
        "r_C": float(correct["revised"].mean()) if len(correct) else float("nan"),
        "r_I": float(incorrect["revised"].mean()) if len(incorrect) else float("nan"),
        "base_accuracy": float(events["pre_correct"].mean()) if total else float("nan"),
        "s_switch_to_correct": (
            float((incorrect[incorrect["revised"] == 1]["post_correct"]).mean())
            if len(incorrect[incorrect["revised"] == 1])
            else float("nan")
        ),
    }


TIE_BREAKS = ("round0_then_agent0", "round0_then_confidence", "agent0")


def panel_vote(
    answers: list[int],
    confidences: list[float],
    fallback: list[int] | None = None,
    tie_break: str = "round0_then_agent0",
) -> int:
    """Majority vote with the pre-registered tie-break.

    With ten options and a weak model, three-way splits are common -- in tier-0 smoke data they
    are ~40% of questions -- so the tie-break is not a detail: it decides a large share of panel
    outcomes. The default is deliberately NEUTRAL (round-0 plurality, then the lowest agent id),
    because breaking ties by confidence would smuggle a confidence baseline into the panel metric
    and make gated-vs-ungated comparisons partly a comparison of tie-break quality.
    ``round0_then_confidence`` is available as a reported variant (docs/RISKS.md risk 13).
    """
    if tie_break not in TIE_BREAKS:
        raise ValueError(f"unknown tie_break {tie_break!r}, expected one of {TIE_BREAKS}")
    counts = Counter(answers)
    top = max(counts.values())
    tied = sorted(a for a, c in counts.items() if c == top)
    if len(tied) == 1:
        return tied[0]
    if tie_break != "agent0" and fallback is not None:
        fb = panel_vote(fallback, confidences, None, tie_break)
        if fb in tied:
            return fb
    if tie_break == "round0_then_confidence":
        cands = [(confidences[i], -answers[i]) for i in range(len(answers)) if answers[i] in tied]
        if cands:
            return -max(cands)[1]
    for a in answers:  # agent order is the neutral fallback
        if a in tied:
            return a
    return tied[0]


def apply_policy(
    events: pd.DataFrame, accept: np.ndarray | pd.Series, tie_break: str = "round0_then_agent0"
) -> pd.DataFrame:
    """Resolve each agent's final answer under an accept/reject decision, then vote per question.

    Non-revised events are unaffected (pre == post), so ``accept`` only matters where revised == 1.
    """
    ev = events.copy()
    ev["accept"] = np.asarray(accept).astype(bool)
    ev["final_answer"] = np.where(ev["accept"], ev["post_answer"], ev["pre_answer"])
    ev["final_confidence"] = np.where(ev["accept"], ev["post_confidence"], ev["pre_confidence"])
    ev["final_correct"] = (ev["final_answer"] == ev["gold_display"]).astype(int)

    # Hot path: this runs once per threshold in a sweep, so avoid per-group DataFrame overhead
    # and work on numpy arrays with explicit group boundaries.
    ev = ev.sort_values(["run_id", "condition", "question_id", "agent_id"], kind="stable")
    keys = list(zip(ev["run_id"], ev["condition"], ev["question_id"], strict=True))
    bounds = [0]
    for i in range(1, len(keys)):
        if keys[i] != keys[i - 1]:
            bounds.append(i)
    bounds.append(len(keys))

    final_answer = ev["final_answer"].to_numpy()
    final_conf = ev["final_confidence"].to_numpy()
    pre_answer = ev["pre_answer"].to_numpy()
    final_correct = ev["final_correct"].to_numpy()
    gold = ev["gold_display"].to_numpy()
    subject = ev["subject"].to_numpy()
    accept_arr = ev["accept"].to_numpy()
    revised_arr = ev["revised"].to_numpy()

    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        run_id, condition, qid = keys[lo]
        answers = final_answer[lo:hi].tolist()
        vote = panel_vote(
            answers, final_conf[lo:hi].tolist(), pre_answer[lo:hi].tolist(), tie_break
        )
        out.append(
            {
                "run_id": run_id,
                "condition": condition,
                "question_id": qid,
                "subject": subject[lo],
                "panel_answer": vote,
                "gold_display": gold[lo],
                "panel_correct": int(vote == gold[lo]),
                "agent_accuracy": float(final_correct[lo:hi].mean()),
                "n_accepted": int(accept_arr[lo:hi].sum()),
                "n_revised": int(revised_arr[lo:hi].sum()),
                "tied": int(len(set(answers)) == len(answers)),
            }
        )
    return pd.DataFrame(out)
