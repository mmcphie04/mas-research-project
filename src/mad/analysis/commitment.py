"""Commitment analyses shared by the saved-probability preview (scripts/exposure_commit.py) and
the GPU curve readouts (scripts/commit_curves.py).

A RESPONSE is one agent's revision (phase ``post``) in one condition. Its *previous* answer is
the same agent's round-0 answer in the same view. "Committed" at a readout means the argmax over
the option letters equals the answer the agent eventually stated. Uncertainty: percentile
intervals from bootstrap resamples of whole questions (all responses of a resampled question,
across conditions, kept together), as in the paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mad.curves import eligibility, outcome_kind

KINDS = ("unchanged_correct", "unchanged_wrong", "helpful", "harmful", "wrong_to_wrong")


def _lp(v) -> np.ndarray | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, str):
        return np.array([float(x) for x in v.split(",")], dtype=float) if v else None
    return np.asarray(v, dtype=float)


def load_view(view: Path, pool: str, condition: str | None = None) -> pd.DataFrame:
    """One row per revision in a view, with eligibility, outcome and the saved letter
    distributions at peer exposure and at the token before the stated letter."""
    view = Path(view)
    st = pd.read_parquet(view / "agent_states.parquet")
    st["question_id"] = st.question_id.astype(str)
    cond = condition or str(st.condition.iloc[0])
    key = ["question_id", "agent_id"]
    pre = st[st.phase == "pre"].set_index(key)
    pe = st[st.phase == "peer_exposure"].set_index(key)
    post = st[st.phase == "post"].copy()
    rows = []
    for r in post.itertuples(index=False):
        k = (r.question_id, r.agent_id)
        p0 = pre.loc[k]
        reason = eligibility(r.answer_source, p0.answer_source)
        rows.append({
            "pool": pool, "condition": cond, "question_id": r.question_id, "agent_id": int(r.agent_id),
            "subject": getattr(r, "subject", None), "n_options": int(r.n_options),
            "gold": int(r.gold_display), "answer": int(r.answer), "answer_source": r.answer_source,
            "stop_reason": r.stop_reason, "prev_answer": int(p0.answer), "prev_source": p0.answer_source,
            "prev_correct": int(p0.correct), "correct": int(r.correct), "eligibility": reason,
            "kind": outcome_kind(int(p0.answer), int(r.answer), int(r.gold_display)),
            "n_shown_agree_pre": getattr(r, "n_shown_agree_pre", np.nan),
            "shown_peer_answers": getattr(r, "shown_peer_answers", None),
            "lp_exposure": _lp(pe.loc[k].readout_logprobs) if k in pe.index else None,
            "lp_before_stated": _lp(r.pos_before_stated_answer_logprobs),
            "n_completion_tokens": r.n_completion_tokens,
        })
    df = pd.DataFrame(rows)
    df["changed"] = (df.answer != df.prev_answer).astype(int)
    df["cluster"] = df.pool + ":" + df.question_id
    return df


def exclusions(df: pd.DataFrame) -> dict:
    """Counts by pool / condition / reason (eligible included), plus capped-but-eligible."""
    out = {}
    for (pool, cond), g in df.groupby(["pool", "condition"]):
        out[f"{pool}/{cond}"] = {
            "revisions": int(len(g)), **{k: int(v) for k, v in g.eligibility.value_counts().items()},
            "capped_but_stated_kept": int(((g.stop_reason == "length") & (g.eligibility == "eligible")).sum()),
        }
    return out


def add_readout(df: pd.DataFrame, col: str, tag: str) -> pd.DataFrame:
    """Argmax-agreement indicators and probabilities for one letter-distribution column."""
    df = df.copy()
    am = [int(np.argmax(v)) if v is not None else -1 for v in df[col]]
    df[f"{tag}_argmax"] = am
    for name, target in (("eventual", "answer"), ("previous", "prev_answer"), ("gold", "gold")):
        df[f"{tag}_is_{name}"] = [float(a == t) if a >= 0 else np.nan for a, t in zip(am, df[target], strict=True)]
        df[f"{tag}_p_{name}"] = [float(np.exp(v[t])) if v is not None else np.nan
                                 for v, t in zip(df[col], df[target], strict=True)]
    return df


def cluster_boot(df: pd.DataFrame, value: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of ``value`` with a question-cluster bootstrap 95% interval."""
    d = df[[value, "cluster"]].dropna()
    if d.empty:
        return {"n": 0, "questions": 0, "est": None, "lo": None, "hi": None}
    g = d.groupby("cluster")[value].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n_boot, len(g)))
    w = np.zeros((n_boot, len(g)))
    np.add.at(w, (np.repeat(np.arange(n_boot), len(g)), idx.ravel()), 1)
    boots = (w @ s) / np.maximum(w @ c, 1)
    return {"n": int(len(d)), "questions": int(len(g)), "est": float(s.sum() / c.sum()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}


def subsets(df: pd.DataFrame):
    """(label, frame) for every reporting subset."""
    yield "all", df
    yield "unchanged", df[df.changed == 0]
    yield "changed", df[df.changed == 1]
    for k in ("helpful", "harmful", "wrong_to_wrong"):
        yield k, df[df.kind == k]


def paired_gap(df: pd.DataFrame, a: str, b: str, value: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """Per-question mean of ``value`` among CHANGED responses in condition a minus condition b,
    over questions with a change in both (paper Table 9: mean, share favouring a, sign test)."""
    from scipy.stats import binomtest

    ch = df[df.changed == 1]
    ma = ch[ch.condition == a].groupby("cluster")[value].mean()
    mb = ch[ch.condition == b].groupby("cluster")[value].mean()
    common = ma.index.intersection(mb.index)
    if len(common) == 0:
        return {"questions": 0}
    d = (ma[common] - mb[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return {"questions": int(len(d)), "mean_diff": float(d.mean()),
            "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975)),
            "favour_a": pos, "favour_b": neg, "ties": int(len(d) - pos - neg),
            "sign_test_p": float(binomtest(pos, pos + neg).pvalue) if pos + neg else None}


def to_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x)))
