"""Events and output-only features, by information time.

One schema adapter for both run formats:
  * new runs store each state's full letter distribution in ``agent_states.parquet``
    (``readout_logprobs``, comma-separated log-probabilities);
  * old 1B runs stored it in a ``readout_full.parquet`` sidecar (``lp`` column).
Widths must equal each question's option count and each distribution must sum to one. Primary
analyses REQUIRE full distributions; there is no silent fallback to a weaker baseline.

Information times. A predictor may only use what exists at the moment its decision is made:

  pre       after round 0: the agent's first answer and its readout distribution, its stated
            confidence, response length/format, and the other agents' first answers.
  exposure  after the agent has read its round-1 message, before it writes: pre + the readout
            distribution at that moment + agreement of its first answer with what it was shown.
  post      after it has written its revision: exposure (when stored) + the post distribution,
            the new answer's identity and probabilities, format of the new response, agreement
            of the new answer with what it was shown and with the panel.

Never used as features: gold labels, correctness, and construction metadata that reveals labels
(which shown peer was fabricated, the misleading target). Option positions enter categorically.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.labels import build_events
from mad.debate import ROWS_FILE

TIMES = ("pre", "exposure", "post", "post_all")  # post_all: post, with all three stages' states
PAD = -30.0


class SchemaError(ValueError):
    pass


def _parse_lp(s) -> np.ndarray:
    return np.array([float(x) for x in str(s).split(",")]) if isinstance(s, str) and s else np.array([])


def load_rows(run_dir: str | Path, require_full: bool = True) -> pd.DataFrame:
    """Agent-state rows with a validated ``lp`` column (np.ndarray per row), either format."""
    run_dir = Path(run_dir)
    rows = pd.read_parquet(run_dir / ROWS_FILE)
    if "readout_logprobs" in rows.columns and rows.readout_logprobs.astype(str).str.len().gt(0).all():
        rows["lp"] = rows.readout_logprobs.map(_parse_lp)
        rows.attrs["lp_source"] = "agent_states.readout_logprobs"
    elif (run_dir / "readout_full.parquet").exists():
        side = pd.read_parquet(run_dir / "readout_full.parquet")[["question_id", "agent_id", "phase", "lp"]]
        rows = rows.merge(side, on=["question_id", "agent_id", "phase"], how="left", validate="1:1")
        if rows.lp.isna().any():
            raise SchemaError(f"{int(rows.lp.isna().sum())} rows lack a distribution in readout_full.parquet")
        rows["lp"] = rows.lp.map(_parse_lp)
        rows.attrs["lp_source"] = "readout_full.parquet"
    elif require_full:
        raise SchemaError(f"{run_dir} has no full answer distributions; primary analyses need them")
    else:
        rows["lp"] = [np.array([]) for _ in range(len(rows))]
        rows.attrs["lp_source"] = "none"
    if rows.attrs["lp_source"] != "none":
        widths = rows.lp.map(len)
        if (widths != rows.n_options).any():
            raise SchemaError("distribution widths differ from the option counts")
        mass = rows.lp.map(lambda v: float(np.exp(v).sum()))
        if (mass.sub(1).abs() > 1e-3).any():
            raise SchemaError("a stored answer distribution does not sum to one")
    return rows


def load_events(run_dir: str | Path, require_full: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, rows). One event per (condition, question, agent), with each phase's
    distribution attached (``lp_pre``, ``lp_post``, and ``lp_exposure`` when stored)."""
    rows = load_rows(run_dir, require_full)
    ev = build_events(rows[rows.phase.isin(["pre", "post"])]).reset_index(drop=True)
    k = ["condition", "question_id", "agent_id"]
    for phase, col in (("pre", "lp_pre"), ("post", "lp_post"), ("peer_exposure", "lp_exposure")):
        sub = rows[rows.phase == phase].set_index(k)
        if len(sub):
            ev[col] = sub.reindex(pd.MultiIndex.from_frame(ev[k]))["lp"].to_numpy()
    for phase in ("pre", "post"):
        sub = rows[rows.phase == phase].set_index(k).reindex(pd.MultiIndex.from_frame(ev[k]))
        for c in ("answer_source", "stop_reason", "freetext_parse_ok", "freetext_agrees_readout",
                  "stated_status", "shown_peer_answers", "response_logprob_mean", "response_logprob_min"):
            ev[f"{phase}_{c}"] = sub[c].to_numpy() if c in sub.columns else np.nan
    # option distributions at the extra token positions, per phase (empty array where absent)
    positions = sorted(c[len("pos_"):-len("_logprobs")] for c in rows.columns
                       if c.startswith("pos_") and c.endswith("_logprobs"))
    for phase, tag in (("pre", "pre"), ("peer_exposure", "exposure"), ("post", "post")):
        sub = rows[rows.phase == phase].set_index(k)
        if not len(sub):
            continue
        sub = sub.reindex(pd.MultiIndex.from_frame(ev[k]))
        for p in positions:
            ev[f"lp_{tag}@{p}"] = sub[f"pos_{p}_logprobs"].map(_parse_lp).to_numpy()
    ev.attrs["lp_source"] = rows.attrs["lp_source"]
    ev.attrs["has_exposure"] = "lp_exposure" in ev.columns
    ev.attrs["positions_with_probabilities"] = positions
    return ev, rows


def _dist_block(cols, tag, lp_list, old, new, width):
    lp = np.full((len(lp_list), width), PAD)
    for i, v in enumerate(lp_list):
        lp[i, : len(v)] = v
    p = np.exp(lp)
    srt = np.sort(lp, axis=1)
    idx = np.arange(len(lp))
    cols[f"p_old@{tag}"] = p[idx, old]
    cols[f"p_new@{tag}"] = p[idx, new] if new is not None else None
    if new is None:
        del cols[f"p_new@{tag}"]
    cols[f"margin@{tag}"] = srt[:, -1] - srt[:, -2]
    cols[f"entropy@{tag}"] = -(p * np.clip(lp, PAD, 0)).sum(axis=1)
    for j in range(width):
        cols[f"lp{j}@{tag}"] = lp[:, j]


def _likelihood(cols, ev, phase):
    """Mean and minimum token log-probability of the response, when the run recorded them."""
    for k in ("mean", "min"):
        v = pd.to_numeric(ev[f"{phase}_response_logprob_{k}"], errors="coerce").to_numpy(float)
        if np.isfinite(v).any():
            cols[f"response_logprob_{k}_{phase}"] = np.nan_to_num(v, nan=0.0)
            cols[f"response_logprob_missing_{phase}"] = np.isnan(v).astype(float)


def _shown(s) -> list[int]:
    return [int(x) for x in str(s).split(",") if x not in ("", "nan")] if isinstance(s, str) else []


def _position_block(cols, ev, tag, p, old, new, width):
    """Option distribution at token position ``p`` in phase ``tag``, with explicit missingness:
    where the state does not exist (no valid stated answer, a flagged letter token, or a
    prompt-only readout) every value is 0 and ``missing`` is 1."""
    lps = ev[f"lp_{tag}@{p}"]
    present = np.array([isinstance(v, np.ndarray) and len(v) > 0 for v in lps])
    filled = [v if ok else np.full(width, PAD) for v, ok in zip(lps, present, strict=True)]
    sub: dict[str, np.ndarray] = {}
    _dist_block(sub, f"{tag}@{p}", filled, old, new, width)
    for k, v in sub.items():
        cols[f"pos:{k}"] = np.where(present, v, 0.0)
    cols[f"pos:missing@{tag}@{p}"] = (~present).astype(float)


def output_features(ev: pd.DataFrame, time: str, positions: tuple[str, ...] | list[str] = ()
                    ) -> tuple[np.ndarray, list[str]]:
    """Output-only features available at ``time`` (see module docstring). Returns (X, names).

    ``positions``: extra token positions whose saved option distributions are added -- the
    position-matched output control for an activation probe at that position. Availability:
    a phase's distribution is used only if that phase exists by ``time``; stated-answer positions
    never exist in the exposure state (a prompt-only readout)."""
    if time not in TIMES:
        raise ValueError(f"time must be one of {TIMES}")
    if time == "post_all":
        time = "post"  # same output information; only the activation phases differ
    if ev.attrs.get("lp_source", "none") == "none" or "lp_pre" not in ev.columns:
        raise SchemaError("output features need full answer distributions")
    if time == "exposure" and not ev.attrs.get("has_exposure"):
        raise SchemaError("this run has no exposure-state distributions")
    width = int(ev.n_options.max())
    old = ev.pre_answer.to_numpy().astype(int)
    new = ev.post_answer.to_numpy().astype(int)
    cols: dict[str, np.ndarray] = {}

    # ---- pre
    _dist_block(cols, "pre", ev.lp_pre, old, new if time == "post" else None, width)
    vc = ev.pre_verbalized.to_numpy(float)
    cols["verbal_conf_pre"] = np.nan_to_num(vc, nan=0.0) / 100
    cols["verbal_conf_pre_missing"] = np.isnan(vc).astype(float)
    cols["len_pre"] = ev.pre_rationale_chars.to_numpy(float)
    _likelihood(cols, ev, "pre")
    cols["fallback_pre"] = (ev.pre_answer_source == "readout_fallback").to_numpy(float)
    cols["hit_cap_pre"] = (ev.pre_stop_reason == "length").to_numpy(float)
    cols["stated_vs_readout_disagree_pre"] = ((ev.pre_freetext_parse_ok == 1)
                                              & (ev.pre_freetext_agrees_readout == 0)).to_numpy(float)
    pre_tab = ev.pivot_table(index=["condition", "question_id"], columns="agent_id",
                             values="pre_answer", aggfunc="first")
    others = [[pre_tab.loc[(c, q), b] for b in pre_tab.columns if b != a]
              for c, q, a in zip(ev.condition, ev.question_id, ev.agent_id, strict=True)]
    cols["panel_agree_old"] = np.array([sum(x == o for x in ot) for ot, o in zip(others, old, strict=True)], float)
    for j in range(width):
        cols[f"old_is_{j}"] = (old == j).astype(float)

    # ---- exposure (what the agent was shown, and its readout after reading it)
    if time in ("exposure", "post"):
        shown = [_shown(s) for s in ev.post_shown_peer_answers]
        cols["n_shown"] = np.array([len(s) for s in shown], float)
        cols["shown_agree_old"] = np.array([sum(x == o for x in s) for s, o in zip(shown, old, strict=True)], float)
        if ev.attrs.get("has_exposure"):
            _dist_block(cols, "exposure", ev.lp_exposure, old, new if time == "post" else None, width)

    # ---- post
    if time == "post":
        _dist_block(cols, "post", ev.lp_post, old, new, width)
        vc = ev.post_verbalized.to_numpy(float)
        cols["verbal_conf_post"] = np.nan_to_num(vc, nan=0.0) / 100
        cols["verbal_conf_post_missing"] = np.isnan(vc).astype(float)
        cols["len_post"] = ev.post_rationale_chars.to_numpy(float)
        _likelihood(cols, ev, "post")
        cols["fallback_post"] = (ev.post_answer_source == "readout_fallback").to_numpy(float)
        cols["hit_cap_post"] = (ev.post_stop_reason == "length").to_numpy(float)
        cols["stated_vs_readout_disagree_post"] = ((ev.post_freetext_parse_ok == 1)
                                                   & (ev.post_freetext_agrees_readout == 0)).to_numpy(float)
        cols["revised"] = ev.revised.to_numpy(float)
        cols["panel_agree_new"] = np.array([sum(x == n for x in ot) for ot, n in zip(others, new, strict=True)], float)
        cols["shown_agree_new"] = np.array([sum(x == n for x in s) for s, n in zip(shown, new, strict=True)], float)
        for j in range(width):
            cols[f"new_is_{j}"] = (new == j).astype(float)
    for p in positions:
        if p == "readout_last":
            continue  # already the main readout blocks
        if p not in ev.attrs.get("positions_with_probabilities", []):
            raise SchemaError(f"this run saved no option probabilities at position {p!r}")
        phases = ["pre"]
        if time in ("exposure", "post") and p == "generation_last" and f"lp_exposure@{p}" in ev.columns:
            phases.append("exposure")
        if time == "post":
            phases.append("post")
        for tag in phases:  # P(new answer) only once the new answer exists
            _position_block(cols, ev, tag, p, old, new if time == "post" else None, width)
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(float), names


# Phases whose ACTIVATIONS a predictor at each time may use.
ACTIVATION_PHASES = {"pre": ("pre",), "exposure": ("pre", "peer_exposure"), "post": ("pre", "post"),
                     "post_all": ("pre", "peer_exposure", "post")}
