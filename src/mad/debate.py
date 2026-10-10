"""One-round multi-agent debate harness.

Pipeline per question:
    round 0  generate (3 agents, independent)
        ->  readout pass   -> canonical pre answers + pre activations
    round 1  generate (each agent sees the other two, or a neutral reconsider prompt)
        ->  readout pass   -> canonical post answers + post activations

Generation and activation extraction are deliberately separate passes, so activations can be
re-extracted at new layers or positions without regenerating a token, and so the gating analysis
downstream needs no GPU at all (docs/PLAN.md section 2).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mad import prompts as P
from mad.backends.base import Backend, OracleEntry, build_backend
from mad.data import Question
from mad.parsing import parse_answer_field, parse_confidence
from mad.permute import Permutation, make_permutation
from mad.provenance import prompt_hashes, sha256_file, write_completion, write_manifest
from mad.store import ActivationWriter

ROWS_FILE = "agent_states.parquet"
TRANSCRIPTS_FILE = "transcripts.jsonl"


def peer_majority(answers: list[int]) -> int | None:
    """Modal peer answer, or None when peers are split."""
    if not answers:
        return None
    counts = Counter(answers).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


@dataclass
class RunResult:
    run_dir: Path
    rows: pd.DataFrame
    n_states: int
    timings: dict | None = None


def run_debate(cfg, questions: list[Question], backend: Backend | None = None) -> RunResult:
    condition = cfg.debate.condition
    n_agents = cfg.debate.n_agents
    perms: dict[str, Permutation] = {
        q.question_id: make_permutation(q.n_options, q.question_id, cfg.data.permutation_seed)
        for q in questions
    }
    gold_display = {
        q.question_id: perms[q.question_id].original_to_display(q.gold_index) for q in questions
    }

    if backend is None:
        oracle = (
            {
                q.question_id: OracleEntry(gold_display[q.question_id], q.n_options)
                for q in questions
            }
            if cfg.model.backend == "stub"
            else None
        )
        backend = build_backend(cfg, oracle=oracle)

    manifest = write_manifest(
        cfg,
        extra={
            "prompt_template_hashes": prompt_hashes(P.TEMPLATES),
            "n_questions": len(questions),
            "question_id_list_hash": prompt_hashes(
                {"ids": ",".join(sorted(q.question_id for q in questions))}
            )["ids"],
            "condition": condition,
            "backend_n_layers": backend.n_layers,
            "backend_d_model": backend.d_model,
        },
    )
    run_dir = manifest.run_dir
    writer = ActivationWriter(
        run_dir, backend.n_layers, backend.d_model, cfg.activations.shard_size
    )

    # ---------------------------------------------------------------- round 0: independent answers
    base_messages: dict[tuple[str, int], list[dict]] = {}
    prompts, metas = [], []
    for q in questions:
        for a in range(n_agents):
            msgs = P.round0_messages(
                persona=cfg.debate.personas[a % len(cfg.debate.personas)],
                subject=q.subject,
                question=q.question,
                options=list(q.options),
                perm=perms[q.question_id],
                ask_confidence=cfg.debate.ask_verbalized_confidence,
            )
            base_messages[(q.question_id, a)] = msgs
            meta = {
                "question_id": q.question_id,
                "agent_id": a,
                "phase": "pre",
                "condition": condition,
            }
            planned = backend.plan_answer(phase="pre", question_id=q.question_id, agent_id=a)
            if planned is not None:
                meta["planned_answer"] = planned
            prompts.append(backend.render(msgs))
            metas.append(meta)

    gens_pre = backend.generate(prompts, seed=cfg.seed, meta=metas)
    texts_pre = [p + g.text for p, g in zip(prompts, gens_pre, strict=True)]
    n_opts_by_qid = {q.question_id: q.n_options for q in questions}
    n_opts = [n_opts_by_qid[m["question_id"]] for m in metas]
    reads_pre = backend.readout(texts_pre, n_opts, meta=metas)

    pre_answer: dict[tuple[str, int], int] = {}
    rows: list[dict] = []
    transcripts: list[dict] = []
    for meta, gen, read in zip(metas, gens_pre, reads_pre, strict=True):
        key = (meta["question_id"], meta["agent_id"])
        rows.append(
            _row(cfg, meta, gen, read, n_opts_by_qid[meta["question_id"]], gold_display, perms)
        )
        pre_answer[key] = rows[-1]["answer"]  # canonical, so peers and labels agree
        transcripts.append({**meta, "text": gen.text})
        writer.add({**_akey(cfg, meta)}, read.hidden)

    # ---------------------------------------------------------------- round 1: revision
    prompts, metas1 = [], []
    for q in questions:
        for a in range(n_agents):
            key = (q.question_id, a)
            peers = [pre_answer[(q.question_id, b)] for b in range(n_agents) if b != a]
            maj = peer_majority(peers) if condition == "debate" else None
            peer_texts = (
                [
                    t["text"]
                    for t in transcripts
                    if t["question_id"] == q.question_id and t["agent_id"] != a
                ]
                if condition == "debate"
                else None
            )
            own = next(
                t["text"]
                for t in transcripts
                if t["question_id"] == q.question_id and t["agent_id"] == a
            )
            msgs = P.round1_messages(
                base=base_messages[key], own_response=own, peer_responses=peer_texts
            )
            meta = {
                "question_id": q.question_id,
                "agent_id": a,
                "phase": "post",
                "condition": condition,
                "peer_majority": -1 if maj is None else maj,
                "n_peers_agree_pre": sum(1 for p in peers if p == pre_answer[key]),
                "peer_unanimous": int(len(set(peers)) == 1),
            }
            planned = backend.plan_answer(
                phase="post",
                question_id=q.question_id,
                agent_id=a,
                pre=pre_answer[key],
                peer_majority=maj,
                condition=condition,
            )
            if planned is not None:
                meta["planned_answer"] = planned
            prompts.append(backend.render(msgs))
            metas1.append(meta)

    gens_post = backend.generate(prompts, seed=cfg.seed + 1, meta=metas1)
    texts_post = [p + g.text for p, g in zip(prompts, gens_post, strict=True)]
    n_opts1 = [n_opts_by_qid[m["question_id"]] for m in metas1]
    reads_post = backend.readout(texts_post, n_opts1, meta=metas1)

    for meta, gen, read in zip(metas1, gens_post, reads_post, strict=True):
        row = _row(cfg, meta, gen, read, n_opts_by_qid[meta["question_id"]], gold_display, perms)
        row["post_equals_peer_majority"] = int(row["answer"] == meta["peer_majority"])
        rows.append(row)
        transcripts.append({**meta, "text": gen.text})
        writer.add({**_akey(cfg, meta)}, read.hidden)

    # ------------------------------------------------------------------------------- persist
    index = writer.close()
    frame = pd.DataFrame(rows)
    subj = {q.question_id: q.subject for q in questions}
    frame["subject"] = frame["question_id"].map(subj)
    frame.to_parquet(run_dir / ROWS_FILE, index=False)
    pd.DataFrame(transcripts).to_json(run_dir / TRANSCRIPTS_FILE, orient="records", lines=True)

    write_completion(
        run_dir,
        row_counts={
            "agent_states": len(frame),
            "activation_states": len(index),
            "questions": len(questions),
        },
        shard_hashes={p.name: sha256_file(p) for p in writer.shard_paths},
    )
    return RunResult(run_dir=run_dir, rows=frame, n_states=len(index))


def _akey(cfg, meta: dict) -> dict:
    return {
        "run_id": cfg.run_id,
        "condition": meta["condition"],
        "question_id": meta["question_id"],
        "agent_id": meta["agent_id"],
        "phase": meta["phase"],
        "position": "readout_last",
    }


def canonical_answer(cfg, stated: int, stated_ok: bool, readout: int, phase: str) -> tuple[int, str]:
    """The answer that counts, and where it came from (see DebateConfig.canonical_answer).

    The peer_exposure state has no generated text, so its only answer is the readout's.
    """
    mode = getattr(cfg.debate, "canonical_answer", "readout")
    if mode == "readout" or phase == "peer_exposure":
        return readout, "readout"
    if stated_ok:
        return stated, "stated"
    return readout, "readout_fallback"


# Metadata copied verbatim from the request into its row when present. "shown" is what the
# agent was shown; "panel" is what its real panel-mates answered in round 0. They differ in
# self_reconsider (nothing shown), and in misleading_one and adversarial (fabricated peers).
ROW_META_KEYS = (
    "request_id", "chunk", "gen_seed", "shown_peer_answers", "shown_peer_sources",
    "shown_peer_majority", "n_shown_agree_pre", "n_shown_missing", "panel_peer_answers",
    "panel_peer_majority",
    "n_panel_agree_pre", "misleading_target",
)


def _row(cfg, meta, gen, read, n_options, gold_display, perms) -> dict:
    qid = meta["question_id"]
    parsed = parse_answer_field(gen.text, n_options)  # the agent's own response only
    ft_idx, ft_ok = parsed.index, parsed.ok
    gold = gold_display[qid]
    answer, source = canonical_answer(cfg, ft_idx, ft_ok, read.answer_index, meta["phase"])
    row = {
        "run_id": cfg.run_id,
        "condition": meta["condition"],
        "question_id": qid,
        "agent_id": meta["agent_id"],
        "phase": meta["phase"],
        "n_options": n_options,
        "gold_display": gold,
        "permutation": ",".join(str(i) for i in perms[qid].as_list()),
        # the canonical answer and its correctness: what every downstream label uses
        "answer": answer,
        "answer_source": source,
        "correct": int(answer == gold),
        "readout_answer": read.answer_index,
        "readout_correct": int(read.answer_index == gold),
        "readout_confidence": read.confidence,
        "readout_logprob_top": float(read.letter_logprobs.max()),
        # full distribution, so entropy / top-two margin / the whole vector are available to
        # output-only baselines later without another forward pass
        "readout_logprobs": ",".join(f"{v:.5f}" for v in read.letter_logprobs),
        "readout_margin": float(
            np.sort(read.letter_logprobs)[-1] - np.sort(read.letter_logprobs)[-2]
        )
        if len(read.letter_logprobs) > 1
        else 0.0,
        "readout_entropy": float(
            -(np.exp(read.letter_logprobs) * read.letter_logprobs).sum()
        ),
        "freetext_answer": ft_idx,  # the stated answer, -1 when there is no valid one
        "freetext_parse_ok": int(ft_ok),
        "stated_status": parsed.status if meta["phase"] != "peer_exposure" else "no_text",
        "stated_n_fields": parsed.n_fields,
        "freetext_agrees_readout": int(ft_ok and ft_idx == read.answer_index),
        "verbalized_confidence": parse_confidence(gen.text),
        "n_prompt_tokens": gen.n_prompt_tokens,
        "n_completion_tokens": gen.n_completion_tokens,
        "stop_reason": getattr(gen, "stop_reason", None),
        "n_readout_tokens": read.n_tokens,
        "rationale_chars": len(gen.text),
        "peer_majority": meta.get("peer_majority", -1),
        "n_peers_agree_pre": meta.get("n_peers_agree_pre", -1),
        "peer_unanimous": meta.get("peer_unanimous", -1),
        "post_equals_peer_majority": -1,
    }
    for k in ROW_META_KEYS:
        if k in meta:
            row[k] = meta[k]
    # where each extra position is, why it is missing, and the option probabilities there
    al = getattr(read, "alignment", None) or {}
    row["replay_exact"] = al.get("replay_exact")
    # token counts of the readout input: prompt part, and the agent's own response part
    row["readout_prompt_tokens"] = al.get("n_prompt_tokens", -1)
    row["readout_response_tokens"] = al.get("n_response_tokens", -1)
    for k in ("response_logprob_mean", "response_logprob_min", "response_logprob_sum"):
        row[k] = al.get(k, np.nan)
    for p in cfg.activations.positions:
        if p == "readout_last":
            continue
        ti = al.get("positions", {}).get(p)
        lp = (getattr(read, "extra_logprobs", None) or {}).get(p)
        row[f"pos_{p}_token_index"] = ti[0] if ti else -1
        row[f"pos_{p}_token_id"] = ti[1] if ti else -1
        row[f"pos_{p}_missing"] = al.get("missing", {}).get(p, "" if ti else "not_returned")
        row[f"pos_{p}_logprobs"] = ",".join(f"{v:.5f}" for v in lp) if lp is not None else ""
    return row
