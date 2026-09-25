"""Readout-only state collection: activations and forced-choice answers, no generation.

This is the cheap path. A correctness probe needs (activation, stated answer, correctness)
triples; it does not need debate, peers, or even chain-of-thought. Skipping generation removes
essentially all the compute, which is what makes a real-activation probe feasible on a CPU.

Several PERMUTATION VARIANTS of each question are collected. Each variant is a separate state
with its own activation and its own correctness -- the model may answer a question correctly
under one option ordering and not another. That yields two things a single state per question
cannot: within-question variation, so CONDITIONAL AUROC is defined and the difficulty-detector
diagnostic works, and a direct measurement of answer-position bias.

Note what this does and does not measure. With no chain-of-thought, the probe is being asked
whether the model's immediate forced choice is correct -- closer to Kadavath et al.'s P(IK) than
to the full debate setting. It validates the machinery and the incremental value over logit
confidence; it is not the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mad import prompts as P
from mad.backends.base import Backend, OracleEntry, build_backend
from mad.data import Question
from mad.permute import make_permutation
from mad.provenance import prompt_hashes, sha256_file, write_completion, write_manifest
from mad.store import ActivationWriter

STATES_FILE = "states.parquet"


@dataclass
class CollectResult:
    run_dir: Path
    rows: pd.DataFrame
    n_states: int


def collect_states(
    cfg,
    questions: list[Question],
    backend: Backend | None = None,
    n_variants: int = 3,
) -> CollectResult:
    """One readout state per (question, permutation variant)."""
    perms = {
        (q.question_id, v): make_permutation(
            q.n_options, f"{q.question_id}|v{v}", cfg.data.permutation_seed
        )
        for q in questions
        for v in range(n_variants)
    }
    if backend is None:
        # tier 0 needs the oracle table at construction; real backends must never receive it
        oracle = (
            {
                q.question_id: OracleEntry(
                    perms[(q.question_id, 0)].original_to_display(q.gold_index), q.n_options
                )
                for q in questions
            }
            if cfg.model.backend == "stub"
            else None
        )
        backend = build_backend(cfg, oracle=oracle)

    manifest = write_manifest(
        cfg,
        extra={
            "mode": "readout_only",
            "prompt_template_hashes": prompt_hashes(P.TEMPLATES),
            "n_questions": len(questions),
            "n_variants": n_variants,
            "backend_n_layers": backend.n_layers,
            "backend_d_model": backend.d_model,
        },
    )
    run_dir = manifest.run_dir
    writer = ActivationWriter(run_dir, backend.n_layers, backend.d_model, cfg.activations.shard_size)

    texts, metas, n_opts = [], [], []
    for q in questions:
        for v in range(n_variants):
            perm = perms[(q.question_id, v)]
            msgs = P.round0_messages(
                persona=cfg.debate.personas[0],
                subject=q.subject,
                question=q.question,
                options=list(q.options),
                perm=perm,
                ask_confidence=False,
            )
            # the backend appends READOUT_SUFFIX itself; appending here too produced
            # 'Final answer: (Final answer: (' and invalidated the first ARC runs
            texts.append(backend.render(msgs))
            n_opts.append(q.n_options)
            planned = backend.plan_answer(phase="pre", question_id=q.question_id, agent_id=v)
            metas.append(
                {
                    "question_id": q.question_id,
                    "agent_id": v,
                    "phase": "pre",
                    "condition": "direct",
                    **({"planned_answer": planned} if planned is not None else {}),
                    "variant": v,
                    "subject": q.subject,
                    "n_options": q.n_options,
                    "gold_display": perm.original_to_display(q.gold_index),
                    "permutation": ",".join(str(i) for i in perm.as_list()),
                }
            )

    reads = backend.readout(texts, n_opts, meta=metas)

    rows = []
    for meta, read in zip(metas, reads, strict=True):
        row_meta = {k: v for k, v in meta.items() if k != "planned_answer"}
        rows.append(
            {
                "run_id": cfg.run_id,
                **row_meta,
                "readout_answer": read.answer_index,
                "readout_confidence": read.confidence,
                "readout_logprob_top": float(read.letter_logprobs.max()),
                "correct": int(read.answer_index == meta["gold_display"]),
                "n_readout_tokens": read.n_tokens,
            }
        )
        writer.add(
            {
                "run_id": cfg.run_id,
                "condition": "direct",
                "question_id": meta["question_id"],
                "agent_id": meta["variant"],
                "phase": "pre",
                "position": "readout_last",
            },
            read.hidden,
        )

    index = writer.close()
    frame = pd.DataFrame(rows)
    frame.to_parquet(run_dir / STATES_FILE, index=False)
    write_completion(
        run_dir,
        row_counts={"states": len(frame), "questions": len(questions), "activations": len(index)},
        shard_hashes={p.name: sha256_file(p) for p in writer.shard_paths},
    )
    return CollectResult(run_dir=run_dir, rows=frame, n_states=len(index))
