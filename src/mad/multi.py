"""Multi-condition debate collection over a SHARED round 0.

Round 0 is generated once and reused by every condition. Two reasons:

1. **It makes the matched analysis structural.** Every condition starts each agent from a
   byte-identical initial response, so "the same agent, from the same starting answer, revised
   one way with peers and another way alone" is true by construction rather than by hoping two
   separate runs sampled identically.
2. It removes the duplicated round-0 cost, which is roughly a third of the compute for three
   conditions.

Conditions differ only in what the agent is shown before it revises:

  self_reconsider  nothing -- the ordinary-instability floor
  debate           the two real peer responses, confidence line stripped
  debate_conf      the two real peer responses with confidence left in (ablation)
  adversarial      two fabricated peers confidently asserting the same wrong answer
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mad import prompts as P
from mad.backends.base import Backend, OracleEntry, build_backend
from mad.data import Question
from mad.debate import ROWS_FILE, TRANSCRIPTS_FILE, RunResult, _akey, _row, peer_majority
from mad.permute import LETTERS, make_permutation
from mad.provenance import prompt_hashes, sha256_file, write_completion, write_manifest
from mad.store import ActivationWriter


class _NullGen:
    """Placeholder for the peer_exposure state, which has no generated text of its own."""

    text = ""
    n_prompt_tokens = 0
    n_completion_tokens = 0


KINDS = ("self_reconsider", "debate", "debate_conf", "adversarial", "misleading_one")


def adversarial_answer(gold: int, pre: int, n_options: int) -> int:
    """A wrong option for both fabricated peers to assert, differing from the focal answer."""
    for i in range(n_options):
        if i != gold and i != pre:
            return i
    return (gold + 1) % n_options


def run_conditions(
    cfg,
    questions: list[Question],
    conditions: list[dict],
    backend: Backend | None = None,
) -> dict[str, RunResult]:
    n_agents = cfg.debate.n_agents
    perms = {q.question_id: make_permutation(q.n_options, q.question_id, cfg.data.permutation_seed)
             for q in questions}
    gold = {q.question_id: perms[q.question_id].original_to_display(q.gold_index) for q in questions}
    n_opts = {q.question_id: q.n_options for q in questions}
    if backend is None:
        oracle = ({q.question_id: OracleEntry(gold[q.question_id], q.n_options) for q in questions}
                  if cfg.model.backend == "stub" else None)
        backend = build_backend(cfg, oracle=oracle)

    # ---------------------------------------------------------------- round 0, generated once
    base_msgs, prompts, metas = {}, [], []
    for q in questions:
        for a in range(n_agents):
            msgs = P.round0_messages(
                persona=cfg.debate.personas[a % len(cfg.debate.personas)],
                subject=q.subject, question=q.question, options=list(q.options),
                perm=perms[q.question_id], ask_confidence=cfg.debate.ask_verbalized_confidence,
            )
            base_msgs[(q.question_id, a)] = msgs
            prompts.append(backend.render(msgs))
            m0 = {"question_id": q.question_id, "agent_id": a, "phase": "pre", "condition": "shared"}
            planned = backend.plan_answer(phase="pre", question_id=q.question_id, agent_id=a)
            if planned is not None:
                m0["planned_answer"] = planned
            metas.append(m0)
    gens0 = backend.generate(prompts, seed=cfg.seed, meta=metas)
    texts0 = [p + g.text for p, g in zip(prompts, gens0, strict=True)]
    reads0 = backend.readout(texts0, [n_opts[m["question_id"]] for m in metas], meta=metas)
    pre_answer = {(m["question_id"], m["agent_id"]): r.answer_index
                  for m, r in zip(metas, reads0, strict=True)}
    own_text = {(m["question_id"], m["agent_id"]): g.text
                for m, g in zip(metas, gens0, strict=True)}

    # ---------------------------------------------------------------- one round 1 per condition
    results: dict[str, RunResult] = {}
    for cond in conditions:
        name, kind = cond["name"], cond["kind"]
        cfg.run_id = cond["run_id"]
        manifest = write_manifest(cfg, extra={
            "prompt_template_hashes": prompt_hashes(P.TEMPLATES), "n_questions": len(questions),
            "condition": name, "kind": kind, "shared_round0": True,
            "peer_shows_confidence": bool(cond.get("peer_confidence", False)),
            "backend_n_layers": backend.n_layers, "backend_d_model": backend.d_model,
        })
        run_dir = manifest.run_dir
        writer = ActivationWriter(run_dir, backend.n_layers, backend.d_model, cfg.activations.shard_size)

        rows, transcripts = [], []
        for meta, gen, read in zip(metas, gens0, reads0, strict=True):
            m = {**meta, "condition": name}
            rows.append(_row(cfg, m, gen, read, n_opts[m["question_id"]], gold, perms))
            transcripts.append({**m, "text": gen.text})
            writer.add(_akey(cfg, m), read.hidden)

        prompts1, metas1 = [], []
        for q in questions:
            qid = q.question_id
            for a in range(n_agents):
                pre = pre_answer[(qid, a)]
                peers = [pre_answer[(qid, b)] for b in range(n_agents) if b != a]
                if kind == "self_reconsider":
                    peer_texts, maj = None, None
                elif kind == "misleading_one":
                    # One dissenting peer, restrained wording, the other peer genuine. Two
                    # emphatic fabricated peers produced 99% adoption, which measures whether
                    # the model copies a confident assertion rather than anything about
                    # conformity. One hedged dissenter should land in an informative range.
                    adv = adversarial_answer(gold[qid], pre, q.n_options)
                    shown = perms[qid].display_options(list(q.options))[adv]
                    other = [b for b in range(n_agents) if b != a][0]
                    peer_texts = [P.MILD_PEER.format(letter=LETTERS[adv], option=shown),
                                  P.strip_confidence(own_text[(qid, other)])]
                    peers = [adv, pre_answer[(qid, other)]]
                    maj = peer_majority(peers)
                elif kind == "adversarial":
                    adv = adversarial_answer(gold[qid], pre, q.n_options)
                    shown = perms[qid].display_options(list(q.options))[adv]
                    peer_texts = [P.ADVERSARIAL_PEER.format(letter=LETTERS[adv], option=shown)] * (n_agents - 1)
                    peers, maj = [adv] * (n_agents - 1), adv
                else:
                    raw = [own_text[(qid, b)] for b in range(n_agents) if b != a]
                    peer_texts = raw if cond.get("peer_confidence") else [P.strip_confidence(t) for t in raw]
                    maj = peer_majority(peers)
                msgs = P.round1_messages(base=base_msgs[(qid, a)], own_response=own_text[(qid, a)],
                                         peer_responses=peer_texts)
                prompts1.append(backend.render(msgs))
                planned = backend.plan_answer(phase="post", question_id=qid, agent_id=a, pre=pre,
                                              peer_majority=maj, condition=kind)
                metas1.append({"question_id": qid, "agent_id": a, "phase": "post", "condition": name,
                               **({"planned_answer": planned} if planned is not None else {}),
                               "peer_majority": -1 if maj is None else maj,
                               "n_peers_agree_pre": sum(1 for p in peers if p == pre),
                               "peers_agree_each_other": int(len(set(peers)) == 1),
                               "peer_unanimous": int(len(set(peers)) == 1)})

        # Third state: after the agent has seen the intervention but BEFORE it commits to a
        # revision. Everything measured so far is the post-answer state, which exists only once
        # the answer is decided, so it may be reading a decision rather than predicting one.
        metas_pe = [{**m, "phase": "peer_exposure",
                     "pre_answer": pre_answer[(m["question_id"], m["agent_id"])]}
                    for m in metas1]
        reads_pe = backend.readout(prompts1, [n_opts[m["question_id"]] for m in metas1], meta=metas_pe)
        for m, read in zip(metas_pe, reads_pe, strict=True):
            row = _row(cfg, m, _NullGen(), read, n_opts[m["question_id"]], gold, perms)
            row["post_equals_peer_majority"] = int(read.answer_index == m["peer_majority"])
            row["peers_agree_each_other"] = m["peers_agree_each_other"]
            rows.append(row)
            writer.add(_akey(cfg, m), read.hidden)

        gens1 = backend.generate(prompts1, seed=cfg.seed + 1, meta=metas1)
        texts1 = [p + g.text for p, g in zip(prompts1, gens1, strict=True)]
        reads1 = backend.readout(texts1, [n_opts[m["question_id"]] for m in metas1], meta=metas1)
        for m, gen, read in zip(metas1, gens1, reads1, strict=True):
            row = _row(cfg, m, gen, read, n_opts[m["question_id"]], gold, perms)
            row["post_equals_peer_majority"] = int(read.answer_index == m["peer_majority"])
            row["peers_agree_each_other"] = m["peers_agree_each_other"]
            rows.append(row)
            transcripts.append({**m, "text": gen.text})
            writer.add(_akey(cfg, m), read.hidden)

        index = writer.close()
        frame = pd.DataFrame(rows)
        frame["subject"] = frame["question_id"].map({q.question_id: q.subject for q in questions})
        frame.to_parquet(run_dir / ROWS_FILE, index=False)
        pd.DataFrame(transcripts).to_json(run_dir / TRANSCRIPTS_FILE, orient="records", lines=True)
        write_completion(run_dir,
                         row_counts={"agent_states": len(frame), "activation_states": len(index),
                                     "questions": len(questions)},
                         shard_hashes={p.name: sha256_file(p) for p in writer.shard_paths})
        results[name] = RunResult(run_dir=Path(run_dir), rows=frame, n_states=len(index))
    return results
