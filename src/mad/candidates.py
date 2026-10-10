"""Candidate evaluation: does the model's state track whether a FIXED proposed answer is right,
and does an unsupported endorsement move it?

For each question: one correct candidate (the gold option) and one incorrect candidate (a wrong
option drawn uniformly at random from a stable seed), each read out under three contexts --
neutral, an endorsement of the correct option, an endorsement of the incorrect option. Six
readout-only passes per question; nothing is generated. Gold is used only to construct the
pairs and to score them; the prompt never says which candidate is correct.

Saved per readout: the full letter distribution (the output-only score of a candidate is the
probability of its letter), and hidden states at the readout position and the last prompt
token, keyed (condition="candidates", question, item 0-5, phase="<candidate>|<context>").
Note the readout here asks which option is correct after seeing the candidate; it is a
different format from debate, and a probe trained here is a different probe.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mad import prompts as P
from mad.backends.base import OracleEntry, build_backend
from mad.multi import stable_rng
from mad.permute import LETTERS, make_permutation
from mad.provenance import prompt_hashes, sha256_file, write_completion, write_manifest
from mad.store import ActivationWriter

KINDS = ("correct", "incorrect")
CONTEXTS = ("neutral", "endorse_correct", "endorse_incorrect")
ROWS = "candidate_states.parquet"


def items_for(q, perm, seed) -> list[dict]:
    gold = perm.original_to_display(q.gold_index)
    wrong_opts = [i for i in range(q.n_options) if i != gold]
    wrong = int(stable_rng(seed, "candidate_wrong", q.question_id).choice(wrong_opts))
    out = []
    for kind in KINDS:
        for ctx in CONTEXTS:
            endorsed = {"neutral": -1, "endorse_correct": gold, "endorse_incorrect": wrong}[ctx]
            out.append({"candidate_kind": kind, "context": ctx, "gold_display": gold,
                        "candidate": gold if kind == "correct" else wrong, "endorsed": endorsed})
    return out


def messages_for(cfg, q, perm, item) -> list[dict]:
    shown = perm.display_options(list(q.options))
    user = P.QUESTION.format(subject=q.subject, question=q.question, options=P.format_options(list(q.options), perm))
    user += "\n\n" + P.CANDIDATE_BLOCK.format(letter=LETTERS[item["candidate"]], option=shown[item["candidate"]])
    if item["endorsed"] >= 0:
        user += "\n\n" + P.ENDORSEMENT.format(letter=LETTERS[item["endorsed"]], option=shown[item["endorsed"]])
    system = P.CANDIDATE_SYSTEM.format(persona=cfg.debate.personas[0])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def run_candidates(cfg, questions, out_dir: Path, backend=None, extra_manifest=None) -> pd.DataFrame:
    """All six readouts for every question, written atomically to ``out_dir``."""
    out_dir = Path(out_dir)
    perms = {q.question_id: make_permutation(q.n_options, q.question_id, cfg.data.permutation_seed)
             for q in questions}
    if backend is None:
        oracle = ({q.question_id: OracleEntry(perms[q.question_id].original_to_display(q.gold_index), q.n_options)
                   for q in questions} if cfg.model.backend == "stub" else None)
        backend = build_backend(cfg, oracle=oracle)
    tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    cfg.run_id = out_dir.name
    write_manifest(cfg, run_dir=tmp, extra={"prompt_template_hashes": prompt_hashes(P.TEMPLATES),
                                            "kind": "candidates", "question_ids": [q.question_id for q in questions],
                                            "contexts": CONTEXTS, "backend": backend.describe(),
                                            **(extra_manifest or {})})
    recs, texts, metas, nopt = [], [], [], []
    for q in questions:
        for k, item in enumerate(items_for(q, perms[q.question_id], cfg.seed)):
            msgs = messages_for(cfg, q, perms[q.question_id], item)
            m = {"question_id": q.question_id, "agent_id": k, "item": k, "condition": "candidates",
                 "phase": f"{item['candidate_kind']}|{item['context']}", "gen_start_char": None}
            planned = backend.plan_answer(phase="candidate", question_id=q.question_id, item=k,
                                          candidate=item["candidate"], endorsed=item["endorsed"])
            if planned is not None:
                m["planned_answer"] = planned
            recs.append({"question_id": q.question_id, "subject": q.subject, "item": k,
                         "n_options": q.n_options, **item})
            texts.append(backend.render(msgs))
            metas.append(m)
            nopt.append(q.n_options)
    reads = backend.readout(texts, nopt, meta=metas)
    writer = ActivationWriter(tmp, backend.n_layers, backend.d_model, cfg.activations.shard_size)
    for rec, m, r in zip(recs, metas, reads, strict=True):
        key = {"run_id": cfg.run_id, "condition": "candidates", "question_id": m["question_id"],
               "agent_id": m["agent_id"], "phase": m["phase"], "position": "readout_last"}
        writer.add(key, r.hidden)
        if "generation_last" in r.extra:
            writer.add({**key, "position": "generation_last"}, r.extra["generation_last"])
        lp = r.letter_logprobs
        al = r.alignment or {}
        rec.update({"readout_tokens": r.n_tokens, "prompt_tokens": al.get("n_prompt_tokens", -1),
                    "generation_last_token": (al.get("positions", {}).get("generation_last") or [-1, -1])[1]})
        rec.update({"readout_logprobs": ",".join(f"{v:.5f}" for v in lp), "readout_answer": r.answer_index,
                    "p_candidate": float(np.exp(lp[rec["candidate"]])),
                    "p_endorsed": float(np.exp(lp[rec["endorsed"]])) if rec["endorsed"] >= 0 else np.nan,
                    "p_gold": float(np.exp(lp[rec["gold_display"]])),
                    "candidate_is_correct": int(rec["candidate_kind"] == "correct")})
    index = writer.close()
    rows = pd.DataFrame(recs)
    rows.to_parquet(tmp / ROWS, index=False)
    # the exact rendered prompts, for audit (the readout appends READOUT_SUFFIX to each)
    pd.DataFrame({"question_id": [m["question_id"] for m in metas], "item": [m["item"] for m in metas],
                  "phase": [m["phase"] for m in metas], "prompt": texts}).to_json(
        tmp / "prompts.jsonl", orient="records", lines=True)
    hashes = {p.name: sha256_file(p) for p in writer.shard_paths}
    for f in (ROWS, "prompts.jsonl", "activation_index.parquet", "activation_meta.parquet"):
        hashes[f] = sha256_file(tmp / f)
    write_completion(tmp, {"candidate_states": len(rows), "activation_states": len(index)}, hashes)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return rows
