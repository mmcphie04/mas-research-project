"""Recover full answer-choice log-probability vectors without regenerating any text.

The debate runs stored only the top probability, so output-only baselines could not use the
top-two margin, the entropy, or the whole vector. The generations themselves are saved, and the
readout is a prefill-only forward pass, so the vectors can be recovered by replaying just that
pass over prompt+generation. No sampling, no new text.

    uv run python scripts/recover_logprobs.py night2-debate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mad import prompts as P
from mad.backends.base import build_backend
from mad.config import load_config
from mad.data import load_mc_dataset, stratified_sample
from mad.permute import make_permutation

BASE = ["configs/base.yaml", "configs/probe_arc.yaml"]


def main(run: str, n_questions: int = 350) -> None:
    run_dir = Path("runs") / run
    chunks = sorted(Path("runs").glob(f"{run}-c*"))
    tr = pd.concat([pd.read_json(c / "transcripts.jsonl", lines=True) for c in chunks],
                   ignore_index=True)
    cfg = load_config(*BASE, **{"model.batch_size": 16})
    qs = {q.question_id: q for q in stratified_sample(
        load_mc_dataset("allenai/ai2_arc", "ARC-Challenge", "test"), n_questions, 7)}
    perms = {k: make_permutation(q.n_options, k, cfg.data.permutation_seed) for k, q in qs.items()}
    backend = build_backend(cfg)

    text_of = {(r.question_id, r.agent_id, r.phase): r.text for r in tr.itertuples()}
    texts, keys, nopt = [], [], []
    for (qid, aid, phase), gen in sorted(text_of.items()):
        if qid not in qs:
            continue
        q = qs[qid]
        base = P.round0_messages(persona=cfg.debate.personas[0], subject=q.subject,
                                 question=q.question, options=list(q.options), perm=perms[qid],
                                 ask_confidence=cfg.debate.ask_verbalized_confidence)
        if phase == "pre":
            msgs = base
        else:
            peers = [text_of[(qid, b, "pre")] for b in (0, 1, 2) if b != aid]
            msgs = P.round1_messages(base=base, own_response=text_of[(qid, aid, "pre")],
                                     peer_responses=[P.strip_confidence(t) for t in peers])
        texts.append(backend.render(msgs) + gen)
        keys.append((qid, aid, phase))
        nopt.append(q.n_options)

    print(f"replaying readout for {len(texts)} states", flush=True)
    reads = backend.readout(texts, nopt)
    out = pd.DataFrame([
        {"question_id": k[0], "agent_id": k[1], "phase": k[2],
         "lp": ",".join(f"{v:.6f}" for v in r.letter_logprobs),
         "top": float(np.exp(r.letter_logprobs.max())),
         "margin": float(np.sort(r.letter_logprobs)[-1] - np.sort(r.letter_logprobs)[-2]),
         "entropy": float(-(np.exp(r.letter_logprobs) * r.letter_logprobs).sum()),
         "argmax": int(r.answer_index)}
        for k, r in zip(keys, reads, strict=True)])
    out.to_parquet(run_dir / "readout_full.parquet", index=False)
    print(f"wrote {run_dir/'readout_full.parquet'}  rows={len(out)}")

    orig = pd.read_parquet(run_dir / "agent_states.parquet")[
        ["question_id", "agent_id", "phase", "readout_answer"]]
    chk = orig.merge(out, on=["question_id", "agent_id", "phase"])
    print(f"replay reproduces the original answer on {(chk.readout_answer == chk.argmax).mean():.4f} "
          f"of {len(chk)} states  <- must be ~1.0 or the reconstruction is wrong")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "night2-debate")
