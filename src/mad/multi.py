"""Multi-condition debate collection over a SHARED, PERSISTED round 0.

Round 0 is generated once and reused by every condition. Two reasons:

1. **It makes the matched analysis structural.** Every condition starts each agent from a
   byte-identical initial response, so "the same agent, from the same starting answer, revised
   one way with peers and another way alone" is true by construction rather than by hoping two
   separate runs sampled identically.
2. It removes the duplicated round-0 cost, which is roughly a third of the compute for three
   conditions.

Round 0 is also SAVED before any condition runs (``save_round0``). A crash between conditions
used to regenerate round 0 on retry, so finished conditions and redone ones started from
different first answers while claiming to share them. Now a retry reloads the saved round 0 and
checks it belongs to the same questions and job.

Conditions differ only in what the agent is shown before it revises:

  self_reconsider     nothing -- the ordinary-instability floor
  debate              the two real peer responses, confidence line stripped
  debate_conf         the two real peer responses with confidence left in (ablation)
  debate_answer_only  the two real peers' stated answers only, no reasoning
  misleading_one      one fabricated, mildly worded peer for a random wrong answer + one real peer
  adversarial         two fabricated peers confidently asserting the same random wrong answer
  independent         not a revision: a FRESH round-0 answer with a new seed (the agent sees
                      nothing of its first answer) -- extra independent samples for comparing
                      debate with simply sampling more answers

Peer order is randomised per (condition, question, agent) from a stable seed and recorded, so
"panel member 1" is not systematically the lower-numbered agent or the fabricated one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mad import prompts as P
from mad.backends.base import Backend, Generation, OracleEntry, Readout, build_backend
from mad.data import Question
from mad.debate import ROWS_FILE, TRANSCRIPTS_FILE, RunResult, _akey, _row, peer_majority
from mad.permute import LETTERS, make_permutation
from mad.provenance import (
    COMPLETION,
    is_complete,
    prompt_hashes,
    sha256_file,
    write_completion,
    write_manifest,
)
from mad.store import ActivationWriter, decode, to_bf16_bits


class _NullGen:
    """Placeholder for the peer_exposure state, which has no generated text of its own."""

    text = ""
    n_prompt_tokens = 0
    n_completion_tokens = 0
    stop_reason = None


KINDS = ("self_reconsider", "debate", "debate_conf", "debate_answer_only", "adversarial",
         "misleading_one", "independent")


def derive_seed(*parts) -> int:
    """A stable 31-bit seed from any identifying parts. Distinct parts give independent streams."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def stable_rng(*parts) -> np.random.Generator:
    return np.random.default_rng(derive_seed(*parts))


def wrong_target(seed: int, kind: str, qid: str, agent: int, gold: int, pre: int, n: int) -> int:
    """A wrong option, uniformly at random among those that are neither gold nor the agent's own.

    The previous rule took the lowest eligible letter. With the strong position bias measured on
    the 1B model (accuracy 70% when the answer was A, 23% when it was D), that confounded "a peer
    pushed a wrong answer" with "a peer pushed answer A".
    """
    eligible = [i for i in range(n) if i != gold and i != pre]
    if not eligible:  # two options, agent already wrong: only the agent's own wrong answer is left
        eligible = [i for i in range(n) if i != gold]
    return int(stable_rng(seed, kind, "target", qid, agent).choice(eligible))


class InvalidRecords(RuntimeError):
    """Readouts that must not be saved as a completed chunk."""


def validate_reads(reads, metas, where: str) -> None:
    """Finite states at every layer and position, normalised distributions, and an exact replay
    wherever one was checked. Raised BEFORE a completion record is written, so a chunk with bad
    records fails (and is redone) instead of being archived as complete."""
    for r, m in zip(reads, metas, strict=True):
        tag = f"{where} {m.get('question_id')} agent {m.get('agent_id')} {m.get('phase')}"
        if not np.isfinite(r.hidden.astype(np.float32)).all():
            raise InvalidRecords(f"non-finite readout state: {tag}")
        for pos, h in r.extra.items():
            if not np.isfinite(h.astype(np.float32)).all():
                raise InvalidRecords(f"non-finite {pos} state: {tag}")
        for name, lp in [("readout", r.letter_logprobs), *r.extra_logprobs.items()]:
            if not np.isfinite(lp).all() or abs(float(np.exp(lp).sum()) - 1) > 1e-3:
                raise InvalidRecords(f"invalid {name} letter distribution: {tag}")
        if r.alignment.get("replay_exact") is False:
            raise InvalidRecords(f"readout did not replay the generated tokens exactly: {tag}")


def request_id(job: str, chunk: str, phase: str, condition: str, qid: str, agent: int) -> str:
    return f"{job}/{chunk}/{condition}/{phase}/{qid}/a{agent}"


def condition_seed(cfg, chunk: str, condition: str) -> int:
    return derive_seed(cfg.seed, "round1", condition, chunk)


def round0_seed(cfg, chunk: str) -> int:
    return derive_seed(cfg.seed, "round0", chunk)


# ----------------------------------------------------------------------------- round 0


@dataclass
class Round0:
    metas: list[dict]  # one per (question, agent), in question-then-agent order
    messages: list[list[dict]]
    prompts: list[str]
    gens: list[Generation]
    reads: list[Readout]
    seed: int
    loaded: bool = False
    timings: dict | None = None
    # record positions in the SAVED round-0 files (differs from 0..n-1 after ``subset``)
    offsets: list[int] | None = None

    def key(self, i: int) -> tuple[str, int]:
        return self.metas[i]["question_id"], self.metas[i]["agent_id"]

    def offset(self, i: int) -> int:
        return self.offsets[i] if self.offsets is not None else i

    def subset(self, question_ids: list[str]) -> Round0:
        """The same saved records, restricted to some questions (a child job's subset)."""
        keep = [i for i, m in enumerate(self.metas) if m["question_id"] in set(question_ids)]
        return Round0([self.metas[i] for i in keep], [self.messages[i] for i in keep],
                      [self.prompts[i] for i in keep], [self.gens[i] for i in keep],
                      [self.reads[i] for i in keep], self.seed, loaded=True,
                      offsets=[self.offset(i) for i in keep])


def _setup(cfg, questions, backend):
    perms = {q.question_id: make_permutation(q.n_options, q.question_id, cfg.data.permutation_seed)
             for q in questions}
    gold = {q.question_id: perms[q.question_id].original_to_display(q.gold_index) for q in questions}
    if backend is None:
        oracle = ({q.question_id: OracleEntry(gold[q.question_id], q.n_options) for q in questions}
                  if cfg.model.backend == "stub" else None)
        backend = build_backend(cfg, oracle=oracle)
    return perms, gold, backend


def generate_round0(cfg, questions, backend, perms, job="dev", chunk="c000") -> Round0:
    n_agents = cfg.debate.n_agents
    metas, messages, prompts = [], [], []
    for q in questions:
        for a in range(n_agents):
            msgs = P.round0_messages(
                persona=cfg.debate.personas[a % len(cfg.debate.personas)],
                subject=q.subject, question=q.question, options=list(q.options),
                perm=perms[q.question_id], ask_confidence=cfg.debate.ask_verbalized_confidence,
            )
            prompt = backend.render(msgs)
            m0 = {"question_id": q.question_id, "agent_id": a, "phase": "pre", "condition": "shared",
                  "request_id": request_id(job, chunk, "pre", "shared", q.question_id, a),
                  "chunk": chunk, "gen_start_char": len(prompt)}
            planned = backend.plan_answer(phase="pre", question_id=q.question_id, agent_id=a)
            if planned is not None:
                m0["planned_answer"] = planned
            metas.append(m0)
            messages.append(msgs)
            prompts.append(prompt)
    seed = round0_seed(cfg, chunk)
    for m in metas:
        m["gen_seed"] = seed
    gens = backend.generate(prompts, seed=seed, meta=metas)
    texts = [p + g.text for p, g in zip(prompts, gens, strict=True)]
    n_opts = {q.question_id: q.n_options for q in questions}
    ro_meta = [{**m, "gen_token_ids": g.token_ids} for m, g in zip(metas, gens, strict=True)]
    reads = backend.readout(texts, [n_opts[m["question_id"]] for m in metas], meta=ro_meta)
    validate_reads(reads, metas, "round 0")
    return Round0(metas, messages, prompts, gens, reads, seed)


R0_RECORDS = "round0.jsonl"
R0_HIDDEN = "round0_hidden_{pos}.npy"


def save_round0(r0: Round0, out_dir: Path, identity: dict) -> Path:
    """Persist round 0 atomically: write to a temporary directory, then rename into place."""
    out_dir = Path(out_dir)
    tmp = out_dir.with_name(out_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with open(tmp / R0_RECORDS, "w") as fh:
        for m, msgs, prompt, g, r in zip(r0.metas, r0.messages, r0.prompts, r0.gens, r0.reads,
                                         strict=True):
            fh.write(json.dumps({
                "meta": m, "messages": msgs, "prompt": prompt, "text": g.text,
                "n_prompt_tokens": g.n_prompt_tokens, "n_completion_tokens": g.n_completion_tokens,
                "token_ids": g.token_ids, "stop_reason": g.stop_reason,
                "letter_logprobs": [float(x) for x in r.letter_logprobs],
                "answer_index": r.answer_index, "n_tokens": r.n_tokens,
                "extra_positions": sorted(r.extra),
                "extra_logprobs": {k: [float(x) for x in v] for k, v in r.extra_logprobs.items()},
                "alignment": r.alignment,
            }) + "\n")
    hidden = {"readout_last": np.stack([r.hidden for r in r0.reads]).astype(np.float32)}
    for pos in sorted({p for r in r0.reads for p in r.extra}):
        arr = np.full_like(hidden["readout_last"], np.nan)  # NaN marks a missing position
        for i, r in enumerate(r0.reads):
            if pos in r.extra:
                arr[i] = r.extra[pos]
        hidden[pos] = arr
    for pos, arr in hidden.items():
        np.save(tmp / R0_HIDDEN.format(pos=pos), to_bf16_bits(arr))  # bfloat16, as every shard
    (tmp / "manifest.json").write_text(json.dumps({**identity, "seed": r0.seed,
                                                   "n_records": len(r0.metas)}, indent=2))
    write_completion(tmp, row_counts={"records": len(r0.metas)},
                     shard_hashes={p.name: sha256_file(p) for p in sorted(tmp.iterdir())
                                   if p.name != COMPLETION})
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(tmp, out_dir)
    return out_dir


def load_round0(out_dir: Path, identity: dict) -> Round0 | None:
    """Reload a saved round 0, or None if absent. Refuses one that belongs to something else."""
    out_dir = Path(out_dir)
    if not is_complete(out_dir):
        return None
    man = json.loads((out_dir / "manifest.json").read_text())
    for k, v in identity.items():
        if man.get(k) != v:
            raise RuntimeError(f"saved round 0 in {out_dir} does not match this chunk ({k} differs); "
                               "refusing to reuse it")
    done = json.loads((out_dir / COMPLETION).read_text())
    for name, h in done["shard_sha256"].items():
        if sha256_file(out_dir / name) != h:
            raise RuntimeError(f"{out_dir / name} fails its checksum; refusing to reuse round 0")
    recs = [json.loads(line) for line in (out_dir / R0_RECORDS).read_text().splitlines()]
    hidden = {p.name[len("round0_hidden_"):-4]: decode(np.load(p)) for p in out_dir.glob("round0_hidden_*.npy")}
    metas, messages, prompts, gens, reads = [], [], [], [], []
    for i, rec in enumerate(recs):
        metas.append(rec["meta"])
        messages.append(rec["messages"])
        prompts.append(rec["prompt"])
        gens.append(Generation(rec["text"], rec["n_prompt_tokens"], rec["n_completion_tokens"],
                               rec["token_ids"], rec["stop_reason"]))
        lp = np.asarray(rec["letter_logprobs"], dtype=np.float32)
        reads.append(Readout(hidden=hidden["readout_last"][i], letter_logprobs=lp,
                             answer_index=rec["answer_index"], n_tokens=rec["n_tokens"],
                             extra={p: hidden[p][i] for p in rec["extra_positions"]},
                             extra_logprobs={k: np.asarray(v, dtype=np.float32)
                                             for k, v in rec.get("extra_logprobs", {}).items()},
                             alignment=rec.get("alignment", {})))
    return Round0(metas, messages, prompts, gens, reads, man["seed"], loaded=True)


# ----------------------------------------------------------------------------- round 1


def _peer_view(cfg, kind, cond, q, a, r0_ans, r0_stated, own_text, perms, gold):
    """What agent ``a`` is shown, in display order, with a record of where each entry came from.

    Missingness policy: a real peer's shown answer is what that peer STATED (-1 if its response
    had no valid answer field), never the readout fallback. Natural peers are shown their raw
    text; answer-only peers with no stated answer show PEER_NO_ANSWER. The fallback is used only
    where it is labelled as such: the agent's own canonical answer and the panel vote.
    """
    qid, n_agents = q.question_id, cfg.debate.n_agents
    others = [b for b in range(n_agents) if b != a]
    pre = r0_ans[(qid, a)]
    shown = perms[qid].display_options(list(q.options))
    target = -1
    if kind in ("self_reconsider", "independent"):
        return None, [], [], target
    if kind == "misleading_one":
        # One dissenting peer, restrained wording, plus one genuine peer chosen at random. Two
        # emphatic fabricated peers produced 99% adoption, which measures whether the model
        # copies a confident assertion rather than anything about conformity.
        target = wrong_target(cfg.seed, kind, qid, a, gold[qid], pre, q.n_options)
        real = others[int(stable_rng(cfg.seed, kind, "peer", qid, a).integers(len(others)))]
        entries = [(P.MILD_PEER.format(letter=LETTERS[target], option=shown[target]), target,
                    "fabricated"),
                   (P.strip_confidence(own_text[(qid, real)]), r0_stated[(qid, real)], f"agent{real}")]
    elif kind == "adversarial":
        target = wrong_target(cfg.seed, kind, qid, a, gold[qid], pre, q.n_options)
        text = P.ADVERSARIAL_PEER.format(letter=LETTERS[target], option=shown[target])
        entries = [(text, target, "fabricated")] * len(others)
    elif kind == "debate_answer_only":
        entries = []
        for b in others:
            ans = r0_stated[(qid, b)]
            text = (P.PEER_ANSWER_ONLY.format(letter=LETTERS[ans], option=shown[ans]) if ans >= 0
                    else P.PEER_NO_ANSWER)
            entries.append((text, ans, f"agent{b}"))
    else:  # debate, debate_conf
        keep_conf = bool(cond.get("peer_confidence"))
        entries = [(own_text[(qid, b)] if keep_conf else P.strip_confidence(own_text[(qid, b)]),
                    r0_stated[(qid, b)], f"agent{b}") for b in others]
    order = stable_rng(cfg.seed, kind, "order", qid, a).permutation(len(entries))
    entries = [entries[i] for i in order]
    return [e[0] for e in entries], [e[1] for e in entries], [e[2] for e in entries], target


def run_condition(cfg, questions, cond, r0: Round0, backend, perms, gold, final_dir: Path,
                  job="dev", chunk="c000", extra_manifest: dict | None = None,
                  round0_dir: Path | None = None) -> RunResult:
    """One condition's round 1 for one chunk, written atomically to ``final_dir``.

    With ``round0_dir`` (a saved, verified round 0 next to ``final_dir``), the round-0 states are
    REFERENCED from that cache rather than copied into every condition: the activation index
    points at the cache's row-major files, and the completion record carries their checksums.
    """
    name, kind = cond["name"], cond["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown condition kind {kind!r}; expected one of {KINDS}")
    n_agents = cfg.debate.n_agents
    n_opts = {q.question_id: q.n_options for q in questions}
    cfg.run_id = cond["run_id"]
    final_dir = Path(final_dir)
    tmp = final_dir.with_name(final_dir.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    seed1 = condition_seed(cfg, chunk, name)
    write_manifest(cfg, run_dir=tmp, extra={
        "prompt_template_hashes": prompt_hashes(P.TEMPLATES), "n_questions": len(questions),
        "question_ids": [q.question_id for q in questions],
        "condition": name, "kind": kind, "condition_spec": cond, "shared_round0": True,
        "round0_seed": r0.seed, "round0_reloaded": r0.loaded, "round1_seed": seed1,
        "peer_shows_confidence": bool(cond.get("peer_confidence", False)),
        "backend_n_layers": backend.n_layers, "backend_d_model": backend.d_model,
        "chunk": chunk, "job": job, **(extra_manifest or {}),
    })
    writer = ActivationWriter(tmp, backend.n_layers, backend.d_model, cfg.activations.shard_size)

    def add_states(m, read):
        writer.add(_akey(cfg, m), read.hidden)
        for pos, h in read.extra.items():
            writer.add({**_akey(cfg, m), "position": pos}, h)

    timings = {}
    t_stage = time.time()
    refs = []  # round-0 states referenced from the shared cache
    rel0 = f"../{Path(round0_dir).name}" if round0_dir is not None else None
    rows, transcripts = [], []
    r0_ans, r0_stated, own_text = {}, {}, {}
    for i, (meta, gen, read) in enumerate(zip(r0.metas, r0.gens, r0.reads, strict=True)):
        m = {**meta, "condition": name}
        row = _row(cfg, m, gen, read, n_opts[m["question_id"]], gold, perms)
        rows.append(row)
        r0_ans[r0.key(i)] = row["answer"]
        r0_stated[r0.key(i)] = row["freetext_answer"]
        own_text[r0.key(i)] = gen.text
        transcripts.append({**m, "messages": r0.messages[i], "prompt": r0.prompts[i],
                            "text": gen.text, "token_ids": gen.token_ids,
                            "stop_reason": gen.stop_reason})
        if rel0 is None:
            add_states(m, read)
        else:
            for pos in ["readout_last", *sorted(read.extra)]:
                refs.append({**_akey(cfg, m), "position": pos, "offset": r0.offset(i),
                             "shard_path": f"{rel0}/{R0_HIDDEN.format(pos=pos)}"})
    base_msgs = {r0.key(i): r0.messages[i] for i in range(len(r0.metas))}

    msgs1, prompts1, metas1 = [], [], []
    for q in questions:
        qid = q.question_id
        for a in range(n_agents):
            pre = r0_ans[(qid, a)]
            panel = [r0_ans[(qid, b)] for b in range(n_agents) if b != a]
            texts, shown_ans, sources, target = _peer_view(cfg, kind, cond, q, a, r0_ans,
                                                           r0_stated, own_text, perms, gold)
            stated_shown = [x for x in shown_ans if x >= 0]  # missing answers do not vote
            maj = peer_majority(stated_shown) if stated_shown else None
            if kind == "independent":
                msgs = base_msgs[(qid, a)]  # the round-0 prompt again, sampled afresh
            else:
                msgs = P.round1_messages(base=base_msgs[(qid, a)], own_response=own_text[(qid, a)],
                                         peer_responses=texts,
                                         ask_confidence=cfg.debate.ask_verbalized_confidence)
            prompt = backend.render(msgs)
            planned = backend.plan_answer(phase="post", question_id=qid, agent_id=a, pre=pre,
                                          peer_majority=maj, condition=kind)
            pmaj = peer_majority(panel)
            metas1.append({
                "question_id": qid, "agent_id": a, "phase": "post", "condition": name,
                "request_id": request_id(job, chunk, "post", name, qid, a), "chunk": chunk,
                "gen_seed": seed1, "gen_start_char": len(prompt),
                **({"planned_answer": planned} if planned is not None else {}),
                # what the agent was SHOWN (the legacy column names keep this meaning)
                "peer_majority": -1 if maj is None else maj,
                "n_peers_agree_pre": sum(1 for p in stated_shown if p == pre) if shown_ans else -1,
                "peers_agree_each_other": (int(len(stated_shown) == len(shown_ans)
                                               and len(set(stated_shown)) == 1) if shown_ans else -1),
                "peer_unanimous": (int(len(stated_shown) == len(shown_ans)
                                       and len(set(stated_shown)) == 1) if shown_ans else -1),
                "shown_peer_answers": ",".join(map(str, shown_ans)),
                "shown_peer_sources": ",".join(sources),
                "shown_peer_majority": -1 if maj is None else maj,
                "n_shown_agree_pre": sum(1 for p in stated_shown if p == pre) if shown_ans else -1,
                "n_shown_missing": sum(1 for p in shown_ans if p < 0) if shown_ans else -1,
                "misleading_target": target,
                # what its real panel-mates answered, whatever it was shown
                "panel_peer_answers": ",".join(map(str, panel)),
                "panel_peer_majority": -1 if pmaj is None else pmaj,
                "n_panel_agree_pre": sum(1 for p in panel if p == pre),
            })
            msgs1.append(msgs)
            prompts1.append(prompt)

    # Third state: after the agent has seen the intervention but BEFORE it commits to a
    # revision. The post state exists only once the answer is decided, so it may be reading a
    # decision rather than predicting one.
    metas_pe = [{**m, "phase": "peer_exposure", "pre_answer": r0_ans[(m["question_id"], m["agent_id"])],
                 "request_id": m["request_id"].replace("/post/", "/peer_exposure/"),
                 "gen_start_char": None}
                for m in metas1]
    t_stage = time.time()
    reads_pe = backend.readout(prompts1, [n_opts[m["question_id"]] for m in metas1], meta=metas_pe)
    timings["exposure_readout_s"] = round(time.time() - t_stage, 2)
    validate_reads(reads_pe, metas_pe, f"{name} exposure")
    for m, read in zip(metas_pe, reads_pe, strict=True):
        row = _row(cfg, m, _NullGen(), read, n_opts[m["question_id"]], gold, perms)
        row["post_equals_peer_majority"] = int(row["answer"] == m["peer_majority"])
        row["peers_agree_each_other"] = m["peers_agree_each_other"]
        rows.append(row)
        add_states(m, read)

    t_stage = time.time()
    gens1 = backend.generate(prompts1, seed=seed1, meta=metas1)
    timings["generation_s"] = round(time.time() - t_stage, 2)
    texts1 = [p + g.text for p, g in zip(prompts1, gens1, strict=True)]
    ro_meta1 = [{**m, "gen_token_ids": g.token_ids} for m, g in zip(metas1, gens1, strict=True)]
    t_stage = time.time()
    reads1 = backend.readout(texts1, [n_opts[m["question_id"]] for m in metas1], meta=ro_meta1)
    timings["readout_s"] = round(time.time() - t_stage, 2)
    validate_reads(reads1, metas1, f"{name} post")
    t_stage = time.time()
    for i, (m, gen, read) in enumerate(zip(metas1, gens1, reads1, strict=True)):
        row = _row(cfg, m, gen, read, n_opts[m["question_id"]], gold, perms)
        row["post_equals_peer_majority"] = int(row["answer"] == m["peer_majority"])
        row["peers_agree_each_other"] = m["peers_agree_each_other"]
        rows.append(row)
        transcripts.append({**{k: v for k, v in m.items() if k != "planned_answer"},
                            "messages": msgs1[i], "prompt": prompts1[i], "text": gen.text,
                            "token_ids": gen.token_ids, "stop_reason": gen.stop_reason})
        add_states(m, read)

    index = writer.close()
    if refs:
        index = _combine_index(tmp, index, refs)
    frame = pd.DataFrame(rows)
    frame["subject"] = frame["question_id"].map({q.question_id: q.subject for q in questions})
    frame.to_parquet(tmp / ROWS_FILE, index=False)
    pd.DataFrame(transcripts).drop(columns=["planned_answer"], errors="ignore").to_json(
        tmp / TRANSCRIPTS_FILE, orient="records", lines=True)
    hashes = {p.name: sha256_file(p) for p in writer.shard_paths}
    for f in (ROWS_FILE, TRANSCRIPTS_FILE, "activation_index.parquet", "activation_meta.parquet"):
        hashes[f] = sha256_file(tmp / f)
    if refs:  # the referenced round-0 files, with the checksums the cache recorded for them
        r0_done = json.loads((Path(round0_dir) / COMPLETION).read_text())["shard_sha256"]
        for path in sorted({r["shard_path"] for r in refs}):
            hashes[path] = r0_done[Path(path).name]
    timings["write_s"] = round(time.time() - t_stage, 2)
    write_completion(tmp, row_counts={"agent_states": len(frame), "activation_states": len(index),
                                      "questions": len(questions)}, shard_hashes=hashes,
                     extra={"timings": timings})
    if final_dir.exists():
        shutil.rmtree(final_dir)  # a stale, incomplete attempt
    os.replace(tmp, final_dir)
    return RunResult(run_dir=final_dir, rows=frame, n_states=len(index), timings=timings)


def _combine_index(run_dir: Path, own: pd.DataFrame, refs: list[dict]) -> pd.DataFrame:
    """Referenced round-0 rows first (as if they were the first shards), then this directory's
    own shards; every row names its file and layout explicitly."""
    ref = pd.DataFrame(refs)
    files = sorted(ref["shard_path"].unique())
    ref["shard"] = ref["shard_path"].map({f: i for i, f in enumerate(files)})
    ref["layout"] = "row_major"
    own = own.copy()
    own["shard_path"] = [f"acts_{int(x):04d}.npy" for x in own["shard"]]
    own["shard"] = own["shard"] + len(files)
    own["layout"] = "layer_major"
    index = pd.concat([ref, own], ignore_index=True)
    index.to_parquet(run_dir / "activation_index.parquet", index=False)
    meta = pd.read_parquet(run_dir / "activation_meta.parquet")
    meta["n_states"] = len(index)
    meta.to_parquet(run_dir / "activation_meta.parquet", index=False)
    return index


def run_conditions(cfg, questions: list[Question], conditions: list[dict],
                   backend: Backend | None = None, **kw) -> dict[str, RunResult]:
    """Back-compatible entry point: ``run_chunk`` without the round-0 handle."""
    return run_chunk(cfg, questions, conditions, backend, **kw)[0]


def run_chunk(
    cfg,
    questions: list[Question],
    conditions: list[dict],
    backend: Backend | None = None,
    *,
    job: str = "dev",
    chunk: str = "c000",
    round0_dir: str | Path | None = None,
    run_dirs: dict[str, Path] | None = None,
    extra_manifest: dict | None = None,
    parent_round0: dict | None = None,
) -> tuple[dict[str, RunResult], Round0]:
    """Round 0 once (reloaded from ``round0_dir`` if already saved), then every condition whose
    output directory is not already complete. Returns results for ALL requested conditions.

    ``parent_round0`` -- {"dir", "job", "question_ids"} -- makes this a CHILD chunk: round 0 is
    taken from the parent job's saved cache (verified against the parent's identity) and
    restricted to this chunk's questions. It is never regenerated.
    """
    perms, gold, backend = _setup(cfg, questions, backend)
    if parent_round0 is not None:
        pdir = Path(parent_round0["dir"])
        pident = {"job": parent_round0["job"], "chunk": chunk,
                  "question_ids": parent_round0["question_ids"], "seed": round0_seed(cfg, chunk)}
        full = load_round0(pdir, pident)
        if full is None:
            raise RuntimeError(f"parent round 0 {pdir} is missing or incomplete; a child job "
                               "never generates its own round 0")
        r0 = full.subset([q.question_id for q in questions])
        round0_dir = pdir
    else:
        ident = {"job": job, "chunk": chunk, "question_ids": [q.question_id for q in questions],
                 "seed": round0_seed(cfg, chunk)}
        r0 = load_round0(Path(round0_dir), ident) if round0_dir else None
    if r0 is None:
        t = time.time()
        r0 = generate_round0(cfg, questions, backend, perms, job=job, chunk=chunk)
        r0.timings = {"round0_s": round(time.time() - t, 2)}
        if round0_dir:
            save_round0(r0, Path(round0_dir), ident)
    results: dict[str, RunResult] = {}
    for cond in conditions:
        final = (run_dirs or {}).get(cond["name"]) or Path(cfg.out_dir) / cond["run_id"]
        if is_complete(final):
            results[cond["name"]] = RunResult(final, pd.read_parquet(final / ROWS_FILE),
                                              len(pd.read_parquet(final / "activation_index.parquet")))
            continue
        results[cond["name"]] = run_condition(cfg, questions, cond, r0, backend, perms, gold, final,
                                              job=job, chunk=chunk, extra_manifest=extra_manifest,
                                              round0_dir=Path(round0_dir) if round0_dir else None)
    return results, r0
