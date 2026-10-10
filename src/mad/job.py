"""The job: one named experiment, its fixed question order, and what makes a resume legitimate.

A long run is many chunks spread over several sessions. Without a job record, a resume could
silently mix work done under different settings: a different question sample, chunk size,
prompt, model revision or code version, each producing chunk directories that look identical.
The job file fixes all of those once, and every later session must match it or stop.

    <out_root>/<name>-job.json         identity + ordered questions + chunk membership + history
    <out_root>/<name>-questions.jsonl  full text of every question in the job (a local snapshot)
    <out_root>/<name>-round0-cNNN/     round 0 for one chunk, saved before any condition runs
    <out_root>/<name>-<cond>-cNNN/     one condition for one chunk
    <out_root>/<name>-<cond>/          a validated VIEW over that condition's complete chunks

Question order. The dataset is put in one fixed MASTER ORDER: subjects interleaved round-robin,
each subject's questions in an order set by a seeded hash of their ids. Any prefix is therefore
subject-balanced, and asking for more questions APPENDS to the existing ones instead of drawing a
new sample. The first ``pilot_size`` questions of the master order are the pilot pool; the main
pool is everything after it, so pilot questions can never reach the main test set.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from mad.data import Question
from mad.debate import ROWS_FILE, TRANSCRIPTS_FILE
from mad.provenance import COMPLETION, MANIFEST, git_state, is_complete, sha256_file

# Settings that may differ between sessions of one job: they change how fast or in what batch
# composition the work is done, not what is computed -- plus the job's size, which may grow
# (the question list itself is checked separately). Everything else is identity.
EXECUTION_FIELDS = ("run_id", "out_dir", "model.batch_size", "model.readout_batch_size",
                    "data.n_questions")


def question_sha256(q: Question) -> str:
    blob = json.dumps(asdict(q), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def master_order(pool: list[Question], seed: int) -> list[Question]:
    """Subject-interleaved fixed order of the whole pool (see module docstring)."""
    by_subject: dict[str, list[Question]] = defaultdict(list)
    for q in pool:
        by_subject[q.subject].append(q)
    for qs in by_subject.values():
        qs.sort(key=lambda q: hashlib.sha256(f"{seed}:{q.question_id}".encode()).hexdigest())
    order, depth = [], 0
    subjects = sorted(by_subject)
    while len(order) < len(pool):
        for s in subjects:
            if depth < len(by_subject[s]):
                order.append(by_subject[s][depth])
        depth += 1
    return order


def select(master: list[Question], pool: str, n: int, pilot_size: int) -> list[Question]:
    if pool not in ("pilot", "main"):
        raise ValueError(f"pool must be 'pilot' or 'main', got {pool!r}")
    if pilot_size >= len(master):
        raise ValueError(f"pilot_size {pilot_size} leaves no main pool in {len(master)} questions")
    src = master[:pilot_size] if pool == "pilot" else master[pilot_size:]
    if n > len(src):
        raise ValueError(f"asked for {n} questions; the {pool} pool has {len(src)}")
    return src[:n]


def chunks_of(ids: list[str], size: int) -> list[list[str]]:
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def chunk_tag(i: int) -> str:
    return f"c{i:03d}"


def _strip_execution(cfg_dict: dict) -> dict:
    out = json.loads(json.dumps(cfg_dict, default=str))
    for dotted in EXECUTION_FIELDS:
        cur = out
        *head, last = dotted.split(".")
        for h in head:
            cur = cur.get(h, {})
        cur.pop(last, None)
    return out


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def build_identity(*, name, cfg, dataset_key, pool, pilot_size, chunk_size, conditions,
                   templates_hash, backend_desc, master_ids_sha) -> dict:
    return {
        "name": name,
        "dataset": {"key": dataset_key, "name": cfg.data.dataset, "split": cfg.data.split,
                    "revision": cfg.data.dataset_revision},
        "pool": {"name": pool, "pilot_size": pilot_size, "order_seed": cfg.data.sample_seed,
                 "master_order_sha256": master_ids_sha},
        "chunk_size": chunk_size,
        "conditions": conditions,
        "config": _strip_execution(cfg.to_dict()),
        "prompt_template_hashes": templates_hash,
        "backend": backend_desc,
    }


class JobMismatch(RuntimeError):
    pass


def open_job(out_root: Path, identity: dict, questions: list[Question], *,
             allow_code_change: bool = False, execution: dict | None = None,
             chunks: list[list[str]] | None = None) -> dict:
    """Create the job file, or validate a resume/extension against it. Returns the job dict.

    Refuses: any identity difference; a question list that is not an extension of the recorded
    one (same ids and content, in the same order); an extension that would change the
    membership of a chunk that already has output; a different code commit unless allowed.
    """
    out_root = Path(out_root)
    path = out_root / f"{identity['name']}-job.json"
    qrec = [{"id": q.question_id, "sha256": question_sha256(q)} for q in questions]
    code = git_state()
    event = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_sha": code["git_sha"],
             "git_dirty": code["git_dirty"], "n_questions": len(questions),
             "execution": execution or {}}
    chunk_size = identity["chunk_size"]

    if not path.exists():
        job = {"identity": identity, "identity_sha256": _sha(identity), "code": code,
               "questions": qrec,
               "chunks": chunks if chunks is not None else chunks_of([q["id"] for q in qrec], chunk_size),
               "history": [{**event, "event": "created"}]}
        _atomic_write(path, json.dumps(job, indent=1))
        _write_snapshot(out_root, identity["name"], questions)
        return job

    job = json.loads(path.read_text())
    if job["identity_sha256"] != _sha(identity):
        diff = sorted(k for k in set(identity) | set(job["identity"])
                      if identity.get(k) != job["identity"].get(k))
        raise JobMismatch(f"{path} was created with different settings ({', '.join(diff)} differ). "
                          "Use a new --name, or restore the original settings.")
    old = job["questions"]
    if len(qrec) < len(old):
        raise JobMismatch(f"the job already holds {len(old)} questions; this request has "
                          f"{len(qrec)}. Ask for at least {len(old)}.")
    if qrec[: len(old)] != old:
        bad = next(i for i, (a, b) in enumerate(zip(qrec, old, strict=False)) if a != b)
        raise JobMismatch(f"question {bad} differs from the job's ({old[bad]['id']}); the question "
                          "order or content changed. Refusing to mix samples.")
    new_chunks = chunks if chunks is not None else chunks_of([q["id"] for q in qrec], chunk_size)
    for i, members in enumerate(job["chunks"]):
        if new_chunks[i] != members and _chunk_started(out_root, identity, i):
            raise JobMismatch(f"extending to {len(qrec)} questions would change the membership of "
                              f"chunk {chunk_tag(i)}, which already has output. Extend by a "
                              f"multiple of the chunk size ({chunk_size}) from a full last chunk.")
    if code["git_sha"] != job["code"]["git_sha"] and not allow_code_change:
        raise JobMismatch(f"this code is at {code['git_sha']}, the job was created at "
                          f"{job['code']['git_sha']}. Check out that commit, or pass "
                          "--allow-code-change to continue and record the change.")
    kind = "extended" if len(qrec) > len(old) else "resumed"
    if code["git_sha"] != job["code"]["git_sha"]:
        kind += "+code_change"
    job["questions"], job["chunks"] = qrec, new_chunks
    job["history"].append({**event, "event": kind})
    _atomic_write(path, json.dumps(job, indent=1))
    _write_snapshot(out_root, identity["name"], questions)
    return job


def _write_snapshot(out_root: Path, name: str, questions: list[Question]) -> None:
    lines = "".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in questions)
    _atomic_write(out_root / f"{name}-questions.jsonl", lines)


def _chunk_started(out_root: Path, identity: dict, i: int) -> bool:
    tag = chunk_tag(i)
    dirs = [f"{identity['name']}-round0-{tag}"] + [f"{identity['name']}-{c}-{tag}"
                                                   for c in identity["conditions"]]
    # a child job's chunks are subsets of its parent's; its round 0 lives with the parent
    return any((out_root / d).exists() for d in dirs)


def check_chunk_dir(run_dir: Path, job: dict, i: int) -> None:
    """A complete chunk directory must belong to this job and hold exactly its chunk's questions."""
    man = json.loads((Path(run_dir) / MANIFEST).read_text())
    if man.get("job_identity_sha256") != job["identity_sha256"]:
        raise JobMismatch(f"{run_dir} was written by a different job")
    if man.get("question_ids") != job["chunks"][i]:
        raise JobMismatch(f"{run_dir} holds different questions than chunk {chunk_tag(i)}")


# ----------------------------------------------------------------------------- merged view


VIEW_FILE = "view.json"


def build_view(out_root: Path, job: dict, cond: str, *, verify_hashes: bool = True) -> Path | None:
    """A validated, read-only view over one condition's complete chunks. Copies no arrays.

    The view's activation index points at the chunk directories' shard files. It is rebuilt
    whenever the set of complete chunks or any chunk's completion record changes. Validation:
    every chunk belongs to this job and holds its recorded questions; every shard matches the
    checksum written when it was produced (unless ``verify_hashes`` is off); state keys are
    unique; and each included question has exactly n_agents x 3 states.
    """
    out_root = Path(out_root)
    name = job["identity"]["name"]
    n_agents = job["identity"]["config"]["debate"]["n_agents"]
    parts = []
    for i in range(len(job["chunks"])):
        d = out_root / f"{name}-{cond}-{chunk_tag(i)}"
        if is_complete(d):
            check_chunk_dir(d, job, i)
            parts.append((i, d, sha256_file(d / COMPLETION)))
    if not parts:
        return None
    out = out_root / f"{name}-{cond}"
    stamp = [[i, d.name, h] for i, d, h in parts]
    if (out / VIEW_FILE).exists() and json.loads((out / VIEW_FILE).read_text()).get("chunks") == stamp:
        return out

    rows, idx, transcripts, metas = [], [], [], []
    for i, d, _ in parts:
        if verify_hashes:
            _check_hashes(d)
        r = pd.read_parquet(d / ROWS_FILE)
        if set(r.question_id) != set(job["chunks"][i]):
            raise JobMismatch(f"{d} rows cover different questions than chunk {chunk_tag(i)}")
        rows.append(r)
        idx.append(_view_index(d))
        metas.append(pd.read_parquet(d / "activation_meta.parquet"))
        transcripts.append((d / TRANSCRIPTS_FILE).read_text())

    rows = pd.concat(rows, ignore_index=True)
    idx = pd.concat(idx, ignore_index=True)
    files = list(dict.fromkeys(idx["shard_path"]))  # first-appearance order
    idx["shard"] = idx["shard_path"].map({f: k for k, f in enumerate(files)})
    if rows.duplicated(["question_id", "agent_id", "phase"]).any():
        raise JobMismatch(f"{cond}: duplicate (question, agent, phase) rows across chunks")
    if idx.duplicated(["question_id", "agent_id", "phase", "position"]).any():
        raise JobMismatch(f"{cond}: duplicate activation keys across chunks")
    per_q = rows.groupby("question_id").size()
    if (per_q != n_agents * 3).any():
        raise JobMismatch(f"{cond}: questions with the wrong number of states: "
                          f"{per_q[per_q != n_agents * 3].index[:5].tolist()}")
    shapes = {(int(m.iloc[0]["n_layers"]), int(m.iloc[0]["d_model"])) for m in metas}
    if len(shapes) != 1:
        raise JobMismatch(f"{cond}: chunks with different activation shapes {shapes}")

    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    rows.to_parquet(tmp / ROWS_FILE, index=False)
    idx.to_parquet(tmp / "activation_index.parquet", index=False)
    metas[0].to_parquet(tmp / "activation_meta.parquet", index=False)
    (tmp / TRANSCRIPTS_FILE).write_text("".join(transcripts))
    n_done, n_all = len(parts), sum(1 for c in job["chunks"] if c)
    first = json.loads((parts[0][1] / MANIFEST).read_text())
    (tmp / MANIFEST).write_text(json.dumps({
        "run_id": out.name, "view_of": [d.name for _, d, _ in parts], "condition": cond,
        "config": first.get("config"), "job_identity_sha256": job["identity_sha256"],
        "prompt_template_hashes": first.get("prompt_template_hashes"),
    }, indent=2, default=str))
    (tmp / VIEW_FILE).write_text(json.dumps({
        "chunks": stamp, "chunks_complete": n_done, "chunks_total": n_all,
        "complete": n_done == n_all, "questions": int(rows.question_id.nunique()),
        "questions_in_job": len(job["questions"]), "hashes_verified": verify_hashes,
    }, indent=2))
    (tmp / COMPLETION).write_text(json.dumps({
        "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "merged_from": [d.name for _, d, _ in parts],
        "row_counts": {"agent_states": len(rows), "activation_states": len(idx)},
        "shard_sha256": {}, "errors": [],
    }, indent=2))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out


def _view_index(d: Path) -> pd.DataFrame:
    """A chunk's activation index with every shard path rewritten relative to a sibling view."""
    ix = pd.read_parquet(d / "activation_index.parquet")
    if "shard_path" not in ix.columns:
        ix["shard_path"] = [f"acts_{int(x):04d}.npy" for x in ix["shard"]]
    if "layout" not in ix.columns:
        ix["layout"] = "layer_major"
    ix["shard_path"] = [os.path.normpath(os.path.join("..", d.name, p)) for p in ix["shard_path"]]
    return ix


def _check_hashes(d: Path) -> None:
    """Every file a directory's completion record lists -- its own and any it references --
    must match the checksum recorded when it was written."""
    done = json.loads((d / COMPLETION).read_text())
    for f, h in done["shard_sha256"].items():
        path = (d / f).resolve()
        if not path.exists():
            raise JobMismatch(f"{d.name}: {f} is missing")
        if sha256_file(path) != h:
            raise JobMismatch(f"{path} does not match its recorded checksum")


def verify_archive(out_root: Path, name: str) -> dict:
    """Full, independent verification of a job's archive. Checks bytes even when a view's
    metadata looks unchanged (``build_view`` skips work then; this never does).

    For every complete round-0 cache and condition chunk: every recorded checksum (activation
    files, rows, transcripts, index, activation metadata, referenced round-0 files); the chunk
    belongs to the job and holds its questions. For every condition view: unique keys, exactly
    n_agents x 3 rows per question, every index row loadable at the recorded dimensions, every
    stored value finite, and how many states exist per phase and position.
    Raises JobMismatch on the first failure; returns a report otherwise.
    """
    from mad.store import ActivationStore

    out_root = Path(out_root)
    job = json.loads((out_root / f"{name}-job.json").read_text())
    report = {"job": name, "identity_sha256": job["identity_sha256"], "chunks": {}, "views": {}}
    for i in range(len(job["chunks"])):
        tag = chunk_tag(i)
        r0 = out_root / f"{name}-round0-{tag}"
        if is_complete(r0):
            _check_hashes(r0)
            if json.loads((r0 / MANIFEST).read_text()).get("question_ids") != job["chunks"][i]:
                raise JobMismatch(f"{r0.name} holds different questions than chunk {tag}")
        for cond in job["identity"]["conditions"]:
            d = out_root / f"{name}-{cond}-{tag}"
            if is_complete(d):
                check_chunk_dir(d, job, i)
                _check_hashes(d)
                report["chunks"][d.name] = "ok"
    n_agents = job["identity"]["config"]["debate"]["n_agents"]
    for cond, d in view_dirs(out_root, name).items():
        rows = pd.read_parquet(d / ROWS_FILE)
        store = ActivationStore(d)
        ix = store.index
        if ix.duplicated(["question_id", "agent_id", "phase", "position"]).any():
            raise JobMismatch(f"{cond}: duplicate activation keys")
        if (rows.groupby("question_id").size() != n_agents * 3).any():
            raise JobMismatch(f"{cond}: a question has the wrong number of states")
        missing_rows = set(map(tuple, rows[["question_id", "agent_id", "phase"]].values)) - set(
            map(tuple, ix[ix.position == "readout_last"][["question_id", "agent_id", "phase"]].values))
        if missing_rows:
            raise JobMismatch(f"{cond}: {len(missing_rows)} states have no readout activation")
        for layer in store.layer_range():
            x = store.load_layer(layer)
            if x.shape != (len(ix), store.d_model):
                raise JobMismatch(f"{cond}: layer {layer} has shape {x.shape}")
            if not np.isfinite(x).all():
                raise JobMismatch(f"{cond}: non-finite values at layer {layer}")
        report["views"][cond] = {
            "questions": int(rows.question_id.nunique()), "states": int(len(ix)),
            "layers": store.n_layers + 1, "d_model": store.d_model,
            "by_phase_and_position": {f"{p}/{q}": int(n) for (p, q), n in
                                      ix.groupby(["phase", "position"]).size().items()},
        }
    return report


def view_dirs(out_root: Path, name: str) -> dict[str, Path]:
    """Condition -> merged view directory, for a job. Uses the job file, never name patterns."""
    out_root = Path(out_root)
    job = json.loads((out_root / f"{name}-job.json").read_text())
    out = {}
    for cond in job["identity"]["conditions"]:
        d = out_root / f"{name}-{cond}"
        if (d / VIEW_FILE).exists() and is_complete(d):
            out[cond] = d
    return out
