"""Resumable multi-condition experiment driver. Same code on the laptop CPU or a Colab GPU.

    python scripts/run_experiment.py \\
        --config configs/base.yaml configs/colab_12b.yaml \\
        --out-root /content/drive/MyDrive/mas_runs --name pilot12b \\
        --dataset mmlu-pro --pool pilot --n-questions 120 \\
        --conditions self_reconsider debate misleading_one --chunk 20 --max-hours 6

Add --smoke to run 2 questions through every condition into a throwaway job, check the outputs
(shapes, finite and non-degenerate values, normalised answer distributions, per-row option
ranges, batched-vs-single readout agreement, memory under worst-case lengths), project the full
run, and stop. Run it first: if something is wrong with the 12B load path it fails in minutes.

Design points:
  * A job file (mad.job) fixes the question order, chunking, settings, prompts, model identity
    and code commit. A resume must match it or the driver refuses to start; asking for more
    questions appends to the same order instead of drawing a new sample.
  * Work is split into chunks. Round 0 for a chunk is saved before any condition runs, and each
    condition directory is written to a temporary name and renamed only when complete, so a
    disconnect costs at most one chunk and never leaves a half-written directory that looks done.
  * The model is loaded ONCE and reused across chunks (Gemma-3-12B is 26 GB of weights).
  * Two consecutive chunk failures, or any out-of-memory error, stop the run rather than burning
    compute failing every remaining chunk. --max-hours caps total wall time across sessions.
  * After the chunks, each condition gets a validated merged view (no copied arrays).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad import prompts as P  # noqa: E402
from mad.backends.base import build_backend  # noqa: E402
from mad.config import load_config  # noqa: E402
from mad.data import load_mc_dataset, load_mmlu_pro, synthetic_questions  # noqa: E402
from mad.debate import ROWS_FILE  # noqa: E402
from mad.job import (  # noqa: E402
    build_identity,
    build_view,
    check_chunk_dir,
    chunk_tag,
    master_order,
    open_job,
    select,
)
from mad.multi import run_chunk  # noqa: E402
from mad.permute import make_permutation  # noqa: E402
from mad.provenance import is_complete, package_versions, prompt_hashes  # noqa: E402
from mad.store import ActivationStore  # noqa: E402

CONDITION_SPECS = {
    "self_reconsider": {"kind": "self_reconsider"},
    "debate": {"kind": "debate", "peer_confidence": False},
    "debate_conf": {"kind": "debate_conf", "peer_confidence": True},
    "debate_answer_only": {"kind": "debate_answer_only"},
    "misleading_one": {"kind": "misleading_one"},
    "adversarial": {"kind": "adversarial"},
    "independent": {"kind": "independent"},
}
DATASETS = {
    "mmlu-pro": ("TIGER-Lab/MMLU-Pro", "test"),
    "arc-challenge": ("allenai/ai2_arc:ARC-Challenge", "test"),
    "arc-easy": ("allenai/ai2_arc:ARC-Easy", "test"),
    "synthetic": ("synthetic", "none"),
}
MAX_CONSECUTIVE_FAILURES = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_overrides(pairs: list[str] | None) -> dict:
    out = {}
    for kv in pairs or []:
        k, v = kv.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_pool(dataset: str, cfg):
    if dataset == "mmlu-pro":
        return load_mmlu_pro(cfg)
    if dataset in ("arc-challenge", "arc-easy"):
        name = "ARC-Challenge" if dataset == "arc-challenge" else "ARC-Easy"
        return load_mc_dataset("allenai/ai2_arc", name, "test")
    if dataset == "synthetic":
        return synthetic_questions(2000, n_options=4, seed=cfg.data.sample_seed)
    raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")


def is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def gpu_stats(reset: bool = False) -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            out = {
                "device": torch.cuda.get_device_name(0),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
            if reset:
                torch.cuda.reset_peak_memory_stats()
            return out
    except Exception:
        pass
    return {}


def diagnostics(run_dir: Path) -> dict:
    """Stop reasons, parse rates and stated/readout agreement -- evidence for the token budget."""
    r = pd.read_parquet(run_dir / ROWS_FILE)
    out = {}
    for phase in ("pre", "post"):
        s = r[r.phase == phase]
        if len(s):
            out[phase] = {
                "stated_answer_parsed": round(float(s.freetext_parse_ok.mean()), 4),
                "hit_token_cap": round(float((s.stop_reason == "length").mean()), 4),
                "stated_agrees_readout": round(float(s.freetext_agrees_readout.mean()), 4),
                "median_tokens": float(s.n_completion_tokens.median()),
            }
    return out


# ----------------------------------------------------------------------------- smoke checks


def smoke_checks(res: dict, r0, backend, cfg, n_q: int) -> dict:
    """Assert the outputs are the shape the analysis expects AND that the values are sane."""
    n_agents = cfg.debate.n_agents
    n_layers = backend.n_layers
    d_model = backend.d_model
    report = {}
    for cond, r in res.items():
        rows = r.rows
        assert set(rows.phase) == {"pre", "peer_exposure", "post"}, f"{cond}: phases {set(rows.phase)}"
        assert len(rows) == n_q * n_agents * 3, f"{cond}: {len(rows)} rows"
        store = ActivationStore(r.run_dir)
        assert store.n_layers == n_layers, f"{cond}: {store.n_layers} layers, expected {n_layers}"
        for layer in sorted({0, n_layers // 2, n_layers}):
            x = store.load_layer(layer)
            assert x.shape[1] == d_model, f"{cond}: layer {layer} shape {x.shape}"
            assert np.isfinite(x).all(), f"{cond}: non-finite activations at layer {layer}"
        assert x.std(axis=0).max() > 0, f"{cond}: every stored state is identical"
        positions = store.index.position.value_counts().to_dict()
        assert positions.get("readout_last") == len(rows), f"{cond}: readout states {positions}"
        for _, row in rows.iterrows():
            lp = np.array([float(v) for v in row.readout_logprobs.split(",")])
            assert len(lp) == row.n_options, f"{cond}: {len(lp)} letter scores for {row.n_options} options"
            assert abs(np.exp(lp).sum() - 1) < 1e-3, f"{cond}: answer distribution not normalised"
            assert 0 <= row.readout_answer < row.n_options and 0 <= row.answer < row.n_options
        missing = {}
        for p in cfg.activations.positions:
            if p == "readout_last":
                continue
            col = rows[f"pos_{p}_logprobs"]
            for v in col[col != ""]:
                lp = np.array([float(x) for x in v.split(",")])
                assert abs(np.exp(lp).sum() - 1) < 1e-3, f"{cond}: {p} distribution not normalised"
            reasons = rows[f"pos_{p}_missing"]
            missing[p] = reasons[reasons != ""].value_counts().to_dict()
        gen_rows = rows[rows.phase != "peer_exposure"]
        replay = gen_rows.replay_exact.value_counts(dropna=False).to_dict()
        if hasattr(backend, "count_tokens"):  # a real tokenizer: every generated row must replay exactly
            assert (gen_rows.replay_exact == True).all(), f"{cond}: inexact replay {replay}"  # noqa: E712
        report[cond] = {"positions": positions, "missing_positions": missing,
                        "replay_exact": {str(k): int(v) for k, v in replay.items()},
                        "answer_source": rows.answer_source.value_counts().to_dict(),
                        "stated_status": rows.stated_status.value_counts().to_dict(),
                        **diagnostics(r.run_dir)}
    if hasattr(backend, "count_tokens"):  # a real model: batched and one-at-a-time must agree
        k = min(4, len(r0.prompts))
        texts = [p + g.text for p, g in zip(r0.prompts[:k], r0.gens[:k], strict=True)]
        n_opts = [len(r.letter_logprobs) for r in r0.reads[:k]]
        single = backend.readout(texts, n_opts, meta=r0.metas[:k], batch_size=1)
        batched = backend.readout(texts, n_opts, meta=r0.metas[:k], batch_size=k)
        dlp = max(float(np.abs(a.letter_logprobs - b.letter_logprobs).max())
                  for a, b in zip(single, batched, strict=True))
        cos = min(float(np.dot(a.hidden[-1].astype(np.float32), b.hidden[-1].astype(np.float32))
                        / (np.linalg.norm(a.hidden[-1].astype(np.float32))
                           * np.linalg.norm(b.hidden[-1].astype(np.float32)) + 1e-9))
                  for a, b in zip(single, batched, strict=True))
        agree = sum(a.answer_index == b.answer_index for a, b in zip(single, batched, strict=True))
        report["batched_vs_single"] = {"max_abs_logprob_diff": round(dlp, 4),
                                       "min_last_layer_cosine": round(cos, 5), "argmax_agree": f"{agree}/{k}"}
        log(f"  batched vs single readout: {report['batched_vs_single']}")
        # gross disagreement means padding or position handling is wrong, not numerical noise
        assert cos > 0.95 and dlp < 0.5, f"batched and single readouts disagree: {report['batched_vs_single']}"
    log(f"  smoke checks passed: {n_layers + 1} layers x {d_model}, finite, non-degenerate, "
        f"normalised, answers in each row's range")
    return report


def coverage(cfg, backend, questions) -> dict:
    """Token lengths over the WHOLE intended question set, from actually rendered prompts.

    Round-0 prompt lengths are measured for every question. The round-1 template's fixed
    overhead (instructions, panel headers, chat turns) is measured by rendering a real round-1
    prompt with empty responses. The worst round-1 prompt is then the longest question plus that
    overhead plus three full responses (the agent's own and two peers', each up to the cap); the
    worst readout adds the new response and the readout suffix. With a tokenizer-less backend
    (the stub) lengths are characters / 4 and marked approximate.
    """
    m = cfg.model.max_new_tokens
    count = getattr(backend, "count_tokens", None)
    exact = count is not None
    count = count or (lambda t: max(1, len(t) // 4))

    def r0(q):
        return P.round0_messages(persona=cfg.debate.personas[0], subject=q.subject, question=q.question,
                                 options=list(q.options), perm=make_permutation(q.n_options, q.question_id, 0),
                                 ask_confidence=cfg.debate.ask_verbalized_confidence)

    lens = [count(backend.render(r0(q))) for q in questions]
    base = r0(questions[int(np.argmax(lens))])
    r1 = P.round1_messages(base=base, own_response="", peer_responses=["", ""],
                           ask_confidence=cfg.debate.ask_verbalized_confidence)
    overhead = count(backend.render(r1)) - count(backend.render(base))
    worst_prompt = max(lens) + overhead + 3 * m
    return {
        "exact_tokens": exact, "n_questions": len(questions), "token_cap": m,
        "round0_prompt_tokens": {"p50": int(np.percentile(lens, 50)), "p95": int(np.percentile(lens, 95)),
                                 "max": int(max(lens))},
        "round1_template_overhead": int(overhead),
        "worst_round1_prompt_tokens": int(worst_prompt),
        "worst_readout_tokens": int(worst_prompt + m + count(P.READOUT_SUFFIX)),
    }


def gpu_identity() -> dict:
    out = {}
    try:
        import torch

        out["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except Exception:
        pass
    try:
        import subprocess

        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out["nvidia_driver"] = r.stdout.strip()
    except Exception:
        pass
    return out


class Ledger:
    """Append-only budget ledger, one JSON line per event, kept beside the job on the output
    drive -- so model loading, stress passes, failed work and every session's wall time are
    accounted for even if the runtime disappears. Colab idle time outside this process is not
    visible here; compare with the compute-unit balance."""

    def __init__(self, path: Path, t0: float):
        self.path, self.t0 = path, t0

    def __call__(self, event: str, **kw) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event,
               "session_seconds": round(time.time() - self.t0, 1), **kw}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--out-root", required=True, help="where runs are written (a Drive path on Colab)")
    ap.add_argument("--name", required=True, help="the job name; prefix for every run directory")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="mmlu-pro")
    ap.add_argument("--pool", choices=["pilot", "main"], default="main",
                    help="pilot: the first --pilot-size questions of the fixed order; main: the rest")
    ap.add_argument("--pilot-size", type=int, default=120)
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--conditions", nargs="+", default=["self_reconsider", "debate", "misleading_one"],
                    choices=sorted(CONDITION_SPECS))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--set", nargs="*", help="dotted config overrides, e.g. model.batch_size=6")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop starting chunks once total wall time across all sessions reaches this")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="resume a job created at a different commit (recorded in the job history)")
    ap.add_argument("--skip-hash-check", action="store_true", help="skip shard checksums when merging")
    ap.add_argument("--smoke", action="store_true", help="2 questions, check everything, report, stop")
    ap.add_argument("--stress-batches", type=int, nargs="+", default=[4, 6, 8],
                    help="generation and readout batch sizes to try in the worst-case memory pass")
    ap.add_argument("--skip-stress", action="store_true", help="never run the memory stress pass")
    ap.add_argument("--parent", help="child job: take round 0 from this job's saved caches")
    ap.add_argument("--subset-n", type=int, help="child job: the first N questions of the parent")
    args = ap.parse_args(argv)
    t_process = time.time()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(*args.config, **parse_overrides(args.set))
    cfg.out_dir = str(out_root)
    ds_name, ds_split = DATASETS[args.dataset]
    cfg.data.dataset, cfg.data.split = ds_name, ds_split
    name, n_total, chunk = args.name, args.n_questions, args.chunk

    if args.smoke:
        name = f"smoke-{args.name}"
        for d in out_root.glob(f"{name}-*"):  # a smoke run always starts clean
            shutil.rmtree(d) if d.is_dir() else d.unlink()
    ledger = Ledger(out_root / f"{name}-ledger.jsonl", t_process)

    parent = None
    if args.parent:
        if args.smoke:
            raise SystemExit("--smoke and --parent cannot be combined")
        parent = json.loads((out_root / f"{args.parent}-job.json").read_text())
        # A child's configuration IS its parent's: rebuilt from the parent's recorded identity,
        # with only execution settings (batch sizes) taken from this command line.
        from mad.config import config_from_dict

        pcfg = json.loads(json.dumps(parent["identity"]["config"]))
        pcfg["model"]["batch_size"] = cfg.model.batch_size
        pcfg["model"]["readout_batch_size"] = cfg.model.readout_batch_size
        pcfg["run_id"], pcfg["out_dir"] = cfg.run_id, cfg.out_dir
        pcfg["data"]["n_questions"] = cfg.data.n_questions
        cfg = config_from_dict(pcfg)
        # the dataset and pool are the parent's, whatever the command line says
        args.dataset = parent["identity"]["dataset"]["key"]
        args.pool = parent["identity"]["pool"]["name"]
        args.pilot_size = parent["identity"]["pool"]["pilot_size"]
        cfg.data.dataset, cfg.data.split = DATASETS[args.dataset]
        intended, chunk = _child_questions(out_root, parent, args.subset_n), parent["identity"]["chunk_size"]
        n_total = len(intended)
    else:
        pool = load_pool(args.dataset, cfg)
        master = master_order(pool, cfg.data.sample_seed)
        intended = select(master, args.pool, n_total, args.pilot_size)
    questions = intended[:2] if args.smoke else intended
    if args.smoke:
        n_total, chunk = 2, 2
    cfg.data.n_questions = n_total  # the resolved config records what actually runs

    log(f"=== {name}: {len(questions)} {args.pool}-pool questions x {len(args.conditions)} conditions, "
        f"dataset={args.dataset}, model={cfg.model.repo_id}, tokens={cfg.model.max_new_tokens}, "
        f"batch gen/readout={cfg.model.batch_size}/{cfg.model.readout_batch_size or cfg.model.batch_size}"
        + (f", child of {args.parent}" if parent else ""))
    ledger("session_start", argv=argv if argv is not None else sys.argv[1:])

    backend = None
    if cfg.model.backend != "stub":
        t = time.time()
        backend = build_backend(cfg)  # loaded once, reused by every chunk
        log(f"  model loaded in {time.time() - t:.0f}s: {backend.n_layers} layers, d={backend.d_model}")
        ledger("model_loaded", seconds=round(time.time() - t, 1))
        desc = backend.describe()
    else:
        desc = {"backend": "stub", "n_layers": cfg.stub.n_layers, "d_model": cfg.stub.d_model}

    conds_all = [{"name": c, **CONDITION_SPECS[c]} for c in args.conditions]
    if parent:
        identity = {**parent["identity"], "name": name,
                    "conditions": {c["name"]: CONDITION_SPECS[c["name"]] for c in conds_all},
                    "parent": {"name": args.parent, "identity_sha256": parent["identity_sha256"],
                               "subset_n": n_total}}
        mine = build_identity(name=name, cfg=cfg, dataset_key=args.dataset, pool=args.pool,
                              pilot_size=args.pilot_size, chunk_size=chunk, conditions={},
                              templates_hash=prompt_hashes(P.TEMPLATES), backend_desc=desc,
                              master_ids_sha="")
        for k in ("config", "prompt_template_hashes", "backend"):
            if mine[k] != parent["identity"][k]:
                raise SystemExit(f"a child job must use its parent's {k}; it differs")
        keep = {x.question_id for x in intended}
        chunks = [[q for q in members if q in keep] for members in parent["chunks"]]
    else:
        master_sha = hashlib.sha256(",".join(q.question_id for q in master).encode()).hexdigest()
        identity = build_identity(
            name=name, cfg=cfg, dataset_key=args.dataset, pool=args.pool, pilot_size=args.pilot_size,
            chunk_size=chunk, conditions={c["name"]: CONDITION_SPECS[c["name"]] for c in conds_all},
            templates_hash=prompt_hashes(P.TEMPLATES), backend_desc=desc, master_ids_sha=master_sha)
        chunks = None
    execution = {"batch_size": cfg.model.batch_size, "readout_batch_size": cfg.model.readout_batch_size,
                 "packages": package_versions(), **gpu_identity()}
    job = open_job(out_root, identity, questions, allow_code_change=args.allow_code_change,
                   execution=execution, chunks=chunks)
    log(f"  job {name}: {sum(1 for c in job['chunks'] if c)} chunks; history: "
        f"{[h['event'] for h in job['history']]}")

    cov = coverage(cfg, backend or _StubRender(cfg), intended)
    log(f"  token coverage over all {len(intended)} intended questions: {cov}")
    stress_path = out_root / f"{name}-stress.json"
    if backend is not None and hasattr(backend, "stress") and not args.skip_stress:
        prior = json.loads(stress_path.read_text()) if stress_path.exists() else {}
        same = (prior.get("identity_sha256") == job["identity_sha256"]
                and prior.get("gpu") == gpu_identity().get("gpu") and prior.get("passed"))
        if same:
            log(f"  stress already passed for this job on this GPU ({stress_path.name}); not repeating")
        else:
            log(f"  stress: worst-case generation and readout at batches {args.stress_batches} ...")
            t = time.time()
            rep = backend.stress(cov["worst_round1_prompt_tokens"], cfg.model.max_new_tokens,
                                 cov["worst_readout_tokens"], args.stress_batches, args.stress_batches)
            rb = cfg.model.readout_batch_size or cfg.model.batch_size
            passed = (any(r["ok"] and r["batch"] >= cfg.model.batch_size for r in rep["generation"])
                      and any(r["ok"] and r.get("finite_all_layers") and r["batch"] >= rb
                              for r in rep["readout"]))
            rec = {"identity_sha256": job["identity_sha256"], **gpu_identity(), "coverage": cov,
                   "configured": {"batch_size": cfg.model.batch_size, "readout_batch_size": rb},
                   "passed": passed, "seconds": round(time.time() - t, 1), **rep}
            stress_path.write_text(json.dumps(rec, indent=2))
            ledger("stress", seconds=rec["seconds"], passed=passed,
                   max_ok_generation_batch=rep["max_ok_generation_batch"],
                   max_ok_readout_batch=rep["max_ok_readout_batch"])
            log(f"  stress {'passed' if passed else 'FAILED'} in {rec['seconds']}s: largest safe batch "
                f"generation {rep['max_ok_generation_batch']}, readout {rep['max_ok_readout_batch']} "
                f"(report: {stress_path.name})")
            if not passed:
                log("stopping: the configured batch sizes do not fit the worst case. Lower them.")
                ledger("session_end", state="stress_failed")
                return 1

    status_path = out_root / f"{name}-status.json"
    prev = json.loads(status_path.read_text()) if status_path.exists() else {}
    hours_before = float(prev.get("elapsed_hours_total", 0.0))
    timings, failures = prev.get("timings", []), prev.get("failures", [])
    sessions = prev.get("sessions", 0) + 1
    t_run = t_process  # the cap counts model loading and the stress pass too
    state, consecutive, smoke_report, last = "running", 0, None, None

    def write_status():
        live = [j for j, members in enumerate(job["chunks"]) if members]
        done = sum(all(is_complete(out_root / f"{name}-{c}-{chunk_tag(j)}") for c in args.conditions)
                   for j in live)
        status_path.write_text(json.dumps({
            "name": name, "state": state, "chunks_done": done, "chunks_total": len(live),
            "questions_total": len(job["questions"]), "conditions": args.conditions,
            "sessions": sessions, "elapsed_hours_total": round(hours_before + (time.time() - t_run) / 3600, 3),
            "max_hours": args.max_hours, "timings": timings, "failures": failures,
            "diagnostics_last_chunk": diagnostics(last) if last else None,
        }, indent=2, default=str))

    by_id = {q.question_id: q for q in questions}
    for i, members in enumerate(job["chunks"]):
        if not members:  # a child job's chunk with none of its questions
            continue
        tag = chunk_tag(i)
        dirs = {c: out_root / f"{name}-{c}-{tag}" for c in args.conditions}
        if all(is_complete(d) for d in dirs.values()):
            for d in dirs.values():
                check_chunk_dir(d, job, i)
            continue
        used = hours_before + (time.time() - t_run) / 3600
        if args.max_hours is not None and used >= args.max_hours:
            state = "stopped_time_cap"
            log(f"time cap reached ({used:.2f} h of {args.max_hours}); stopping before {tag}")
            break
        part = [by_id[qid] for qid in members]
        conds = [{**c, "run_id": f"{name}-{c['name']}-{tag}"} for c in conds_all]
        t = time.time()
        gpu_stats(reset=True)
        try:
            parent_r0 = ({"dir": out_root / f"{args.parent}-round0-{tag}", "job": args.parent,
                          "question_ids": parent["chunks"][i]} if parent else None)
            res, r0 = run_chunk(cfg, part, conds, backend=backend, job=name, chunk=tag,
                                round0_dir=out_root / f"{name}-round0-{tag}", run_dirs=dirs,
                                extra_manifest={"job_identity_sha256": job["identity_sha256"]},
                                parent_round0=parent_r0)
        except Exception as exc:
            consecutive += 1
            failures.append({"chunk": tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "error": type(exc).__name__, "message": str(exc)[:500]})
            ledger("chunk_failed", chunk=tag, seconds=round(time.time() - t, 1),
                   error=type(exc).__name__)
            log(f"FAILED {tag}\n{traceback.format_exc()}")
            if args.smoke or is_oom(exc) or consecutive >= MAX_CONSECUTIVE_FAILURES:
                state = "failed"
                why = "out of memory" if is_oom(exc) else f"{consecutive} consecutive failures"
                log(f"stopping: {why}. Fix the cause, then re-run the same command to resume.")
                write_status()
                return 1
            write_status()
            continue
        consecutive = 0
        last = dirs[args.conditions[-1]]
        dt = time.time() - t
        stages = {"round0_s": (r0.timings or {}).get("round0_s", 0.0),
                  **{f"{c}/{k}": v for c, r in res.items() for k, v in (r.timings or {}).items()}}
        timings.append({"chunk": tag, "questions": len(part), "seconds": round(dt, 1),
                        "seconds_per_question": round(dt / len(part), 1),
                        "round0_reloaded": r0.loaded, "stages": stages, **gpu_stats()})
        ledger("chunk_done", chunk=tag, seconds=round(dt, 1), questions=len(part), stages=stages)
        log(f"  {tag}: {len(part)} questions in {dt:.0f}s ({dt / len(part):.0f} s/question)  {gpu_stats()}")
        if args.smoke:
            smoke_report = smoke_checks(res, r0, backend or _StubShape(cfg), cfg, len(part))
        write_status()
    if state == "running":
        complete = all(is_complete(out_root / f"{name}-{c}-{chunk_tag(j)}")
                       for c in args.conditions for j, members in enumerate(job["chunks"]) if members)
        state = "finished" if complete else "incomplete"
    write_status()

    for c in args.conditions:
        out = build_view(out_root, job, c, verify_hashes=not args.skip_hash_check)
        if out:
            log(f"VIEW {c}: {out}  {diagnostics(out)}")

    if args.smoke and timings:
        spq = timings[-1]["seconds_per_question"]
        hours = spq * args.n_questions / 3600
        (out_root / f"{name}-smoke-report.json").write_text(json.dumps(smoke_report, indent=2, default=str))
        log(f"SMOKE PASSED. {spq:.0f} s/question with {len(args.conditions)} conditions -> "
            f"~{hours:.1f} h for {args.n_questions} questions (an overestimate: 2 questions batch poorly). "
            f"{gpu_stats()}")
    ledger("session_end", state=state)
    log(f"{state}; failures this job: {len(failures)}")
    return 0 if state in ("finished", "stopped_time_cap") else 1


def _child_questions(out_root: Path, parent: dict, subset_n: int | None) -> list:
    """The first ``subset_n`` questions of a parent job, from its verified local snapshot."""
    from mad.data import Question
    from mad.job import question_sha256

    name = parent["identity"]["name"]
    snap = [json.loads(line) for line in
            (out_root / f"{name}-questions.jsonl").read_text().splitlines()]
    qs = [Question(**{**d, "options": tuple(d["options"])}) for d in snap]
    recorded = {q["id"]: q["sha256"] for q in parent["questions"]}
    for q in qs:
        if recorded.get(q.question_id) != question_sha256(q):
            raise SystemExit(f"{name}-questions.jsonl does not match the parent job's hashes")
    order = [q["id"] for q in parent["questions"]]
    by_id = {q.question_id: q for q in qs}
    return [by_id[i] for i in order[: subset_n or len(order)]]


class _StubRender:
    """Rendering for coverage when the stub backend is built inside each chunk."""

    def __init__(self, cfg):
        from mad.backends.stub import StubBackend

        self._b = StubBackend(cfg, {})

    def render(self, msgs):
        return self._b.render(msgs)


class _StubShape:
    """Shape facts for smoke checks when the stub is built inside each chunk."""

    def __init__(self, cfg):
        self.n_layers, self.d_model = cfg.stub.n_layers, cfg.stub.d_model


if __name__ == "__main__":
    raise SystemExit(main())
