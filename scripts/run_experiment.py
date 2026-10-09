"""Resumable multi-condition experiment driver. Same code on the laptop CPU or a Colab GPU.

    python scripts/run_experiment.py \\
        --config configs/base.yaml configs/colab_12b.yaml \\
        --out-root /content/drive/MyDrive/mas_runs --name pilot12b \\
        --dataset mmlu-pro --n-questions 60 \\
        --conditions self_reconsider debate misleading_one --chunk 10

Add --smoke to run 2 questions through every condition, assert that the outputs have the shapes
the analysis expects, report time and memory, project the full run, and stop. Run it first: if
something is wrong with the 12B load path it fails in minutes instead of after an hour.

Design points:
  * Work is split into chunks. Each chunk writes a completion record for every condition and is
    skipped on re-run, so a Colab disconnect costs one chunk and re-running resumes.
  * The model is loaded ONCE and reused across chunks. Gemma-3-12B is 26 GB of weights; reloading
    per chunk would cost minutes each time.
  * Round 0 is generated once per chunk and shared by every condition (see mad.multi).
  * A status file is rewritten after every chunk, so progress can be read from Drive.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from mad.backends.base import build_backend  # noqa: E402
from mad.config import load_config  # noqa: E402
from mad.data import (  # noqa: E402
    load_mc_dataset,
    load_mmlu_pro,
    stratified_sample,
    synthetic_questions,
)
from mad.debate import ROWS_FILE  # noqa: E402
from mad.multi import run_conditions  # noqa: E402
from mad.provenance import is_complete  # noqa: E402
from mad.store import ActivationStore, merge_runs  # noqa: E402

CONDITION_SPECS = {
    "self_reconsider": {"kind": "self_reconsider"},
    "debate": {"kind": "debate", "peer_confidence": False},
    "debate_conf": {"kind": "debate_conf", "peer_confidence": True},
    "misleading_one": {"kind": "misleading_one"},
    "adversarial": {"kind": "adversarial"},
}
DATASETS = ("mmlu-pro", "arc-challenge", "arc-easy", "synthetic")


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


def load_questions(dataset: str, n: int, cfg):
    if dataset == "mmlu-pro":
        pool = load_mmlu_pro(cfg)
    elif dataset in ("arc-challenge", "arc-easy"):
        name = "ARC-Challenge" if dataset == "arc-challenge" else "ARC-Easy"
        pool = load_mc_dataset("allenai/ai2_arc", name, "test")
    elif dataset == "synthetic":
        pool = synthetic_questions(max(n * 2, 50), n_options=4, seed=cfg.data.sample_seed)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    return stratified_sample(pool, n, cfg.data.sample_seed)


def gpu_stats() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "device": torch.cuda.get_device_name(0),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            }
    except Exception:
        pass
    return {}


def diagnostics(run_dir: Path, max_tok: int) -> dict:
    """Completion-length and parse-rate diagnostics -- the evidence for the token budget."""
    r = pd.read_parquet(run_dir / ROWS_FILE)
    out = {}
    for phase in ("pre", "post"):
        s = r[r.phase == phase]
        if len(s):
            out[phase] = {
                "parse_ok": round(float(s.freetext_parse_ok.mean()), 4),
                "hit_cap": round(float((s.n_completion_tokens >= max_tok - 2).mean()), 4),
                "median_tokens": float(s.n_completion_tokens.median()),
            }
    return out


def chunk_dirs(out_root: Path, name: str, conds: list[str], tag: str) -> dict[str, Path]:
    return {c: out_root / f"{name}-{c}-{tag}" for c in conds}


def merge_condition(out_root: Path, name: str, cond: str, dirs: list[Path]) -> Path | None:
    """(Re)merge every completed chunk for one condition. Chunk dirs are the source of truth."""
    done = [d for d in dirs if is_complete(d)]
    if not done:
        return None
    out = out_root / f"{name}-{cond}"
    if is_complete(out):
        merged = json.loads((out / "completion.json").read_text()).get("merged_from", [])
        if len(merged) == len(done):
            return out
    if out.exists():
        shutil.rmtree(out)  # a stale or partial merge; rebuild from chunks
    merge_runs(done, out, ROWS_FILE)
    return out


def smoke_checks(res: dict, n_layers: int, d_model: int, n_q: int, n_agents: int) -> None:
    """Assert the outputs have exactly the shape the analysis depends on."""
    for cond, r in res.items():
        phases = set(r.rows.phase)
        assert phases == {"pre", "peer_exposure", "post"}, f"{cond}: phases {phases}"
        assert len(r.rows) == n_q * n_agents * 3, f"{cond}: {len(r.rows)} rows"
        store = ActivationStore(r.run_dir)
        x = store.load_layer(store.n_layers)
        assert store.n_layers == n_layers, f"{cond}: {store.n_layers} layers stored, expected {n_layers}"
        assert x.shape == (len(r.rows), d_model), f"{cond}: layer shape {x.shape}"
        assert r.rows.readout_answer.between(0, r.rows.n_options.max() - 1).all()
        assert (r.rows.readout_confidence > 0).all()
    log(f"  shape checks passed: {n_layers + 1} layers x {d_model} per state, "
        f"three states per agent, answers in range")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--out-root", required=True, help="where runs are written (a Drive path on Colab)")
    ap.add_argument("--name", required=True, help="prefix for every run directory")
    ap.add_argument("--dataset", choices=DATASETS, default="mmlu-pro")
    ap.add_argument("--n-questions", type=int, default=60)
    ap.add_argument("--conditions", nargs="+", default=["self_reconsider", "debate", "misleading_one"],
                    choices=sorted(CONDITION_SPECS))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--set", nargs="*", help="dotted config overrides, e.g. model.batch_size=6")
    ap.add_argument("--smoke", action="store_true", help="2 questions, assert shapes, report, stop")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(*args.config, **parse_overrides(args.set))
    cfg.out_dir = str(out_root)
    n_total = args.n_questions
    name = args.name

    if args.smoke:
        name = f"smoke-{args.name}"
        n_total, args.chunk = 2, 2
        for d in out_root.glob(f"{name}-*"):  # a smoke run always starts clean
            shutil.rmtree(d) if d.is_dir() else d.unlink()

    log(f"=== {name}: {n_total} questions x {len(args.conditions)} conditions, "
        f"dataset={args.dataset}, model={cfg.model.repo_id}, tokens={cfg.model.max_new_tokens}, "
        f"batch={cfg.model.batch_size}")
    questions = load_questions(args.dataset, n_total, cfg)
    log(f"  loaded {len(questions)} questions across {len({q.subject for q in questions})} subjects")

    backend = None
    if cfg.model.backend != "stub":
        t = time.time()
        backend = build_backend(cfg)  # loaded once, reused by every chunk
        log(f"  model loaded in {time.time() - t:.0f}s: {backend.n_layers} layers, d={backend.d_model}")
        if hasattr(backend, "estimated_readout_gb"):
            seq = 900 + 3 * cfg.model.max_new_tokens  # question + own answer + two peer answers
            est = backend.estimated_readout_gb(cfg.model.batch_size, seq)
            kv = backend.estimated_kv_cache_gb(cfg.model.batch_size, seq)
            g = gpu_stats()
            log(f"  worst-case activation memory at batch {cfg.model.batch_size}, ~{seq} tokens: "
                f"readout {est:.1f} GB, KV cache {kv:.1f} GB; weights resident {g.get('peak_gb', 0):.1f} GB "
                f"of {g.get('total_gb', 0):.1f} GB")

    conds_all = [{"name": c, **CONDITION_SPECS[c]} for c in args.conditions]
    parts = [questions[i : i + args.chunk] for i in range(0, len(questions), args.chunk)]
    per_cond: dict[str, list[Path]] = {c: [] for c in args.conditions}
    timings, failures = [], []
    status_path = out_root / f"{name}-status.json"
    t_run = time.time()

    for i, part in enumerate(parts):
        tag = f"c{i:03d}"
        dirs = chunk_dirs(out_root, name, args.conditions, tag)
        for c, d in dirs.items():
            per_cond[c].append(d)
        if all(is_complete(d) for d in dirs.values()):
            continue
        conds = [{**c, "run_id": f"{name}-{c['name']}-{tag}"} for c in conds_all]
        t = time.time()
        try:
            res = run_conditions(cfg, part, conds, backend=backend)
        except Exception:
            failures.append(tag)
            log(f"FAILED {tag}\n{traceback.format_exc()}")
            if args.smoke:
                return 1
            continue
        dt = time.time() - t
        timings.append({"chunk": tag, "questions": len(part), "seconds": round(dt, 1),
                        "seconds_per_question": round(dt / len(part), 1), **gpu_stats()})
        log(f"  {tag}: {len(part)} questions in {dt:.0f}s ({dt / len(part):.0f} s/question)  {gpu_stats()}")
        if args.smoke:
            n_l = backend.n_layers if backend else cfg.stub.n_layers
            d_m = backend.d_model if backend else cfg.stub.d_model
            smoke_checks(res, n_l, d_m, len(part), cfg.debate.n_agents)
        done = sum(all(is_complete(d) for d in chunk_dirs(out_root, name, args.conditions, f"c{j:03d}").values())
                   for j in range(len(parts)))
        status_path.write_text(json.dumps({
            "name": name, "chunks_done": done, "chunks_total": len(parts),
            "questions_total": len(questions), "conditions": args.conditions,
            "elapsed_hours": round((time.time() - t_run) / 3600, 2),
            "timings": timings, "failures": failures,
            "diagnostics_last_condition_last_chunk": diagnostics(dirs[args.conditions[-1]], cfg.model.max_new_tokens),
        }, indent=2, default=str))

    for c, dirs in per_cond.items():
        out = merge_condition(out_root, name, c, dirs)
        if out:
            log(f"MERGED {c}: {out}  {diagnostics(out, cfg.model.max_new_tokens)}")

    if args.smoke and timings:
        spq = timings[-1]["seconds_per_question"]
        hours = spq * args.n_questions / 3600
        log(f"SMOKE PASSED. {spq:.0f} s/question with {len(args.conditions)} conditions -> "
            f"~{hours:.1f} h for {args.n_questions} questions (an overestimate: 2 questions batch poorly). "
            f"Peak GPU {gpu_stats().get('peak_gb', 'n/a')} GB.")
    log(f"finished; failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
