"""Generation tasks for Experiments A/B and the judge confirmation (vLLM; run with the runtime's
separate vLLM environment). One model per process; output chunks are resumable and hash-recorded.

    PY=/content/vllm_env/bin/python
    $PY scripts/gen_run.py --task bank --mode gemma --out-root /content/expab --job expab --split dev
    $PY scripts/gen_run.py --task choose --arm A2_5 --bank-root /content/expab --bank-job expab ...
    $PY scripts/gen_run.py --task b --mode qwen_t --arms debate alone --split dev --n-panels 100 ...
    $PY scripts/gen_run.py --task judge --out-root /content/judgec --job judgec
    # CPU check: --engine stub

Parts (chunk directory prefixes): bank_<mode>, choose_<arm>, b_<mode>_<arm>, judge_traces, judge_own.
The job record lists every part's expected chunk count; a part's count is written when its task
starts (choosers and B depend on the bank). Chunks already complete are skipped (resume).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.gen import archive as A  # noqa: E402
from mad.gen import engine as E  # noqa: E402
from mad.gen import prompts_ab as PA  # noqa: E402
from mad.gen import tasks as T  # noqa: E402
from mad.provenance import is_complete  # noqa: E402

MODEL_LIMIT = {"Qwen/Qwen3-8B": 40960, "google/gemma-3-12b-it": 131072}
JUDGE_EVENTS = ROOT / "artifacts/revision_gate/judge/confirm_events.json"


def chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def code_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def update_job(out_root: Path, job: str, part: str, n_chunks: int, info: dict):
    p = out_root / f"{job}-job.json"
    rec = json.loads(p.read_text()) if p.exists() else {"job": job, "parts": {}, "part_info": {}, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    rec["parts"][part] = n_chunks
    rec["part_info"][part] = info
    A.write_job(out_root, job, rec)


_ENGINES: dict = {}


def make_engine(kind: str, mode: E.Mode, prompt_lens: list[int]):
    """One engine per model per process (several parts reuse it). The context limit covers the
    longest prompt plus the generation budget, at least 32k (capped at the model's limit), so a
    later part with longer prompts rarely forces a reload."""
    need = (max(prompt_lens) if prompt_lens else 0) + E.budget(mode) + 64
    limit = MODEL_LIMIT.get(mode.repo_id, 32768)
    mml = min(limit, max(32768, ((need + 1023) // 1024) * 1024))
    key = (kind, mode.repo_id)
    eng = _ENGINES.get(key)
    if eng is not None and eng.max_model_len >= min(need, limit):
        return eng
    _ENGINES.pop(key, None)
    eng = E.StubEngine(max_model_len=mml) if kind == "stub" else E.VLLMEngine(E.MODES[mode.name], max_model_len=mml)
    _ENGINES[key] = eng
    return eng


def prompt_lengths(kind: str, mode: E.Mode, reqs) -> list[int]:
    if kind == "stub":
        tok = E.StubTokenizer()
    else:
        from transformers import AutoTokenizer  # noqa: PLC0415

        tok = AutoTokenizer.from_pretrained(mode.repo_id)
    eng = type("L", (), {"tokenizer": tok})()
    return [len(E.render_ids(eng, mode, r)) for r in reqs]


def run_part(args, out_root, job, part, mode, batches, info, answer=True, post=None):
    """batches: list of request lists (one chunk each)."""
    update_job(out_root, job, part, len(batches), {**info, "mode": mode.name, "repo_id": mode.repo_id,
                                                   "template_hashes": PA.TEMPLATE_HASHES, "code": code_commit()})
    todo = []
    for i in range(len(batches)):
        d = A.chunk_dir(out_root, job, part, i)
        if not is_complete(d):
            todo.append(i)
            continue
        have = json.loads((d / "manifest.json").read_text()).get("question_ids")
        if have != sorted({r.question_id for r in batches[i]}):  # a different selection under the same name
            raise SystemExit(f"{d.name} is complete but holds other questions; refusing to resume into it")
    if not todo:
        return
    lens = prompt_lengths(args.engine, mode, [r for i in todo for r in batches[i]])
    eng = make_engine(args.engine, mode, lens)
    t0 = time.time()
    for n, i in enumerate(todo):
        rows, calls = E.run_requests(eng, mode, batches[i], answer=answer, call_prefix=f"{part}-c{i:03d}-")
        if post:
            rows = post(rows)
        A.write_chunk(out_root, job, part, i, rows, calls, {"question_ids": sorted({r.question_id for r in batches[i]}),
                                                            "max_model_len": eng.max_model_len})
        A.write_status(out_root, job, {"part": part, "chunks_done": len(batches) - len(todo) + n + 1,
                                       "chunks_total": len(batches), "seconds": round(time.time() - t0, 1),
                                       "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        print(f"[{time.strftime('%H:%M:%S')}] {part} chunk {i}: {len(rows)} rows, "
              f"{sum(c['seconds'] for c in calls):.1f}s", flush=True)


def part_name(base: str, split: str, tag: str | None) -> str:
    """``base`` for the whole set; ``base__<split>[_<tag>]`` otherwise, so a split (or a pilot)
    run separately never shares chunk names with another selection."""
    if split == "all" and not tag:
        return base
    return f"{base}__{split}" + (f"_{tag}" if tag else "")


def select(qs, assign, split, limit):
    q = [x for x in qs if split == "all" or assign[x.question_id] == split]
    return q[:limit] if limit else q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["bank", "choose", "b", "judge", "sanity"])
    ap.add_argument("--mode", choices=sorted(E.MODES))
    ap.add_argument("--arm", nargs="+", choices=sorted(T.ARMS))
    ap.add_argument("--arms", nargs="+", choices=T.B_ARMS)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--split", default="all", choices=["dev", "heldout", "all"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions of the split (smoke)")
    ap.add_argument("--samples", type=int, default=None, help="bank: samples per question (default: the plan's)")
    ap.add_argument("--n-panels", type=int, default=None)
    ap.add_argument("--panel-seed", type=int, default=20261122)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--bank-root", default=None)
    ap.add_argument("--bank-job", default=None)
    ap.add_argument("--chunk", type=int, default=50, help="questions per chunk")
    ap.add_argument("--engine", default="vllm", choices=["vllm", "stub"])
    ap.add_argument("--tag", default=None, help="suffix for the part names, e.g. pilot or smoke")
    a = ap.parse_args(argv)
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    qs, assign, manifest = T.load_questions(a.plan)
    qsel = select(qs, assign, a.split, a.limit)
    plan_info = {"question_ids_sha256": manifest["question_ids_sha256"], "split": a.split, "limit": a.limit}
    pid = out / f"{a.job}.pid"
    if not pid.exists():
        pid.write_text(str(os.getpid()))
    if a.task == "bank":
        mode = E.MODES[a.mode]
        batches = [T.bank_requests(c, a.mode, a.samples) for c in chunks(qsel, a.chunk)]
        run_part(a, out, a.job, part_name(f"bank_{a.mode}", a.split, a.tag), mode, batches, {**plan_info, "samples": a.samples or T.BANK[a.mode]})
    elif a.task == "choose":
        bank = T.bank_index(A.read_rows(a.bank_root or out, a.bank_job or a.job, "bank_*"))  # bank parts only

        def post(rows):
            for r in rows:
                order = json.loads(r["meta_order_answers"])
                pos = T.parse_pick(r["text"], r["meta_m"])
                r["pick_position"], r["pick_answer"] = pos, (order[pos - 1] if pos > 0 else -1)
            return rows
        for arm in a.arm:  # arms with the same chooser share the loaded model
            chooser, spec = T.ARMS[arm]
            reqs = []
            for q in qsel:
                pool = T.pool_rows(bank, q.question_id, spec)
                if T.disagreeing(pool):
                    reqs.append(T.chooser_requests(q, arm, T.candidates(pool, q.question_id, arm)))
            batches = [[r for g in c for r in g] for c in chunks(reqs, a.chunk)]
            run_part(a, out, a.job, part_name(f"choose_{arm}", a.split, a.tag), E.MODES[chooser], batches,
                     {**plan_info, "arm": arm, "pool": spec, "questions_with_disagreement": len(reqs)}, post=post)
    elif a.task == "b":
        bank = T.bank_index(A.read_rows(a.bank_root or out, a.bank_job or a.job, "bank_*"))  # bank parts only
        ids = T.select_panels(bank, qs, assign, a.split, a.mode, a.n_panels, a.panel_seed)
        byq = {q.question_id: q for q in qs}
        for arm in a.arms:
            reqs = [T.b_requests(byq[i], a.mode, T.panel(bank, a.mode, i), arm) for i in ids]
            batches = [[r for g in c for r in g] for c in chunks(reqs, a.chunk)]
            run_part(a, out, a.job, part_name(f"b_{a.mode}_{arm}", a.split, a.tag), E.MODES[a.mode], batches,
                     {**plan_info, "arm": arm, "panels": len(ids), "panel_seed": a.panel_seed})
    elif a.task == "sanity":
        # vLLM on questions that already have HF-generated answers under the same protocol (the judge
        # run's own answers: same system prompt, persona, 250 words, question block): a like-for-like check
        from mad import judge as J  # noqa: PLC0415

        jq = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/questions.jsonl")][: a.limit or 20]
        n = a.samples or 3
        reqs = [E.Request(f"{q['question_id']}|{s}", q["question_id"], s, f"sanity/{a.mode}", q["n_options"],
                          messages=J.own_messages(q["question_block"])) for q in jq for s in range(n)]
        run_part(a, out, a.job, part_name(f"sanity_{a.mode}", "all", a.tag), E.MODES[a.mode], chunks(reqs, a.chunk * n),
                 {"questions": len(jq), "samples": n, "source": "artifacts/revision_gate/judge/questions.jsonl"})
    elif a.task == "judge":
        rec = json.loads(JUDGE_EVENTS.read_text())
        ev = {json.loads(x)["event_id"]: json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/events.jsonl")}
        events = [ev[e] for e in rec["event_ids"]][: a.limit]
        questions = [json.loads(x) for x in open(ROOT / "artifacts/revision_gate/judge/questions.jsonl")][: a.limit]
        reqs = T.judge_confirm_requests(events, questions)
        judge = [r for r in reqs if r.condition.endswith("/judge")]
        own = [r for r in reqs if r.condition.endswith("/own")]
        mode = E.MODES["qwen_t_judge"]
        info = {"events_sha256": rec["sha256"], "events": len(events), "questions": len(questions)}
        run_part(a, out, a.job, part_name("judge_traces", "all", a.tag), mode, chunks(judge, a.chunk * 2), info, answer=False)
        run_part(a, out, a.job, part_name("judge_own", "all", a.tag), mode, chunks(own, a.chunk * 3), info, answer=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
