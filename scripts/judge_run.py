"""GPU runtime for the cross-model judge and the own-answer baseline (revision gate, section 5).

    python scripts/judge_run.py --model qwen3 --task judge own --out /content/judge \\
        --events artifacts/revision_gate/judge/events.jsonl --questions artifacts/revision_gate/judge/questions.jsonl
    # smoke: --events-limit 10 (own answers then cover only those events' questions)

Models (``MODELS``): gemma (Gemma-3-12B-it, same-model control), qwen3 (Qwen3-8B, thinking off),
qwen3_think (thinking on: judge and own answer only on the thinking subsample, the trace capped
at --think-cap tokens), llama (Llama-3.1-8B-Instruct). No activations are stored: outputs are
answers, readouts, token counts and timings. Sampling: own answers use the main collection's
protocol (temperature 0.7, top_p 0.95) with top_k 64 and repetition_penalty 1.0 passed
explicitly (Gemma's effective settings); thinking traces use Qwen3's recommended thinking
settings (0.6 / 0.95 / 20). Judge readouts are deterministic (no sampling).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mad import judge as J  # noqa: E402
from mad.parsing import parse_answer, parse_confidence  # noqa: E402
from mad.prompts import READOUT_SUFFIX  # noqa: E402

MODELS = {
    "gemma": {"repo_id": "google/gemma-3-12b-it", "kwargs": None, "thinking": False, "batch": 8, "budget": 20000},
    "qwen3": {"repo_id": "Qwen/Qwen3-8B", "kwargs": {"enable_thinking": False}, "thinking": False, "batch": 16,
              "budget": 40000},
    "qwen3_think": {"repo_id": "Qwen/Qwen3-8B", "kwargs": {"enable_thinking": True}, "thinking": True, "batch": 16,
                    "budget": 60000},
    "llama": {"repo_id": "meta-llama/Llama-3.1-8B-Instruct", "kwargs": None, "thinking": False, "batch": 16,
              "budget": 40000},
    # CPU test only (tests/test_judge.py): code paths, not science
    "tiny": {"repo_id": "google/gemma-3-270m-it", "kwargs": None, "thinking": False, "batch": 4, "budget": None,
             "configs": ["configs/base.yaml", "configs/smoke_tiny.yaml"], "max_new": 32},
}
THINK_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def load_backend(name: str):
    from mad.backends.base import build_backend
    from mad.config import load_config

    m = MODELS[name]
    cfg = load_config(*m.get("configs", ["configs/base.yaml", "configs/colab_12b.yaml"]), **{
        "model.repo_id": m["repo_id"], "model.readout_head": "fp32_head", "model.max_new_tokens": m.get("max_new", 1280),
        "model.top_k": 64, "model.repetition_penalty": 1.0, "model.batch_size": m["batch"],
        "model.gen_batch_tokens": m["budget"], "model.readout_batch_size": 1, "debate.brevity_words": 250})
    cfg.model.chat_template_kwargs = m["kwargs"]
    return build_backend(cfg), cfg


def ids_of(backend, text: str) -> list[int]:
    return backend.tokenizer(text, add_special_tokens=False)["input_ids"]


def generate_ids(backend, seqs: list[list[int]], max_new: int, sampling: dict, stop_ids: list[int], seed: int,
                 batch: int) -> list[list[int]]:
    """Sampled continuations of token-id prompts, left-padded batches; trailing pads removed."""
    torch, tok = backend.torch, backend.tokenizer
    pad = tok.pad_token_id
    out = []
    for s in range(0, len(seqs), batch):
        chunk = seqs[s:s + batch]
        w = max(len(x) for x in chunk)
        ids = torch.tensor([[pad] * (w - len(x)) + x for x in chunk], device=backend.device)
        mask = torch.tensor([[0] * (w - len(x)) + [1] * len(x) for x in chunk], device=backend.device)
        torch.manual_seed(seed + s)
        with torch.no_grad():
            g = backend.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=True,
                                       eos_token_id=stop_ids, pad_token_id=pad, **sampling)
        for i in range(len(chunk)):
            new = g[i, w:].tolist()
            while new and new[-1] == pad:
                new.pop()
            out.append(new)
    return out


def think_then(backend, prompts_ids, cap, seed, batch):
    """Thinking traces (stopped at </think> or ``cap`` tokens), each closed, as id lists."""
    tok = backend.tokenizer
    end = tok.convert_tokens_to_ids("</think>")
    traces = generate_ids(backend, prompts_ids, cap, THINK_SAMPLING, [end, *backend._stop_ids], seed, batch)
    closed, out = [], []
    for t in traces:
        if end in t:
            t = t[: t.index(end) + 1] + ids_of(backend, "\n\n")
            closed.append(True)
        else:
            t = t + ids_of(backend, J.THINK_CLOSE)
            closed.append(False)
        out.append(t)
    return out, closed


def run_judge(backend, name, events, out_dir, think_cap, batch, log):
    tok = backend.tokenizer
    choice = [ids_of(backend, c)[0] for c in J.CHOICES]
    prefill = ids_of(backend, J.JUDGE_PREFILL)
    thinking = MODELS[name]["thinking"]
    rows, t0 = [], time.time()
    items = [(e, order, msgs) for e in events for order, msgs in J.both_orders(e).items()]
    prompt_ids = [ids_of(backend, backend.render(m)) for _, _, m in items]
    traces, closed = ([], [])
    if thinking:
        traces, closed = think_then(backend, prompt_ids, think_cap, 20261031, batch)
    t_gen = time.time() - t0
    for i, (e, order, _) in enumerate(items):
        seq = prompt_ids[i] + (traces[i] if thinking else []) + prefill
        lp = backend.choice_logprobs_ids([seq], choice)[0]
        p = np.exp(lp)
        rows.append({"event_id": e["event_id"], "order": order, "p1": float(p[0]), "p2": float(p[1]),
                     "prompt_tokens": len(prompt_ids[i]), "thinking_tokens": len(traces[i]) if thinking else 0,
                     "thinking_closed": closed[i] if thinking else None})
    secs = time.time() - t0
    (out_dir / f"{name}_judge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log(f"{name} judge: {len(events)} events x 2 orders in {secs:.0f}s (thinking generation {t_gen:.0f}s)")
    return {"events": len(events), "passes": len(rows), "seconds": round(secs, 1), "thinking_generation_seconds":
            round(t_gen, 1), "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in rows])),
            "thinking_tokens_mean": float(np.mean([r["thinking_tokens"] for r in rows])) if thinking else 0,
            "thinking_closed_share": float(np.mean(closed)) if thinking else None}


def run_own(backend, name, questions, samples, out_dir, think_cap, batch, log):
    tok = backend.tokenizer
    thinking = MODELS[name]["thinking"]
    suffix = ids_of(backend, READOUT_SUFFIX)
    stop = sorted(backend._stop_ids)
    items = [(q, s) for q in questions for s in range(samples)]
    prompt_ids = [ids_of(backend, backend.render(J.own_messages(q["question_block"]))) for q, _ in items]
    t0 = time.time()
    if thinking:
        traces, closed = think_then(backend, prompt_ids, think_cap, 20261032, batch)
        cont = [p + t for p, t in zip(prompt_ids, traces, strict=True)]
        sampling = {"temperature": 0.7, "top_p": 0.95, **backend.sampling_overrides()}
        answers = generate_ids(backend, cont, 1280, sampling, stop, 20261033, batch)
        gen = [t + a for t, a in zip(traces, answers, strict=True)]
    else:
        prompts = [backend.render(J.own_messages(q["question_block"])) for q, _ in items]
        g = backend.generate(prompts, seed=20261034)
        gen = [list(x.token_ids or []) for x in g]
        answers, closed = gen, [None] * len(gen)
    t_gen = time.time() - t0
    rows = []
    for i, (q, s) in enumerate(items):
        a_ids = list(answers[i])
        while a_ids and a_ids[-1] in backend._stop_ids:
            a_ids.pop()
        text = tok.decode(a_ids, skip_special_tokens=True)
        visible = J.after_thinking(text) if thinking else text
        idx, ok = parse_answer(visible, q["n_options"])
        body = (prompt_ids[i] + (gen[i][: len(gen[i]) - len(answers[i])] if thinking else []) + a_ids)
        lp = backend.letter_logprobs_ids([body + suffix], [q["n_options"]], batch_size=1, head="fp32_head")[0]
        rows.append({"question_id": q["question_id"], "sample": s, "answer": int(idx) if ok else -1, "parsed": bool(ok),
                     "confidence": parse_confidence(visible), "readout_logprobs": [round(float(x), 5) for x in lp],
                     "prompt_tokens": len(prompt_ids[i]), "generated_tokens": len(gen[i]),
                     "answer_tokens": len(answers[i]), "capped": len(answers[i]) >= 1280,
                     "thinking_closed": closed[i] if thinking else None, "text": visible[-1500:]})
    secs = time.time() - t0
    (out_dir / f"{name}_own.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    log(f"{name} own answers: {len(rows)} responses in {secs:.0f}s (generation {t_gen:.0f}s)")
    return {"responses": len(rows), "seconds": round(secs, 1), "generation_seconds": round(t_gen, 1),
            "generated_tokens_mean": float(np.mean([r["generated_tokens"] for r in rows])),
            "parse_rate": float(np.mean([r["parsed"] for r in rows])), "capped": int(sum(r["capped"] for r in rows))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--task", nargs="+", default=["judge", "own"], choices=["judge", "own"])
    ap.add_argument("--events", default="artifacts/revision_gate/judge/events.jsonl")
    ap.add_argument("--questions", default="artifacts/revision_gate/judge/questions.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--events-limit", type=int, default=None)
    ap.add_argument("--questions-limit", type=int, default=None,
                    help="own answers on the first N questions (default: the limited events' questions, or all)")
    ap.add_argument("--own-samples", type=int, default=None, help="default: 1 thinking, 3 otherwise")
    ap.add_argument("--think-cap", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = MODELS[a.model]
    events = [json.loads(x) for x in open(a.events)]
    questions = [json.loads(x) for x in open(a.questions)]
    if m["thinking"]:
        events = [e for e in events if e["thinking_subsample"]]
        questions = [q for q in questions if q["thinking_subsample"]]
    if a.events_limit:
        events = events[: a.events_limit]
        if not a.questions_limit:
            keep = {e["question_id"] for e in events}
            questions = [q for q in questions if q["question_id"] in keep]
    if a.questions_limit:
        questions = questions[: a.questions_limit]

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.time()
    backend, cfg = load_backend(a.model)
    torch = backend.torch
    load_s = time.time() - t0
    log(f"{a.model} loaded in {load_s:.0f}s: {backend.describe().get('model_commit')}")
    summary = {"model": a.model, "repo_id": m["repo_id"], "model_commit": backend.describe().get("model_commit"),
               "chat_template_sha256": backend.describe().get("chat_template_sha256"),
               "template_hashes": J.TEMPLATE_HASHES, "load_seconds": round(load_s, 1),
               "events": len(events), "questions": len(questions), "think_cap": a.think_cap}
    if "judge" in a.task:
        summary["judge"] = run_judge(backend, a.model, events, out, a.think_cap, m["batch"], log)
    if "own" in a.task:
        n = a.own_samples or (1 if m["thinking"] else 3)
        summary["own"] = run_own(backend, a.model, questions, n, out, a.think_cap, m["batch"], log)
    if torch.cuda.is_available():
        summary["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        summary["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    summary["total_seconds"] = round(time.time() - t0, 1)
    summary["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    (out / f"{a.model}_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    log(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
