"""Forced readouts and activations for generated rows (the runtime's HF environment; never on the
GPU at the same time as vLLM). Batch size 1, fp32 head (mad.gen.readouts).

    python scripts/hf_readouts.py --task judge --out-root /content/judgec --job judgec
    python scripts/hf_readouts.py --task own   --out-root /content/judgec --job judgec
    python scripts/hf_readouts.py --task b     --out-root /content/expab --job expab --parts b_qwen_t_debate ...
    python scripts/hf_readouts.py --task posctl --out-root /content/expab --job expab --mode qwen_t
    # CPU check: --fake (a deterministic fake model; code paths only)

Reads the generation chunks of the given parts and writes, per chunk, a readout chunk in part
``<part>_ro`` (rows.parquet keyed by request_id; B also acts.npy: [rows, 2 positions, layers, d],
float16). Resumable like the generation chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mad.gen import archive as A  # noqa: E402
from mad.gen import readouts as RO  # noqa: E402
from mad.gen import tasks as T  # noqa: E402
from mad.gen.engine import MODES, StubTokenizer  # noqa: E402
from mad.job import chunk_tag  # noqa: E402
from mad.provenance import is_complete  # noqa: E402


class FakeReadouter:
    """Deterministic fake (CPU tests): hidden states are seeded by the ids."""

    d = 8

    def forward(self, ids, positions, layers):
        def vec(i, layer):
            return np.random.default_rng(hash((tuple(ids[: i + 1][-5:]), layer)) % 2 ** 32).normal(size=self.d)
        return {p: {layer: vec(p, layer) for layer in layers} for p in positions}, vec(len(ids) - 1, -1)

    def letters(self, final, n):
        z = final[:n] if len(final) >= n else np.resize(final, n)
        return z - np.log(np.exp(z).sum())

    def choices(self, final, token_ids):
        z = final[: len(token_ids)]
        return z - np.log(np.exp(z).sum())


def load_hf(repo_id):
    from mad.backends.base import build_backend  # noqa: PLC0415
    from mad.config import load_config  # noqa: PLC0415

    cfg = load_config("configs/base.yaml", "configs/colab_12b.yaml", **{
        "model.repo_id": repo_id, "model.readout_head": "fp32_head", "model.readout_batch_size": 1})
    b = build_backend(cfg)
    return RO.HFReadouter(b), b.tokenizer, b.describe().get("model_commit")


def chunk_dirs(out_root: Path, job: str, part: str) -> list[Path]:
    return sorted(d for d in out_root.glob(f"{job}-{part}-c*") if d.is_dir() and ".tmp-" not in d.name and is_complete(d))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["judge", "own", "b", "posctl"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--parts", nargs="+", default=None)
    ap.add_argument("--mode", default="qwen_t", help="posctl: the prompt format (qwen_t / qwen_nt)")
    ap.add_argument("--n-posctl", type=int, default=50)
    ap.add_argument("--plan", default=str(ROOT / "artifacts/expab/plan"))
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_root)
    if a.fake:
        R, tok, commit = FakeReadouter(), StubTokenizer(), None
    else:
        R, tok, commit = load_hf("Qwen/Qwen3-8B")
    from mad import judge as J  # noqa: PLC0415

    if a.task == "posctl":
        qs, assign, _ = T.load_questions(a.plan)
        dev = [q for q in qs if assign[q.question_id] == "dev"][: a.n_posctl]
        rng = np.random.default_rng(20261123)
        mode = MODES[a.mode]
        rows, acts = [], []
        t0 = time.time()
        for q in dev:
            letter = "ABCDEFGHIJ"[int(rng.integers(0, q.n_options))]
            msgs = RO.positive_control_messages(T.round0(q), letter)
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **(mode.template_kwargs or {}))
            row = {"request_id": f"{q.question_id}|posctl", "prompt_ids": RO.enc(tok, text), "gen_ids": [],
                   "answer_tokens": 0, "thinking_tokens": 0, "n_options": q.n_options}
            rec, act = RO.b_row(R, tok, row, thinking=mode.thinking, curves=False)
            rows.append({**rec, "question_id": q.question_id, "instructed": "ABCDEFGHIJ".index(letter)})
            acts.append(act)
        A.write_chunk(out, a.job, f"posctl_{a.mode}", 0, rows, [{"seconds": round(time.time() - t0, 2), "requests": len(rows)}],
                      {"layers": list(RO.LAYERS), "model_commit": commit}, arrays={"acts": np.stack(acts)})
        return 0
    parts = a.parts or {"judge": ["judge_traces"], "own": ["judge_own"]}.get(a.task)
    for part in parts:
        dirs = chunk_dirs(out, a.job, part)
        n_total = len(dirs)
        for i, d in enumerate(dirs):
            idx = int(d.name.rsplit("-c", 1)[1])
            if is_complete(A.chunk_dir(out, a.job, f"{part}_ro", idx)):
                continue
            rows = pd.read_parquet(d / "rows.parquet").to_dict("records")
            t0 = time.time()
            recs, acts = [], []
            for r in rows:
                if r.get("skipped"):
                    continue
                if a.task == "judge":
                    recs.append(RO.judge(R, tok, r, J.JUDGE_PREFILL))
                elif a.task == "own":
                    recs.append(RO.letters_after_answer(R, tok, r))
                else:
                    rec, act = RO.b_row(R, tok, r, thinking=r["mode"] == "qwen_t")
                    recs.append(rec)
                    acts.append(act)
            calls = [{"seconds": round(time.time() - t0, 2), "requests": len(recs),
                      "readout_tokens": int(sum(x.get("readout_tokens", 0) for x in recs))}]
            A.write_chunk(out, a.job, f"{part}_ro", idx, recs, calls,
                          {"source_chunk": d.name, "layers": list(RO.LAYERS), "model_commit": commit, "task": a.task},
                          arrays={"acts": np.stack(acts)} if acts else None)
            A.write_status(out, a.job, {"part": f"{part}_ro", "chunks_done": i + 1, "chunks_total": n_total,
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"[{time.strftime('%H:%M:%S')}] {part}_ro {chunk_tag(idx)}: {len(recs)} in {calls[0]['seconds']}s", flush=True)
        rec = json.loads((out / f"{a.job}-job.json").read_text())
        rec["parts"][f"{part}_ro"] = rec["parts"].get(part, n_total)
        A.write_job(out, a.job, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
