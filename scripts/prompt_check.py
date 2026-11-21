"""Gate: the vLLM path's rendered prompt token ids must equal the HF prompts used in main250
(Gemma) and the judge run (Qwen3 non-thinking) for 5 questions. Run in the vLLM environment as the
first step of a launch (vLLM's tokenizer loader, its transformers); any mismatch exits 1 and ends
the session before generation.

    python scripts/prompt_check.py --write --cache-dir DIR   # laptop: record the expected ids
    python scripts/prompt_check.py --out-root X --job Y      # runtime: compare (args for gen_session)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad.data import Question  # noqa: E402
from mad.gen import engine as E  # noqa: E402
from mad.gen import tasks as T  # noqa: E402

EXPECTED = ROOT / "artifacts/expab/plan/prompt_check.json"
QIDS = ["9885", "11080", "3126", "6744", "9465"]  # main250 questions that the judge run also used
MODES = {"gemma": "google/gemma-3-12b-it", "qwen_nt": "Qwen/Qwen3-8B"}


def questions():
    snap = {json.loads(x)["question_id"]: json.loads(x) for x in open(ROOT / "runs/main250/main250-questions.jsonl")} \
        if (ROOT / "runs/main250/main250-questions.jsonl").exists() else None
    return snap


def load_tok(repo, cache_dir=None):
    try:
        from vllm.transformers_utils.tokenizer import (
            get_tokenizer,  # noqa: PLC0415 -- vLLM's own loader
        )
        return get_tokenizer(repo)
    except ImportError:
        from transformers import AutoTokenizer  # noqa: PLC0415
        return AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)


def render(tok, mode, q):
    return E.render_ids(SimpleNamespace(tokenizer=tok), E.MODES[mode],
                        E.Request("x", q.question_id, 0, "c", q.n_options, messages=T.round0(q)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--job", default=None)
    a = ap.parse_args(argv)
    if a.write:
        snap = questions()
        qs = {q: snap[q] for q in QIDS}
        out = {"questions": qs, "ids": {}}
        for mode, repo in MODES.items():
            tok = load_tok(repo, a.cache_dir)
            out["ids"][mode] = {q: render(tok, mode, Question(**{**{k: d[k] for k in ("question_id", "subject", "question", "gold_index")},
                                                               "options": tuple(d["options"])})) for q, d in qs.items()}
        EXPECTED.write_text(json.dumps(out) + "\n")
        print("wrote", EXPECTED, hashlib.sha256(EXPECTED.read_bytes()).hexdigest()[:12])
        return 0
    exp = json.loads(EXPECTED.read_text())
    bad = []
    for mode, repo in MODES.items():
        tok = load_tok(repo)
        for q, d in exp["questions"].items():
            qo = Question(**{**{k: d[k] for k in ("question_id", "subject", "question", "gold_index")}, "options": tuple(d["options"])})
            got = render(tok, mode, qo)
            ok = got == exp["ids"][mode][q]
            print(json.dumps({"mode": mode, "question": q, "identical": ok, "tokens": len(got)}), flush=True)
            if not ok:
                bad.append((mode, q))
    if bad:
        print(f"PROMPT MISMATCH: {bad}; refusing to generate", flush=True)
        return 1
    print("prompt ids identical to the HF prompts (main250 / judge run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
