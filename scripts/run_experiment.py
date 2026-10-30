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
sys.path.insert(0, str(ROOT / "scripts"))

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
    pool_overlaps,
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
    # a second sample of current debate in a later child job: same kind, distinct name, so its
    # round-1 seed differs from the parent's debate condition (seeds derive from the name)
    "debate_s2": {"kind": "debate", "peer_confidence": False},
    "debate_structured": {"kind": "debate_structured", "peer_confidence": False},
    "debate_structured_open": {"kind": "debate_structured_open", "peer_confidence": False},
    "self_structured": {"kind": "self_structured"},
    # main collection (docs/NEXT_PHASE_PLAN.md, 2026-10-29 evening): a pool of n_agents (10)
    # independent answers; nested 3- and 5-agent panels; ordinary and structured debate matched
    # in initial answers, peer texts and peer order; optional second rounds and controls.
    "pool": {"kind": "pool"},
    "debate3": {"kind": "debate", "panel": 3, "peer_confidence": False},
    "structured3": {"kind": "debate_structured", "panel": 3, "peer_confidence": False},
    "debate5": {"kind": "debate", "panel": 5, "peer_confidence": False},
    "structured5": {"kind": "debate_structured", "panel": 5, "peer_confidence": False},
    "debate3_r2": {"kind": "debate", "panel": 3, "round": 2, "after": "debate3", "peer_confidence": False},
    "structured3_r2": {"kind": "debate_structured", "panel": 3, "round": 2, "after": "structured3", "peer_confidence": False},
    "debate5_r2": {"kind": "debate", "panel": 5, "round": 2, "after": "debate5", "peer_confidence": False},
    "structured5_r2": {"kind": "debate_structured", "panel": 5, "round": 2, "after": "structured5", "peer_confidence": False},
    "self_structured5": {"kind": "self_structured", "panel": 5},
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


# Known (decoder layers, width) for models this project uses: the smoke test asserts them.
EXPECTED_SHAPES = {"google/gemma-3-12b-it": (48, 3840), "google/gemma-3-1b-it": (26, 1152),
                   "google/gemma-3-270m-it": (18, 640)}


def smoke_checks(res: dict, r0, backend, cfg, n_q: int) -> dict:
    """Assert the outputs are the shape the analysis expects AND that the values are sane."""
    n_agents = cfg.debate.n_agents
    n_layers = backend.n_layers
    d_model = backend.d_model
    want = EXPECTED_SHAPES.get(cfg.model.repo_id)
    if cfg.model.backend != "stub" and want:
        assert (n_layers, d_model) == want, f"{cfg.model.repo_id}: {n_layers} x {d_model}, expected {want}"
    extra_pos = [p for p in cfg.activations.positions if p != "readout_last"]
    report = {}
    for cond, r in res.items():
        rows = r.rows
        spec = CONDITION_SPECS.get(cond, {})
        k = n_agents if spec.get("kind") == "pool" else int(spec.get("panel", n_agents))
        phases, per = ({"pre"}, k) if spec.get("kind") == "pool" else ({"pre", "peer_exposure", "post"}, 3 * k)
        assert set(rows.phase) == phases, f"{cond}: phases {set(rows.phase)}"
        assert len(rows) == n_q * per, f"{cond}: {len(rows)} rows"
        store = ActivationStore(r.run_dir)
        assert store.n_layers == n_layers, f"{cond}: {store.n_layers} layers, expected {n_layers}"
        for layer in store.layer_range():  # every stored layer, not a sample
            x = store.load_layer(layer)
            assert x.shape == (len(store.index), d_model), f"{cond}: layer {layer} shape {x.shape}"
            assert np.isfinite(x).all(), f"{cond}: non-finite stored values at layer {layer}"
        assert x.std(axis=0).max() > 0, f"{cond}: every stored state is identical"
        positions = store.index.position.value_counts().to_dict()
        assert positions.get("readout_last") == len(rows), f"{cond}: readout states {positions}"
        for _, row in rows.iterrows():
            lp = np.array([float(v) for v in row.readout_logprobs.split(",")])
            assert len(lp) == row.n_options, f"{cond}: {len(lp)} letter scores for {row.n_options} options"
            assert abs(np.exp(lp).sum() - 1) < 1e-3, f"{cond}: answer distribution not normalised"
            assert 0 <= row.readout_answer < row.n_options and 0 <= row.answer < row.n_options
        missing = {}
        for p in extra_pos:
            col = rows[f"pos_{p}_logprobs"]
            for v in col[col != ""]:
                lp = np.array([float(x) for x in v.split(",")])
                assert abs(np.exp(lp).sum() - 1) < 1e-3, f"{cond}: {p} distribution not normalised"
            reasons = rows[f"pos_{p}_missing"]
            missing[p] = reasons[reasons != ""].value_counts().to_dict()
        # exposure: a prompt-only readout -- its generation_last is the prompt's last token, and
        # it has no stated-answer states, by design
        exp = rows[rows.phase == "peer_exposure"]
        if "generation_last" in extra_pos and cfg.model.backend != "stub":
            assert (exp.pos_generation_last_token_index == exp.readout_prompt_tokens - 1).all(), \
                f"{cond}: exposure generation_last is not the last prompt token"
        for p in ("stated_answer", "before_stated_answer"):
            if p in extra_pos:
                assert (exp[f"pos_{p}_missing"] == "no_generated_response").all(), \
                    f"{cond}: exposure rows should have no {p} state"
        gen_rows = rows[rows.phase != "peer_exposure"]
        replay = gen_rows.replay_exact.value_counts(dropna=False).to_dict()
        if hasattr(backend, "count_tokens"):  # a real tokenizer: every generated row must replay exactly
            assert (gen_rows.replay_exact == True).all(), f"{cond}: inexact replay {replay}"  # noqa: E712
        report[cond] = {"positions": positions, "missing_positions": missing,
                        "replay_exact": {str(k): int(v) for k, v in replay.items()},
                        "answer_source": rows.answer_source.value_counts().to_dict(),
                        "stated_status": rows.stated_status.value_counts().to_dict(),
                        **diagnostics(r.run_dir)}
    if hasattr(backend, "count_tokens"):  # a real model
        report["position_fixture"] = position_fixture(backend, cfg)
        k = min(4, len(r0.prompts))
        texts = [p + g.text for p, g in zip(r0.prompts[:k], r0.gens[:k], strict=True)]
        n_opts = [len(r.letter_logprobs) for r in r0.reads[:k]]
        # the generated ids go in, so both readouts replay the exact generated sequence
        meta = [{**m, "gen_token_ids": g.token_ids} for m, g in zip(r0.metas[:k], r0.gens[:k], strict=True)]
        single = backend.readout(texts, n_opts, meta=meta, batch_size=1)
        batched = backend.readout(texts, n_opts, meta=meta, batch_size=k)
        dlp = max(float(np.abs(a.letter_logprobs - b.letter_logprobs).max())
                  for a, b in zip(single, batched, strict=True))
        cos = min(float(np.dot(a.hidden[-1].astype(np.float32), b.hidden[-1].astype(np.float32))
                        / (np.linalg.norm(a.hidden[-1].astype(np.float32))
                           * np.linalg.norm(b.hidden[-1].astype(np.float32)) + 1e-9))
                  for a, b in zip(single, batched, strict=True))
        agree = sum(a.answer_index == b.answer_index for a, b in zip(single, batched, strict=True))
        assert all(x.alignment.get("replay_exact") for x in single + batched), "batched/single readout not exact"
        report["batched_vs_single"] = {"max_abs_logprob_diff": round(dlp, 4),
                                       "min_last_layer_cosine": round(cos, 5), "argmax_agree": f"{agree}/{k}"}
        log(f"  batched vs single readout: {report['batched_vs_single']}")
        # gross disagreement means padding or position handling is wrong, not numerical noise
        assert cos > 0.95 and dlp < 0.5, f"batched and single readouts disagree: {report['batched_vs_single']}"
    log(f"  smoke checks passed: all {n_layers + 1} layers x {d_model}, finite, non-degenerate, "
        f"normalised, answers in each row's range, exposure semantics")
    return report


def position_fixture(backend, cfg, tol: float = 0.02) -> dict:
    """A fixed transcript with a known own answer and DISTRACTING answer lines in the prompt (the
    agent's earlier answer and two peers'). Checks: every configured position is found, on the
    right token id; the stated letter is the standalone letter token being scored; the token
    before it is the one predicting it; generation_last is the response's last token; and the
    option probabilities at every position match the model's own full logits within ``tol``
    (probability units; bf16-safe). Then the exposure (prompt-only) readout, and a permissive
    answer whose letter token may differ from the scored one."""
    import torch

    from mad.permute import identity_permutation

    tok, L = backend.tokenizer, backend._letter_ids
    base = P.round0_messages(persona=cfg.debate.personas[0], subject="physics",
                             question="Which quantity is conserved in an elastic collision?",
                             options=["mass only", "momentum only", "kinetic energy and momentum", "neither"],
                             perm=identity_permutation(4), ask_confidence=True)
    msgs = P.round1_messages(base=base, own_response="Earlier view.\nAnswer: (B)\nConfidence: 50",
                             peer_responses=["Peer reasoning.\nAnswer: (A)\nConfidence: 90",
                                             "Other reasoning.\nAnswer: (D)\nConfidence: 80"])
    prompt = backend.render(msgs)
    response = "Both are conserved in an elastic collision.\nAnswer: (C)\nConfidence: 70"
    gen = tok.encode(response, add_special_tokens=False)
    r = backend.readout([prompt + response], [4], meta=[{"gen_start_char": len(prompt), "gen_token_ids": gen}],
                        batch_size=1)[0]
    seq = tok.encode(prompt, add_special_tokens=False) + gen + tok.encode(P.READOUT_SUFFIX, add_special_tokens=False)
    pos = r.alignment["positions"]
    want = [p for p in cfg.activations.positions if p != "readout_last"]
    assert set(want) <= set(pos), f"fixture: positions {sorted(pos)} missing {set(want) - set(pos)}"
    for name, (j, tid) in pos.items():
        assert seq[j] == tid, f"fixture: {name} token id {tid} is not the input token {seq[j]}"
    if "stated_answer" in pos:
        assert pos["stated_answer"][1] == L[2], "fixture: stated-answer token is not the scored letter C"
    if "before_stated_answer" in pos:
        assert pos["before_stated_answer"][0] == pos["stated_answer"][0] - 1
    if "generation_last" in pos:
        assert pos["generation_last"][0] == len(seq) - len(tok.encode(P.READOUT_SUFFIX, add_special_tokens=False)) - 1
    with torch.no_grad():
        logits = backend.model(input_ids=torch.tensor([seq], device=backend.device), use_cache=False).logits[0]
    diffs = {}
    for name, (j, _) in [("readout_last", (len(seq) - 1, None)), *pos.items()]:
        ref = torch.softmax(logits[j, L[:4]].float(), -1).cpu().numpy()
        got = np.exp(r.letter_logprobs if name == "readout_last" else r.extra_logprobs[name])
        diffs[name] = round(float(np.abs(ref - got).max()), 5)
        assert diffs[name] < tol, f"fixture: {name} option probabilities differ from the model's logits by {diffs[name]}"
    del logits
    exp = backend.readout([prompt], [4], meta=[{"gen_start_char": None}], batch_size=1)[0]
    n_prompt = len(tok.encode(prompt, add_special_tokens=False))
    if "generation_last" in want:
        assert exp.alignment["positions"]["generation_last"][0] == n_prompt - 1, "exposure: not the prompt end"
    for p in ("stated_answer", "before_stated_answer"):
        if p in want:
            assert exp.alignment["missing"].get(p) == "no_generated_response", f"exposure: unexpected {p}"
    perm_resp = "Both are conserved.\nAnswer: C"  # permissive form
    pg = tok.encode(perm_resp, add_special_tokens=False)
    rp = backend.readout([prompt + perm_resp], [4], meta=[{"gen_start_char": len(prompt), "gen_token_ids": pg}],
                         batch_size=1)[0]
    if "stated_answer" in want:
        if "stated_answer" in rp.alignment["positions"]:
            assert rp.alignment["positions"]["stated_answer"][1] == L[2], "permissive: a non-letter token was stored"
            permissive = "stored (letter token matches)"
        else:
            assert rp.alignment["missing"]["stated_answer"] == "letter_token_mismatch", rp.alignment["missing"]
            permissive = "flagged letter_token_mismatch"
    else:
        permissive = "not requested"
    out = {"max_prob_diff_vs_model_logits": diffs, "permissive_answer": permissive,
           "positions_checked": sorted(pos)}
    log(f"  position fixture passed: {out}")
    return out


def coverage(cfg, backend, questions, conditions: list[dict] | None = None) -> dict:
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
                                 ask_confidence=cfg.debate.ask_verbalized_confidence,
                                 brevity=P.brevity_text(cfg.debate.brevity_words))

    lens = [count(backend.render(r0(q))) for q in questions]
    base = r0(questions[int(np.argmax(lens))])
    # the largest panel and the latest round requested set the worst case: a round-R prompt holds
    # the agent's own R earlier responses and its k-1 peers' previous-round responses
    rev = [c for c in (conditions or [{"kind": "debate"}]) if c.get("kind") != "pool"]
    k = max([int(c.get("panel", cfg.debate.n_agents)) for c in rev] or [1])
    rounds = max([int(c.get("round", 1)) for c in rev] or [0])
    structured = any(c.get("kind", "").startswith(("debate_structured", "self_structured")) for c in rev)
    brev = P.brevity_text(cfg.debate.brevity_words)
    msgs = base
    if rounds:
        msgs = P.round1_messages(base=base, own_response="", peer_responses=[""] * (k - 1) if k > 1 else None,
                                 ask_confidence=cfg.debate.ask_verbalized_confidence, structured=structured, brevity=brev)
        for _ in range(rounds - 1):
            msgs = P.next_round_messages(previous=msgs, own_previous_response="",
                                         peer_responses=[""] * (k - 1) if k > 1 else None,
                                         ask_confidence=cfg.debate.ask_verbalized_confidence, structured=structured,
                                         brevity=brev)
    overhead = count(backend.render(msgs)) - count(backend.render(base))
    worst_prompt = max(lens) + overhead + ((rounds + k - 1) * m if rounds else 0)
    return {
        "panel_max": k, "rounds_max": rounds,
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


def stress_context(job: dict, cfg, cov: dict) -> dict:
    """Everything a stress result depends on, recorded beside it."""
    from mad.provenance import git_state

    return {"identity_sha256": job["identity_sha256"], "gpu_identity": gpu_identity(),
            "packages": package_versions(), "git_sha": git_state()["git_sha"],
            "likelihood_chunk": cfg.activations.likelihood_chunk,
            "positions": list(cfg.activations.positions), "coverage": cov}


def stress_reusable(prior: dict, now: dict, batch: int | None = None,
                    readout_batch: int | None = None) -> tuple[bool, str]:
    """May an earlier stress result stand in for running it again? Only if it tested these
    batch sizes, lengths at least as long, and the same model, code, packages, GPU and
    extraction settings. Returns (reusable, reason when not)."""
    if not prior or not prior.get("passed"):
        return False, "no passing stress report"
    for k in ("identity_sha256", "gpu_identity", "packages", "git_sha", "likelihood_chunk", "positions"):
        if prior.get(k) != now.get(k):
            return False, f"{k} changed since the stress report"
    pc, nc = prior.get("coverage", {}), now.get("coverage", {})
    for k in ("worst_round1_prompt_tokens", "worst_readout_tokens", "token_cap"):
        if nc.get(k, 0) > pc.get(k, -1):
            return False, f"{k} {nc.get(k)} exceeds the tested {pc.get(k)}"
    cfgd = prior.get("configured", {})
    b = batch if batch is not None else now.get("batch_size")
    rb = readout_batch if readout_batch is not None else now.get("readout_batch_size")
    ok_g = {r["batch"] for r in prior.get("generation", []) if r.get("ok")}
    ok_r = {r["batch"] for r in prior.get("readout", [])
            if r.get("ok") and r.get("finite_all_layers") and r.get("likelihood_computed")}
    if (b or cfgd.get("batch_size")) not in ok_g:
        return False, f"generation batch {b} was not tested and passed"
    if (rb or cfgd.get("readout_batch_size")) not in ok_r:
        return False, f"readout batch {rb} was not tested and passed"
    return True, ""


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
    ap.add_argument("--pool", choices=["pilot", "validation", "main"], default="main",
                    help="pilot: the first --pilot-size questions of the fixed order; main: the rest")
    ap.add_argument("--pilot-size", type=int, default=120)
    ap.add_argument("--validation-size", type=int, default=0,
                    help="questions reserved after the pilot pool for validation; the main pool starts after them")
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
    ap.add_argument("--stress-batches", type=int, nargs="+", default=None,
                    help="EXTRA batch sizes to try in the worst-case memory pass, for information; "
                         "the configured sizes are always tested and must pass themselves")
    ap.add_argument("--skip-stress", action="store_true", help="never run the memory stress pass")
    ap.add_argument("--split-manifest", help="frozen split from scripts/make_split.py: must cover exactly this "
                                             "job's questions (a child's must be its parent's); its hash "
                                             "becomes part of the job identity")
    ap.add_argument("--require-split", action="store_true",
                    help="refuse to start (before loading the model) without a split manifest; the "
                         "notebook passes this for main-pool collection")
    ap.add_argument("--parent", help="child job: take round 0 from this job's saved caches")
    ap.add_argument("--subset-n", type=int, help="child job: the first N questions of the parent")
    ap.add_argument("--subset-ids", help="child job: a JSON file with the parent questions to use "
                                         "(a list, or an object with 'question_ids'); parent order is kept")
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
        if parent["identity"].get("parent"):
            raise SystemExit(f"{args.parent} is itself a child job and owns no round 0; use its parent")
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
        args.validation_size = parent["identity"]["pool"].get("validation_size", 0)
        cfg.data.dataset, cfg.data.split = DATASETS[args.dataset]
        intended, chunk = (_child_questions(out_root, parent, args.subset_n, args.subset_ids),
                           parent["identity"]["chunk_size"])
        n_total = len(intended)
    else:
        pool = load_pool(args.dataset, cfg)
        master = master_order(pool, cfg.data.sample_seed)
        intended = select(master, args.pool, n_total, args.pilot_size, args.validation_size)
        clash = pool_overlaps(out_root, args.pool, args.dataset, [q.question_id for q in intended], name)
        if clash and not args.smoke:
            raise SystemExit(f"{name} ({args.pool} pool) shares questions with job(s) of another pool in "
                             f"{out_root}: { {k: len(v) for k, v in clash.items()} }. Check --pilot-size / "
                             "--validation-size.")
    # The split is checked BEFORE any model is loaded, so a main run with a missing or wrong split
    # costs nothing.
    split_info, split_src = None, None
    if args.split_manifest and not args.smoke:
        from make_split import load_split

        _, smeta = load_split(args.split_manifest, [q.question_id for q in intended], allow_subset=bool(parent))
        split_info = {"assignment_sha256": smeta["assignment_sha256"], "counts": smeta["counts"]}
        if parent and parent["identity"].get("split") and parent["identity"]["split"] != split_info:
            raise SystemExit("a child job must use its parent's split manifest")
        split_src = Path(args.split_manifest)
    elif parent and parent["identity"].get("split") and not args.smoke:
        split_src = out_root / f"{args.parent}-split.json"  # inherited: the parent's archived copy
        if not split_src.exists():
            raise SystemExit(f"{args.parent} records a split but {split_src.name} is missing")
    if args.require_split and not args.smoke and split_info is None and not (parent and parent["identity"].get("split")):
        raise SystemExit("--require-split: this run has no split manifest. Make one with "
                         "scripts/make_split.py before any main-pool question is generated.")
    if split_src is not None:  # the assignment itself travels with the job, not only its hash
        _archive_split(split_src, out_root / f"{name}-split.json")
    questions = intended[:2] if args.smoke else intended
    if args.smoke:
        n_total, chunk = 2, 2
    cfg.data.n_questions = n_total  # the resolved config records what actually runs

    log(f"=== {name}: {len(questions)} {args.pool}-pool questions x {len(args.conditions)} conditions, "
        f"dataset={args.dataset}, model={cfg.model.repo_id}, tokens={cfg.model.max_new_tokens}, "
        f"batch gen/readout={cfg.model.batch_size}/{cfg.model.readout_batch_size or cfg.model.batch_size}"
        + (f", child of {args.parent}" if parent else ""))
    from mad.provenance import git_state

    code = git_state()
    log(f"  code at {code['git_sha']}{' (UNCOMMITTED CHANGES)' if code['git_dirty'] else ''}")
    ledger("session_start", argv=argv if argv is not None else sys.argv[1:], git_sha=code["git_sha"],
           git_dirty=code["git_dirty"])

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
    for i, c in enumerate(conds_all):  # a further round needs its previous round, listed earlier
        if c.get("after") and c["after"] not in args.conditions[:i] and not (
                parent and c["after"] in parent["identity"]["conditions"]):
            raise SystemExit(f"{c['name']} needs {c['after']} listed before it in --conditions "
                             "(or present in the parent job)")
        if int(c.get("panel", 0)) > cfg.debate.n_agents:
            raise SystemExit(f"{c['name']}: panel {c['panel']} exceeds debate.n_agents={cfg.debate.n_agents}")
    if parent:
        identity = {**parent["identity"], "name": name,
                    "conditions": {c["name"]: CONDITION_SPECS[c["name"]] for c in conds_all},
                    "prompt_template_hashes": prompt_hashes(P.TEMPLATES),
                    "parent": {"name": args.parent, "identity_sha256": parent["identity_sha256"],
                               "subset_n": n_total,
                               **({"subset_ids_sha256": hashlib.sha256(",".join(
                                   q.question_id for q in intended).encode()).hexdigest()}
                                  if args.subset_ids else {})}}
        mine = build_identity(name=name, cfg=cfg, dataset_key=args.dataset, pool=args.pool,
                              pilot_size=args.pilot_size, chunk_size=chunk, conditions={},
                              templates_hash=prompt_hashes(P.TEMPLATES), backend_desc=desc,
                              master_ids_sha="", validation_size=args.validation_size)
        for k in ("config", "backend"):
            if mine[k] != parent["identity"][k]:
                raise SystemExit(f"a child job must use its parent's {k}; it differs")
        changed = template_changes(parent["identity"]["prompt_template_hashes"], mine["prompt_template_hashes"])
        if changed:
            raise SystemExit(f"a child job must keep every template its parent used; changed or removed: {changed}")
        keep = {x.question_id for x in intended}
        chunks = [[q for q in members if q in keep] for members in parent["chunks"]]
    else:
        master_sha = hashlib.sha256(",".join(q.question_id for q in master).encode()).hexdigest()
        identity = build_identity(
            name=name, cfg=cfg, dataset_key=args.dataset, pool=args.pool, pilot_size=args.pilot_size,
            chunk_size=chunk, conditions={c["name"]: CONDITION_SPECS[c["name"]] for c in conds_all},
            templates_hash=prompt_hashes(P.TEMPLATES), backend_desc=desc, master_ids_sha=master_sha,
            validation_size=args.validation_size)
        chunks = None
    if split_info is not None:
        identity["split"] = split_info
    execution = {"batch_size": cfg.model.batch_size, "readout_batch_size": cfg.model.readout_batch_size,
                 "packages": package_versions(), **gpu_identity()}
    job = open_job(out_root, identity, questions, allow_code_change=args.allow_code_change,
                   execution=execution, chunks=chunks)
    log(f"  job {name}: {sum(1 for c in job['chunks'] if c)} chunks; history: "
        f"{[h['event'] for h in job['history']]}")

    cov = coverage(cfg, backend or _StubRender(cfg), intended, conds_all)
    log(f"  token coverage over all {len(intended)} intended questions: {cov}")
    stress_path = out_root / f"{name}-stress.json"
    if backend is not None and hasattr(backend, "stress") and not args.skip_stress:
        prior = json.loads(stress_path.read_text()) if stress_path.exists() else {}
        rb = cfg.model.readout_batch_size or cfg.model.batch_size
        now = stress_context(job, cfg, cov)
        gen_cfg_b = worst_case_generation_batch(cfg, cov)
        reuse, why = stress_reusable(prior, now, gen_cfg_b, rb)
        if reuse:
            log(f"  stress report {stress_path.name} covers these settings; not repeating")
        else:
            gen_b = sorted(set(args.stress_batches or []) | {gen_cfg_b})
            ro_b = sorted(set(args.stress_batches or []) | {rb})
            log(f"  stress ({why}): the real generation and readout/extraction workload at the "
                f"worst-case lengths, generation batches {gen_b}, readout batches {ro_b} ...")
            t = time.time()
            rep = backend.stress(cov["worst_round1_prompt_tokens"], cfg.model.max_new_tokens,
                                 cov["worst_readout_tokens"], gen_b, ro_b)
            # the CONFIGURED sizes must themselves pass; a bigger passing size is not enough
            passed = (any(r["ok"] and r["batch"] == gen_cfg_b for r in rep["generation"])
                      and any(r["ok"] and r.get("finite_all_layers") and r.get("likelihood_computed")
                              and r["batch"] == rb for r in rep["readout"]))
            rec = {**now, "configured": {"batch_size": cfg.model.batch_size, "readout_batch_size": rb,
                                         "gen_batch_tokens": cfg.model.gen_batch_tokens,
                                         "worst_case_generation_batch": gen_cfg_b},
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
        # a further round whose previous round is not in this job comes from the parent's output
        prev_dirs = {c["after"]: out_root / f"{args.parent}-{c['after']}-{tag}" for c in conds_all
                     if c.get("after") and c["after"] not in args.conditions and parent}
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
            orders = archived_peer_orders(out_root, args.parent, tag) if parent else None
            res, r0 = run_chunk(cfg, part, conds, backend=backend, job=name, chunk=tag,
                                round0_dir=out_root / f"{name}-round0-{tag}", run_dirs={**prev_dirs, **dirs},
                                extra_manifest={"job_identity_sha256": job["identity_sha256"],
                                                "split_assignment_sha256":
                                                    (job["identity"].get("split") or {}).get("assignment_sha256")},
                                parent_round0=parent_r0, peer_orders=orders)
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


def _archive_split(src: Path, dst: Path) -> None:
    """Keep the exact split assignment as ``<job>-split.json`` beside the job, so the archive set
    carries it. Refuses to replace a different assignment already there."""
    data = src.read_bytes()
    if dst.exists() and dst.resolve() != src.resolve():
        if dst.read_bytes() != data:
            raise SystemExit(f"{dst.name} already holds a different split assignment")
        return
    if dst.resolve() != src.resolve():
        dst.write_bytes(data)


def archived_peer_orders(out_root: Path, parent: str, tag: str) -> dict | None:
    """The parent's ordinary-debate peer order for one chunk, {(qid, agent): [sources]}, read from
    its archived (complete, hash-recorded) debate chunk; None when the parent has no debate
    chunk there, in which case the shared debate order stream is derived (and is identical)."""
    d = Path(out_root) / f"{parent}-debate-{tag}"
    if not is_complete(d):
        return None
    out = {}
    for line in open(d / "transcripts.jsonl"):
        r = json.loads(line)
        if r["phase"] == "post" and r.get("shown_peer_sources"):
            out[(str(r["question_id"]), int(r["agent_id"]))] = r["shown_peer_sources"].split(",")
    return out


def worst_case_generation_batch(cfg, cov: dict) -> int:
    """The generation batch the backend would form from worst-case-length prompts: batch_size
    (legacy fixed batches), or what the token budget allows (``plan_generation_batches``)."""
    from mad.backends.hf import plan_generation_batches

    b = cfg.model.batch_size
    worst = int(cov["worst_round1_prompt_tokens"])
    return plan_generation_batches([worst] * b, b, cfg.model.max_new_tokens, cfg.model.gen_batch_tokens)[0][1]


def template_changes(parent: dict, child: dict) -> list[str]:
    """Templates the parent recorded that the child changed or dropped. New templates (for new
    condition kinds) are allowed: they cannot have influenced the parent's round 0."""
    return sorted(k for k, h in parent.items() if child.get(k) != h)


def _child_questions(out_root: Path, parent: dict, subset_n: int | None, subset_ids: str | None = None) -> list:
    """The first ``subset_n`` questions of a parent job -- or the listed ``subset_ids``, in the
    parent's order -- from its verified local snapshot."""
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
    if subset_ids:
        if subset_n:
            raise SystemExit("--subset-n and --subset-ids cannot be combined")
        raw = json.loads(Path(subset_ids).read_text())
        want = [str(x) for x in (raw["question_ids"] if isinstance(raw, dict) else raw)]
        missing = sorted(set(want) - set(order))
        if missing:
            raise SystemExit(f"{len(missing)} requested questions are not in the parent job: {missing[:5]}")
        keep = set(want)
        return [by_id[i] for i in order if i in keep]
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
