"""Within-response prediction curves on SAVED responses (docs/NEXT_PHASE_PLAN.md, Stage 2).

CPU (no model):
    python scripts/commit_curves.py build --sources core   --out runs/next_commit/build_core
    python scripts/commit_curves.py build --sources extras --out runs/next_commit/build_extras

GPU (NOT run without approval; see docs/NEXT_PHASE_CPU_RESULTS.md for the session plan):
    python scripts/commit_curves.py smoke --build runs/next_commit/build_core --out runs/next_commit/smoke \\
        --batch-sizes 4 8 16 --max-minutes 25
    python scripts/commit_curves.py run --build runs/next_commit/build_core --expect-sources core \\
        --out runs/next_commit/rows_core --batch-size B --max-minutes 75

CPU:
    python scripts/commit_curves.py analyze --build runs/next_commit/build_core --rows runs/next_commit/rows_core

Every input is built from the archived prompt string and the SAVED generated token ids
(mad.curves). Outputs are bound to the build's identity and the model's; resume, partial output,
coverage and the smoke report are handled by mad.curve_runner. No activations are saved.
New runs (structured debate) are added to a build with ``--source POOL:CONDITION:VIEW``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import curve_runner as CR  # noqa: E402
from mad.analysis import commitment as CM  # noqa: E402

MODEL = {"repo_id": "google/gemma-3-12b-it", "model_commit": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
         "chat_template_sha256": "7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4"}
SOURCE_SETS = {
    "core": [("pilot", "runs/colab/pilot12b-debate", "debate"),
             ("pilot", "runs/colab/pilot12b-self_reconsider", "self_reconsider"),
             ("val", "runs/val12b/val12b-debate", "debate"),
             ("val", "runs/val12b/val12b-self-self_reconsider", "self_reconsider")],
    "extras": [("pilot", "runs/colab/pilot12b-debate_answer_only", "debate_answer_only"),
               ("pilot", "runs/colab/pilot12b-misleading_one", "misleading_one"),
               ("val", "runs/val12b/val12b-independent", "independent")],
}
OUT_ART = ROOT / "artifacts/next_phase/commit_curves"
KEEP = ("pool", "condition", "question_id", "agent_id", "subject", "n_options", "gold", "answer", "prev_answer",
        "answer_source", "prev_source", "stop_reason", "eligibility", "kind", "changed", "n_shown_agree_pre")


def records_from_views(sources) -> list[dict]:
    recs = []
    for pool, view, cond in sources:
        view = ROOT / view if not Path(view).is_absolute() else Path(view)
        df = CM.load_view(view, pool, cond)
        T = {}
        for line in open(view / "transcripts.jsonl"):
            r = json.loads(line)
            if r["phase"] == "post":
                T[(str(r["question_id"]), int(r["agent_id"]))] = r
        for r in df.to_dict("records"):
            t = T[(r["question_id"], r["agent_id"])]
            recs.append({**{k: r[k] for k in KEEP}, "view": str(view), "prompt": t["prompt"], "text": t["text"],
                         "token_ids": t["token_ids"], "prefill": t.get("prefill", ""),
                         "lp_exposure": r["lp_exposure"], "lp_before_stated": r["lp_before_stated"]})
    return recs


def build(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL["repo_id"], revision=MODEL["model_commit"])
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCDEFGHIJ"]
    sources = [s for n in args.sources for s in SOURCE_SETS[n]]
    for spec in args.source or []:
        pool, cond, view = spec.split(":", 2)
        sources.append((pool, view, cond))
    R, M, flat, off = CR.build_tables(records_from_views(sources), tok, letter_ids, args.variants)
    params = {"source_sets": sorted(args.sources), "extra_sources": sorted(args.source or []),
              "sources": [f"{p}/{c}:{v}" for p, v, c in sources], "variants": list(args.variants),
              "tokenizer": {"repo_id": MODEL["repo_id"], "revision": MODEL["model_commit"]}}
    man = CR.write_build(Path(args.out), R, M, flat, off, params)
    el = R[R.eligibility == "eligible"]
    core = M[M.variant.isin(CR.CORE_VARIANTS)]
    summary = {
        "build_id": man["build_id"], "params": params,
        "responses": {f"{p}/{c}": int(len(g)) for (p, c), g in R.groupby(["pool", "condition"])},
        "exclusions_by_condition_and_reason": {f"{p}/{c}": {k: int(v) for k, v in g.eligibility.value_counts().items()}
                                               for (p, c), g in R.groupby(["pool", "condition"])},
        "eligible_build_status": {k: int(v) for k, v in el.status.value_counts().items()},
        "capped_but_stated_kept": int(((R.stop_reason == "length") & (R.eligibility == "eligible")).sum()),
        "replay_exact": {str(k): int(v) for k, v in el.replay_exact.value_counts(dropna=False).items()},
        "standard_f1_identical_to_exact_replay": {str(k): int(v) for k, v in el.std1_equals_exact_replay.value_counts(dropna=False).items()},
        "reasoning_names_final_option": {str(k): int(v) for k, v in el.reasoning_names_final_option.value_counts(dropna=False).items()},
        "items_by_variant": {k: int(v) for k, v in M.variant.value_counts().items()},
        "boundary_notes": {str(k): int(v) for k, v in M.note.value_counts().items()},
        "retokenized_inputs_differing_from_saved_ids": len(CR.retok_differing(CR.load_build(Path(args.out)))),
        "core_unique_inputs": int(core.seq_hash.nunique()),
        "core_tokens_unique": int(core.drop_duplicates("seq_hash").n_tokens.sum()),
        "max_input_tokens": int(core.n_tokens.max()) if len(core) else 0,
    }
    tag = "_".join(sorted(args.sources)) + ("_plus" if args.source else "")
    OUT_ART.mkdir(parents=True, exist_ok=True)
    CM.to_json(summary, OUT_ART / f"build_summary_{tag}.json")
    R.drop(columns=["lp_exposure", "lp_before_stated"]).to_csv(OUT_ART / f"responses_{tag}.csv.gz", index=False)
    print(json.dumps(summary, indent=1, default=str))
    return 0


def _backend(args):
    if args.backend == "stub":
        from mad.backends.stub import StubBackend
        from mad.config import load_config

        b = StubBackend(load_config("configs/base.yaml", "configs/smoke_stub.yaml"), {})
        return b, CR.backend_identity(b)
    from mad.backends.base import build_backend
    from mad.config import load_config

    cfg = load_config("configs/base.yaml", args.config, **{"model.revision": MODEL["model_commit"]})
    b = build_backend(cfg)
    ident = CR.backend_identity(b)
    for k in ("model_commit", "chat_template_sha256"):
        if ident.get(k) != MODEL[k]:
            raise SystemExit(f"model {k} {ident.get(k)} != archive {MODEL[k]}; refusing to read")
    return b, ident


def run(args) -> int:
    B = CR.load_build(Path(args.build))
    backend, ident = _backend(args)
    st = CR.run(B, Path(args.out), backend, ident, how=args.select, batch_size=args.batch_size,
                max_minutes=args.max_minutes, expect_sources=args.expect_sources)
    print(json.dumps({k: v for k, v in st.items() if k != "batches"}, indent=1, default=str))
    return 0 if st["state"] == "complete" else 2


def smoke(args) -> int:
    B = CR.load_build(Path(args.build))
    backend, ident = _backend(args)
    rep = CR.smoke(B, Path(args.out), backend, ident, batch_sizes=tuple(args.batch_sizes), max_minutes=args.max_minutes)
    print(json.dumps(rep, indent=1, default=str))
    return 0 if rep["overall"] == "PASS" else 1


def analyze(args) -> int:
    B = CR.load_build(Path(args.build))
    res = CR.analyze(B, Path(args.rows), Path(args.out), partial_smoke=args.partial_smoke, allow_stub=args.allow_stub)
    print(json.dumps(res["reproduction"], indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", nargs="*", default=["core"], choices=sorted(SOURCE_SETS))
    b.add_argument("--source", nargs="*", help="extra views as POOL:CONDITION:VIEW_DIR")
    b.add_argument("--variants", nargs="*", default=["standard", "bridge", "exact_replay", "retokenized"])
    b.add_argument("--out", required=True)
    for name in ("run", "smoke"):
        r = sub.add_parser(name)
        r.add_argument("--build", required=True)
        r.add_argument("--out", required=True)
        r.add_argument("--backend", choices=["hf", "stub"], default="hf")
        r.add_argument("--config", default="configs/colab_12b.yaml")
        r.add_argument("--max-minutes", type=float, default=None)
        if name == "run":
            r.add_argument("--select", choices=["all", "retokenized"], default="all")
            r.add_argument("--batch-size", type=int, default=8)
            r.add_argument("--expect-sources", nargs="*", default=None,
                           help="refuse unless the build holds exactly these source sets (e.g. core)")
        else:
            r.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16])
    a = sub.add_parser("analyze")
    a.add_argument("--build", required=True)
    a.add_argument("--rows", required=True)
    a.add_argument("--out", default=str(OUT_ART))
    a.add_argument("--partial-smoke", action="store_true", help="labelled analysis of an incomplete run")
    a.add_argument("--allow-stub", action="store_true", help="tests only")
    args = ap.parse_args(argv)
    return {"build": build, "run": run, "smoke": smoke, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
