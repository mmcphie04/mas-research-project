"""Within-response prediction curves on SAVED responses (docs/NEXT_PHASE_PLAN.md, Stage 2).

    # CPU: build every readout input from the archives with the real tokenizer, validate
    # token boundaries and replay, report exclusions and the GPU workload. No model is loaded.
    python scripts/commit_curves.py build --sources core extras --out runs/next_commit/build

    # GPU (NOT run without approval): letter log-probabilities for the built inputs.
    python scripts/commit_curves.py run --build runs/next_commit/build --out runs/next_commit/rows \\
        --select stress --batch-sizes 4 8 16          # Stage 1 timing/stress subset
    python scripts/commit_curves.py run --build runs/next_commit/build --out runs/next_commit/rows

    # CPU: curves, paired gaps and reproduction checks.
    python scripts/commit_curves.py analyze --build runs/next_commit/build --rows runs/next_commit/rows

Inputs per eligible revision (mad.curves): six standardized fractions (reasoning + "\\n\\nAnswer: ("),
two bridge readouts with "\\nFinal answer: (" at f = 0 and 1, and the exact replay ending before
the stated letter; optionally the paper's re-tokenized construction. No activations are saved.
Sources are archive views; new runs (structured debate) are added with ``--source``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import curves as C  # noqa: E402
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
KEY = ["pool", "condition", "question_id", "agent_id"]
OUT_ART = ROOT / "artifacts/next_phase/commit_curves"


# ----------------------------------------------------------------------------- build


def _sources(names, extra):
    out = [s for n in names for s in SOURCE_SETS[n]]
    for spec in extra or []:
        pool, cond, view = spec.split(":", 2)
        out.append((pool, view, cond))
    return out


def _transcripts(view: Path) -> dict:
    T = {}
    for line in open(view / "transcripts.jsonl"):
        r = json.loads(line)
        if r["phase"] == "post":
            T[(str(r["question_id"]), int(r["agent_id"]))] = r
    return T


def build(args) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL["repo_id"], revision=MODEL["model_commit"])
    letter_ids = [tok.encode(x, add_special_tokens=False)[0] for x in "ABCDEFGHIJ"]
    variants = tuple(args.variants)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    resp_rows, meta, flat, offsets = [], [], [], [0]
    t0 = time.time()
    for pool, view, cond in _sources(args.sources, args.source):
        view = ROOT / view
        df = CM.load_view(view, pool, cond)
        T = _transcripts(view)
        for r in df.itertuples(index=False):
            rec = T[(r.question_id, r.agent_id)]
            row = {**{k: getattr(r, k) for k in (*KEY, "subject", "n_options", "gold", "answer", "prev_answer",
                                                 "answer_source", "prev_source", "stop_reason", "eligibility",
                                                 "kind", "changed", "n_shown_agree_pre")},
                   "view": str(view.relative_to(ROOT)), "status": "excluded", "n_items": 0}
            if r.eligibility == "eligible":
                b = C.build_items(tokenizer=tok, prompt=rec["prompt"], text=rec["text"], gen_ids=rec["token_ids"],
                                  n_options=r.n_options, stated_letter_id=letter_ids[r.answer], variants=variants)
                row.update(status=b.status, reasoning_chars=b.reasoning_chars, replay_exact=b.replay_exact,
                           reasoning_names_final_option=b.mentions_final_letter, n_items=len(b.items))
                std1 = next((it for it in b.items if it.variant == "standard" and it.fraction == 1.0), None)
                ex = next((it for it in b.items if it.variant == "exact_replay"), None)
                row["exact_replay_note"] = ex.note if ex else None
                row["std1_equals_exact_replay"] = (bool(ex.ids) and std1 is not None and ex.ids == std1.ids) if ex else None
                for it in b.items:
                    if not it.ids:
                        continue
                    meta.append({**{k: row[k] for k in KEY}, "item": len(meta), "variant": it.variant,
                                 "seq_hash": hashlib.sha1(np.asarray(it.ids, dtype=np.int32).tobytes()).hexdigest(),
                                 "fraction": it.fraction, "n_tokens": len(it.ids), "body_tokens": it.body_tokens,
                                 "char_cut": it.char_cut, "boundary_exact": it.boundary_exact, "note": it.note,
                                 "n_options": r.n_options})
                    flat.extend(it.ids)
                    offsets.append(len(flat))
            resp_rows.append(row)
    R = pd.DataFrame(resp_rows)
    M = pd.DataFrame(meta)
    M.to_parquet(out / "items_meta.parquet", index=False)
    np.save(out / "items_ids.npy", np.asarray(flat, dtype=np.int32))
    np.save(out / "items_offsets.npy", np.asarray(offsets, dtype=np.int64))
    R.to_csv(out / "responses.csv.gz", index=False)
    el = R[R.eligibility == "eligible"]
    std = M[M.variant == "standard"]
    summary = {
        "built_seconds": round(time.time() - t0, 1), "model": MODEL, "variants": list(variants),
        "responses": {f"{p}/{c}": int(len(g)) for (p, c), g in R.groupby(["pool", "condition"])},
        "exclusions_by_condition_and_reason": {f"{p}/{c}": {k: int(v) for k, v in g.eligibility.value_counts().items()}
                                               for (p, c), g in R.groupby(["pool", "condition"])},
        "eligible_build_status": {k: int(v) for k, v in el.status.value_counts().items()},
        "capped_but_stated_kept": int(((R.stop_reason == "length") & (R.eligibility == "eligible")).sum()),
        "replay_exact": {str(k): int(v) for k, v in el.replay_exact.value_counts(dropna=False).items()},
        "exact_replay_notes": {str(k): int(v) for k, v in el.exact_replay_note.value_counts(dropna=False).items()},
        "standard_f1_identical_to_exact_replay": {str(k): int(v) for k, v in el.std1_equals_exact_replay.value_counts(dropna=False).items()},
        "reasoning_names_final_option": {str(k): int(v) for k, v in el.reasoning_names_final_option.value_counts(dropna=False).items()},
        "boundary_exact_by_variant": {v: {str(k): int(n) for k, n in g.boundary_exact.value_counts(dropna=False).items()}
                                      for v, g in M.groupby("variant")},
        "items_by_variant": {k: int(v) for k, v in M.variant.value_counts().items()},
        "tokens": {"total": int(len(flat)), "median_item": int(M.n_tokens.median()) if len(M) else 0,
                   "max_item": int(M.n_tokens.max()) if len(M) else 0,
                   "standard_median_by_fraction": {str(f): int(g.n_tokens.median()) for f, g in std.groupby("fraction")}},
        "boundary_notes": {str(k): int(v) for k, v in M.note.value_counts().items()},
        "retokenized_identical_to_saved_ids": _retok_identical(M),
        "gpu_passes": _workload(M),
    }
    CM.to_json(summary, out / "build_summary.json")
    OUT_ART.mkdir(parents=True, exist_ok=True)
    CM.to_json(summary, OUT_ART / "build_summary.json")
    R.to_csv(OUT_ART / "responses.csv.gz", index=False)
    print(json.dumps(summary, indent=1, default=str))
    return 0


def _retok_differs(M: pd.DataFrame) -> pd.Series:
    """For each re-tokenized item: does its input differ from the saved-id construction?"""
    k = [*KEY, "fraction"]
    std = M[M.variant == "standard"].set_index(k).seq_hash.rename("std")
    rt = M[M.variant == "retokenized"].set_index(k)
    j = rt.join(std, how="left")
    return pd.Series((j.seq_hash != j["std"]).to_numpy(), index=j.item.to_numpy())


def _retok_identical(M: pd.DataFrame) -> dict:
    if not (M.variant == "retokenized").any():
        return {}
    d = _retok_differs(M)
    return {"identical": int((~d).sum()), "differ": int(d.sum()),
            "note": "only differing inputs need a GPU pass to compare the two constructions"}


def _workload(M: pd.DataFrame) -> dict:
    """Forward passes (identical id sequences are read once: the standardized f = 1 input and the
    exact replay coincide whenever the agent wrote "\n\nAnswer: (") and the minutes they would
    take at the measured-to-assumed 4.6-9 passes/s. Estimates, not measurements."""
    out = {}
    _rd = _retok_differs(M) if (M.variant == "retokenized").any() else pd.Series(dtype=bool)
    core = M.condition.isin(["debate", "self_reconsider"]) & M.pool.isin(["pilot", "val"])
    for name, sel in (("core_standard_bridge_replay", core & (M.variant != "retokenized")),
                      ("extras_standard_bridge_replay", ~core & (M.variant != "retokenized")),
                      ("retokenized_differing_only", M.item.isin(_rd[_rd].index) if len(_rd) else M.item < 0)):
        n = int(M[sel].drop_duplicates("seq_hash").shape[0]) if "seq_hash" in M else int(sel.sum())
        out[name] = {"passes": n, "minutes": [round(n / 9 / 60, 1), round(n / 4.6 / 60, 1)]}
    return out


# ----------------------------------------------------------------------------- run (GPU)


def _load_build(d: Path):
    M = pd.read_parquet(d / "items_meta.parquet")
    ids, off = np.load(d / "items_ids.npy"), np.load(d / "items_offsets.npy")
    return M, ids, off


def select_items(M: pd.DataFrame, how: str, seed: int = 2026) -> pd.DataFrame:
    """``all``: every non-retokenized item. ``stress``: every item of 40 responses (10 per core
    pool/condition cell, half changed) including their re-tokenized variants, plus the 20
    longest items overall. ``retokenized``: only the re-tokenized variant."""
    if how == "all":
        return M[M.variant != "retokenized"]
    if how == "retokenized":  # only where the paper's construction gives a different input
        d = _retok_differs(M)
        return M[M.item.isin(d[d].index)]
    if how != "stress":
        raise ValueError(how)
    R = pd.read_csv(M.attrs["responses"]) if "responses" in M.attrs else None
    keys = M.drop_duplicates(KEY)[KEY]
    if R is not None:
        keys = keys.merge(R[KEY + ["changed"]].astype({"question_id": str}), on=KEY, how="left")
    rng = np.random.default_rng(seed)
    chosen = []
    for (_pool, _cond), g in keys.groupby(["pool", "condition"]):
        for ch, n in ((1, 5), (0, 5)):
            sub = g[g.changed == ch] if "changed" in g else g
            if len(sub):
                chosen.append(sub.iloc[rng.choice(len(sub), min(n, len(sub)), replace=False)])
    pick = pd.concat(chosen)[KEY] if chosen else keys.iloc[:0]
    S = M.merge(pick, on=KEY)
    longest = M[M.variant != "retokenized"].nlargest(20, "n_tokens")
    return pd.concat([S, longest]).drop_duplicates("item")


def run(args) -> int:
    d, out = Path(args.build), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    M, ids, off = _load_build(d)
    M.attrs["responses"] = str(d / "responses.csv.gz")
    S = select_items(M, args.select)
    if args.limit:
        S = S.head(args.limit)
    done = set()
    for p in out.glob("rows_*.parquet"):
        done |= set(pd.read_parquet(p, columns=["item"]).item)
    S = S[~S.item.isin(done)]
    dup = S[S.duplicated("seq_hash", keep="first")]  # identical inputs are read once
    S = S.drop_duplicates("seq_hash").sort_values("n_tokens", ascending=False)  # longest first: OOM early
    if args.backend == "stub":
        from mad.backends.stub import StubBackend
        from mad.config import load_config

        backend = StubBackend(load_config("configs/base.yaml", "configs/smoke_stub.yaml"), {})
        ident = {"backend": "stub"}
    else:
        from mad.backends.base import build_backend
        from mad.config import load_config

        cfg = load_config("configs/base.yaml", args.config, **{"model.revision": MODEL["model_commit"]})
        backend = build_backend(cfg)
        ident = backend.describe()
        for k in ("model_commit", "chat_template_sha256"):
            if ident.get(k) != MODEL[k] and not args.allow_model_mismatch:
                raise SystemExit(f"model {k} {ident.get(k)} != archive {MODEL[k]}; refusing to read")
    log = {"identity": ident, "select": args.select, "items": int(len(S)), "batches": []}
    torch = getattr(backend, "torch", None)
    t_start = time.time()
    for bs in args.batch_sizes:
        part = S if len(args.batch_sizes) == 1 else S.head(args.timing_items)
        rows = []
        for start in range(0, len(part), bs):
            chunk = part.iloc[start: start + bs]
            seqs = [ids[off[i]: off[i + 1]].tolist() for i in chunk.item]
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            lp = backend.letter_logprobs_ids(seqs, chunk.n_options.tolist(), batch_size=bs)
            dt = time.time() - t
            peak = (torch.cuda.max_memory_allocated() / 1e9) if torch is not None and torch.cuda.is_available() else None
            log["batches"].append({"batch_size": bs, "n": len(seqs), "tokens": int(sum(map(len, seqs))),
                                   "max_len": int(max(map(len, seqs))), "seconds": round(dt, 3),
                                   "peak_gb": None if peak is None else round(peak, 2)})
            rows += [{"item": int(i), "logprobs": v.tolist(), "batch_size": bs} for i, v in zip(chunk.item, lp, strict=True)]
            if len(rows) >= 2000 or start + bs >= len(part):
                pd.DataFrame(rows).to_parquet(out / f"rows_bs{bs}_{int(time.time() * 1000)}.parquet", index=False)
                rows = []
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                log["stopped"] = "max_minutes reached"
                break
    if len(dup):  # copy results to items whose input is byte-identical to one that was read
        got = pd.concat([pd.read_parquet(p) for p in out.glob("rows_*.parquet")]).drop_duplicates("item")
        got = got.merge(M[["item", "seq_hash"]], on="item")
        fan = dup[["item", "seq_hash"]].merge(got.drop(columns="item"), on="seq_hash")
        if len(fan):
            fan.drop(columns="seq_hash").to_parquet(out / f"rows_dedup_{int(time.time() * 1000)}.parquet", index=False)
        log["deduplicated_items"] = int(len(fan))
    log["seconds_total"] = round(time.time() - t_start, 1)
    b = pd.DataFrame(log["batches"])
    if len(b):
        log["passes_per_second_by_batch_size"] = {int(k): round(g.n.sum() / g.seconds.sum(), 2) for k, g in b.groupby("batch_size")}
        log["peak_gb_max"] = None if b.peak_gb.isna().all() else float(b.peak_gb.max())
    (out / f"run_log_{int(time.time())}.json").write_text(json.dumps(log, indent=1, default=str))
    print(json.dumps({k: v for k, v in log.items() if k != "batches"}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------------------- analyze (CPU)


def analyze(args) -> int:
    d, rows_dir = Path(args.build), Path(args.rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    M, _, _ = _load_build(d)
    rows = pd.concat([pd.read_parquet(p) for p in sorted(rows_dir.glob("rows_*.parquet"))]).drop_duplicates("item")
    X = M.merge(rows, on="item")
    R = pd.read_csv(d / "responses.csv.gz", dtype={"question_id": str})
    saved = pd.concat([CM.load_view(ROOT / v, p, c)[KEY + ["lp_exposure", "lp_before_stated"]]
                       for (p, c, v) in R[["pool", "condition", "view"]].drop_duplicates().itertuples(index=False)])
    X = X.merge(R[KEY + ["answer", "prev_answer", "gold", "kind", "changed", "reasoning_names_final_option"]], on=KEY)
    X["cluster"] = X.pool + ":" + X.question_id
    am = np.array([int(np.argmax(v)) for v in X.logprobs])
    X["is_eventual"], X["is_previous"], X["is_gold"] = (am == X.answer).astype(float), (am == X.prev_answer).astype(float), (am == X.gold).astype(float)
    X["p_eventual"] = [float(np.exp(v[a])) for v, a in zip(X.logprobs, X.answer, strict=True)]
    curves = []
    std = X[X.variant == "standard"]
    for pool_lab, gp in (("pilot", std[std.pool == "pilot"]), ("val", std[std.pool == "val"]), ("pooled", std)):
        for (cond, f), g in gp.groupby(["condition", "fraction"]):
            for sub, gs in CM.subsets(g):
                for m in ("is_eventual", "is_gold", "is_previous", "p_eventual"):
                    curves.append({"pool": pool_lab, "condition": cond, "fraction": f, "subset": sub, "metric": m,
                                   **CM.cluster_boot(gs, m, seed=3)})
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    paired = {str(f): CM.paired_gap(std[std.fraction == f], "debate", "self_reconsider", "is_eventual", seed=4)
              for f in C.FRACS}
    # reproduction and consistency checks
    S = X.merge(saved, on=KEY, how="left")
    rep = {}

    def compare(sub, col, name):
        a = [(np.asarray(v), s) for v, s in zip(sub.logprobs, sub[col], strict=True) if s is not None]
        if not a:
            return
        agree = np.mean([np.argmax(v) == np.argmax(s) for v, s in a])
        dtop = np.array([abs(v[np.argmax(s)] - s[np.argmax(s)]) for v, s in a])
        rep[name] = {"n": len(a), "argmax_agreement": float(agree), "median_abs_dlogp_top": float(np.median(dtop)),
                     "max_abs_dlogp_top": float(dtop.max())}

    compare(S[(S.variant == "bridge") & (S.fraction == 0.0)], "lp_exposure", "bridge_f0_vs_saved_exposure")
    compare(S[S.variant == "exact_replay"], "lp_before_stated", "exact_replay_vs_saved_before_stated_answer")
    s1 = X[(X.variant == "standard") & (X.fraction == 1.0)].set_index(KEY)
    ex = X[X.variant == "exact_replay"].set_index(KEY)
    common = s1.index.intersection(ex.index)
    if len(common):
        rep["standard_f1_vs_exact_replay_argmax_agreement"] = float(np.mean(
            [np.argmax(s1.loc[k].logprobs) == np.argmax(ex.loc[k].logprobs) for k in common]))
    rep["standard_f1_argmax_equals_stated"] = float(X[(X.variant == "standard") & (X.fraction == 1.0)].is_eventual.mean())
    rt = X[X.variant == "retokenized"].set_index(KEY + ["fraction"])
    st = X[X.variant == "standard"].set_index(KEY + ["fraction"])
    both = rt.index.intersection(st.index)
    if len(both):
        rep["retokenized_vs_standard"] = {"n": int(len(both)),
                                          "identical_ids_share": float(np.mean([rt.loc[k].seq_hash == st.loc[k].seq_hash for k in both])),
                                          "argmax_agreement": float(np.mean([np.argmax(rt.loc[k].logprobs) == np.argmax(st.loc[k].logprobs) for k in both]))}
    CM.to_json({"paired_debate_minus_self_changed_is_eventual": paired, "reproduction": rep}, out / "analysis.json")
    print(json.dumps(rep, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--sources", nargs="*", default=["core"], choices=sorted(SOURCE_SETS))
    b.add_argument("--source", nargs="*", help="extra views as POOL:CONDITION:VIEW_DIR")
    b.add_argument("--variants", nargs="*", default=["standard", "bridge", "exact_replay", "retokenized"])
    b.add_argument("--out", default="runs/next_commit/build")
    r = sub.add_parser("run")
    r.add_argument("--build", default="runs/next_commit/build")
    r.add_argument("--out", default="runs/next_commit/rows")
    r.add_argument("--backend", choices=["hf", "stub"], default="hf")
    r.add_argument("--config", default="configs/colab_12b.yaml")
    r.add_argument("--select", choices=["all", "stress", "retokenized"], default="all")
    r.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
    r.add_argument("--timing-items", type=int, default=64, help="items per batch size when several are timed")
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--max-minutes", type=float, default=None)
    r.add_argument("--allow-model-mismatch", action="store_true", help="tests with a different model only")
    a = sub.add_parser("analyze")
    a.add_argument("--build", default="runs/next_commit/build")
    a.add_argument("--rows", default="runs/next_commit/rows")
    a.add_argument("--out", default=str(OUT_ART))
    args = ap.parse_args(argv)
    return {"build": build, "run": run, "analyze": analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
