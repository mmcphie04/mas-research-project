"""Running and checking the within-response curve readouts (scripts/commit_curves.py).

Safety properties (docs/NEXT_PHASE_PLAN.md, review of 2026-10-29):

* **Bound outputs.** A build directory has a ``build_id`` = hash of its files and parameters,
  re-verified on every load. An output directory records the ``build_id``, the backend identity
  (backend, repo, model commit, chat template, dtype) and its purpose, and refuses to continue
  under any other -- so stub outputs can never be resumed into a model run, and outputs of a
  rebuilt input set can never be mixed in.
* **Row validation.** Every stored row carries the item's sequence hash and option count, and is
  checked against the build (unknown item, hash or option-count mismatch, wrong length,
  non-finite or unnormalised distribution -> refusal). Item numbers alone are never trusted.
* **Stopping.** One deadline for the whole command (all batch sizes and modes). Completed rows
  are flushed atomically (temporary file, then rename) at every flush point and on a time limit,
  a keyboard interrupt or an out-of-memory error. A status file records the state, the expected
  and completed items, and which are missing.
* **Coverage.** Final analysis requires every expected item. A partial run can be analysed only
  as an explicitly labelled smoke analysis, restricted to responses that are complete at every
  fraction, so populations never differ across fractions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mad import curves as C

KEY = ["pool", "condition", "question_id", "agent_id"]
CORE_VARIANTS = ("standard", "standard_after_opening", "bridge", "bridge_after_opening",
                 "bridge_pre_opening", "exact_replay")
STANDARD_FAMILIES = ("standard", "standard_after_opening")
IDENTITY_KEYS = ("backend", "repo_id", "model_commit", "chat_template_sha256", "dtype")
BUILD_FILES = ("items_meta.parquet", "items_ids.npy", "items_offsets.npy", "responses.parquet")


class RunMismatch(RuntimeError):
    """Inputs, model or backend differ from what an output directory was created with."""


class InvalidRows(RuntimeError):
    """Stored rows that do not belong to this build or are not valid distributions."""


class IncompleteRun(RuntimeError):
    """Final analysis asked for while expected items are missing."""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seq_hash(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- build


def build_tables(records: list[dict], tokenizer, letter_ids: list[int], variants) -> tuple:
    """Readout inputs for archived responses. ``records``: one dict per revision with KEY fields,
    eligibility, outcome fields, ``prompt``, ``text``, ``token_ids``, ``prefill``, ``n_options``,
    ``answer`` and the saved ``lp_exposure`` / ``lp_before_stated`` (lists or None).
    Returns (responses, items_meta, flat_ids, offsets)."""
    resp, meta, flat, offsets = [], [], [], [0]
    for r in records:
        row = {k: r[k] for k in r if k not in ("prompt", "text", "token_ids", "lp_exposure", "lp_before_stated")}
        row["lp_exposure"] = json.dumps(None if r.get("lp_exposure") is None else [float(x) for x in r["lp_exposure"]])
        row["lp_before_stated"] = json.dumps(None if r.get("lp_before_stated") is None else [float(x) for x in r["lp_before_stated"]])
        row.update(status="excluded", n_items=0)
        if r["eligibility"] == "eligible":
            b = C.build_items(tokenizer=tokenizer, prompt=r["prompt"], text=r["text"], gen_ids=r["token_ids"],
                              n_options=r["n_options"], stated_letter_id=letter_ids[r["answer"]],
                              variants=tuple(variants), prefill=r.get("prefill") or "")
            ex = next((it for it in b.items if it.variant == "exact_replay"), None)
            s1 = next((it for it in b.items if it.variant in STANDARD_FAMILIES and it.fraction == 1.0), None)
            row.update(status=b.status, reasoning_chars=b.reasoning_chars, replay_exact=b.replay_exact,
                       reasoning_names_final_option=b.mentions_final_letter, n_items=len(b.items),
                       exact_replay_note=ex.note if ex else None,
                       std1_equals_exact_replay=bool(ex and ex.ids and s1 and ex.ids == s1.ids))
            for it in b.items:
                if not it.ids:
                    continue
                meta.append({**{k: r[k] for k in KEY}, "item": len(meta), "variant": it.variant,
                             "fraction": it.fraction, "n_tokens": len(it.ids), "body_tokens": it.body_tokens,
                             "char_cut": it.char_cut, "boundary_exact": it.boundary_exact, "note": it.note,
                             "n_options": int(r["n_options"]), "seq_hash": seq_hash(it.ids)})
                flat.extend(it.ids)
                offsets.append(len(flat))
        resp.append(row)
    return pd.DataFrame(resp), pd.DataFrame(meta), np.asarray(flat, dtype=np.int32), np.asarray(offsets, dtype=np.int64)


def write_build(out: Path, R, M, flat, offsets, params: dict) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for f in BUILD_FILES + ("build_manifest.json",):
        if (out / f).exists():
            (out / f).unlink()  # a rebuild replaces the whole input set (and so its build_id)
    M.to_parquet(out / "items_meta.parquet", index=False)
    np.save(out / "items_ids.npy", flat)
    np.save(out / "items_offsets.npy", offsets)
    R.to_parquet(out / "responses.parquet", index=False)
    files = {f: sha256_file(out / f) for f in BUILD_FILES}
    bid = hashlib.sha256(json.dumps({"files": files, "params": params}, sort_keys=True).encode()).hexdigest()
    man = {"build_id": bid, "files": files, "params": params, "n_items": int(len(M)), "n_responses": int(len(R)),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    _atomic_json(out / "build_manifest.json", man)
    return man


@dataclass
class Build:
    dir: Path
    manifest: dict
    M: pd.DataFrame
    ids: np.ndarray
    off: np.ndarray
    R: pd.DataFrame

    @property
    def build_id(self) -> str:
        return self.manifest["build_id"]

    def seq(self, item: int) -> list[int]:
        return self.ids[self.off[item]: self.off[item + 1]].tolist()


def load_build(d: Path) -> Build:
    d = Path(d)
    man = json.loads((d / "build_manifest.json").read_text())
    for f, h in man["files"].items():
        if sha256_file(d / f) != h:
            raise RunMismatch(f"{d / f} does not match its build manifest; the build was altered")
    M = pd.read_parquet(d / "items_meta.parquet")
    R = pd.read_parquet(d / "responses.parquet")
    return Build(d, man, M, np.load(d / "items_ids.npy"), np.load(d / "items_offsets.npy"), R)


# ----------------------------------------------------------------------------- binding & rows


def backend_identity(backend) -> dict:
    d = backend.describe() if hasattr(backend, "describe") else {"backend": getattr(backend, "name", "?")}
    return {k: d.get(k) for k in IDENTITY_KEYS}


def bind_output(out: Path, B: Build, identity: dict, purpose: str) -> dict:
    """Create the output directory's binding, or verify an existing one."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "run_manifest.json"
    want = {"build_id": B.build_id, "build_dir": str(B.dir), "identity": identity, "purpose": purpose}
    if p.exists():
        have = json.loads(p.read_text())
        for k in ("build_id", "identity", "purpose"):
            if have.get(k) != want[k]:
                raise RunMismatch(f"{out} holds outputs for a different {k}: {have.get(k)} != {want[k]}; "
                                  "use a new output directory")
        return have
    _atomic_json(p, {**want, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return want


def load_rows(out: Path, B: Build) -> pd.DataFrame:
    """Every stored row, validated against the build."""
    parts = sorted(Path(out).glob("rows_*.parquet"))
    if not parts:
        return pd.DataFrame(columns=["item", "mode", "seq_hash", "n_options", "logprobs"])
    rows = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    meta = B.M.set_index("item")
    unknown = ~rows.item.isin(meta.index)
    if unknown.any():
        raise InvalidRows(f"{int(unknown.sum())} rows name items not in build {B.build_id[:12]}")
    m = meta.loc[rows.item]
    if (rows.seq_hash.to_numpy() != m.seq_hash.to_numpy()).any():
        raise InvalidRows("rows whose sequence hash differs from the build's input for that item")
    if (rows.n_options.to_numpy() != m.n_options.to_numpy()).any():
        raise InvalidRows("rows whose option count differs from the build's")
    for lp, n in zip(rows.logprobs, rows.n_options, strict=True):
        v = np.asarray(lp, dtype=float)
        if len(v) != n or not np.isfinite(v).all() or abs(np.exp(v).sum() - 1) > 1e-3:
            raise InvalidRows("a stored distribution is not a finite, normalised distribution over its options")
    return rows


class PartWriter:
    """Buffers rows and writes them atomically as numbered parquet parts."""

    def __init__(self, out: Path, mode: str):
        self.out, self.mode, self.buf, self.n = Path(out), mode, [], 0

    def add(self, rows: list[dict]):
        self.buf += rows

    def flush(self):
        if not self.buf:
            return
        name = f"rows_{self.mode}_{int(time.time() * 1e6)}_{self.n:04d}.parquet"
        tmp = self.out / (name + f".tmp-{os.getpid()}")
        pd.DataFrame(self.buf).to_parquet(tmp, index=False)
        os.replace(tmp, self.out / name)
        self.n += 1
        self.buf = []


# ----------------------------------------------------------------------------- reading


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def read_items(backend, B: Build, items: pd.DataFrame, batch_size: int, mode: str, out: Path,
               deadline: float | None, stats: list, flush_every: int = 25, head: str = "fp32_head",
               on_flush=None) -> str:
    """Read ``items`` (longest first) in batches; returns 'complete', 'time_limit', 'interrupted'
    or 'out_of_memory'. Completed rows are always flushed before returning; ``on_flush`` (e.g.
    a status writer) runs after every flush. Each row records its batch, its left padding and
    the lengths of its batch-mates."""
    w = PartWriter(out, mode)

    def flush():
        w.flush()
        if on_flush is not None:
            on_flush()
    torch = getattr(backend, "torch", None)
    cuda = torch is not None and torch.cuda.is_available()
    items = items.sort_values("n_tokens", ascending=False)
    state = "complete"
    try:
        for bi, start in enumerate(range(0, len(items), batch_size)):
            if deadline is not None and time.time() >= deadline:
                state = "time_limit"
                break
            chunk = items.iloc[start: start + batch_size]
            seqs = [B.seq(int(i)) for i in chunk.item]
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                lp = backend.letter_logprobs_ids(seqs, chunk.n_options.tolist(), batch_size=batch_size, head=head)
            except RuntimeError as exc:
                if _is_oom(exc):
                    state = "out_of_memory"
                    stats.append({"mode": mode, "batch_size": batch_size, "error": "out_of_memory",
                                  "max_len": int(max(map(len, seqs)))})
                    break
                raise
            dt = time.time() - t
            stats.append({"mode": mode, "batch_size": batch_size, "n": len(seqs), "tokens": int(sum(map(len, seqs))),
                          "max_len": int(max(map(len, seqs))), "seconds": round(dt, 4),
                          "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None})
            width, lens = max(map(len, seqs)), ",".join(str(len(x)) for x in seqs)
            w.add([{"item": int(i), "mode": mode, "seq_hash": h, "n_options": int(n), "logprobs": v.tolist(),
                    "batch_size": batch_size, "batch_index": bi, "pad": width - len(sq), "batch_lens": lens,
                    "head": head}
                   for i, h, n, v, sq in zip(chunk.item, chunk.seq_hash, chunk.n_options, lp, seqs, strict=True)])
            if (bi + 1) % flush_every == 0:
                flush()
    except KeyboardInterrupt:
        state = "interrupted"
    finally:
        flush()
    return state


def _fan_out(out: Path, B: Build, wanted: pd.DataFrame, mode: str) -> int:
    """Rows for items whose input is byte-identical to one already read in this mode."""
    rows = load_rows(out, B)
    rows = rows[rows["mode"] == mode]
    have = set(rows.item)
    need = wanted[~wanted.item.isin(have)]
    src = rows.drop_duplicates("seq_hash").set_index("seq_hash")
    need = need[need.seq_hash.isin(src.index)]
    if need.empty:
        return 0
    w = PartWriter(out, mode)
    w.add([{"item": int(i), "mode": mode, "seq_hash": h, "n_options": int(n), "logprobs": list(src.loc[h].logprobs),
            "batch_size": int(src.loc[h].batch_size), "copied_from_identical_input": True}
           for i, h, n in zip(need.item, need.seq_hash, need.n_options, strict=True)])
    w.flush()
    return int(len(need))


def _pending(out: Path, B: Build, wanted: pd.DataFrame, mode: str) -> pd.DataFrame:
    rows = load_rows(out, B)
    rows = rows[rows["mode"] == mode]
    rest = wanted[~wanted.item.isin(set(rows.item)) & ~wanted.seq_hash.isin(set(rows.seq_hash))]
    return rest.drop_duplicates("seq_hash")


def _status(out: Path, B: Build, wanted: dict[str, pd.DataFrame], state: str, t0: float, stats: list,
            extra: dict | None = None) -> dict:
    rows = load_rows(out, B)
    modes = {}
    for mode, w in wanted.items():
        have = set(rows[rows["mode"] == mode].item)
        miss = sorted(set(w.item) - have)
        modes[mode] = {"expected": int(len(w)), "done": int(len(set(w.item) & have)), "missing": len(miss),
                       "missing_sample": miss[:50]}
    complete = all(v["missing"] == 0 for v in modes.values())
    st = {"state": "complete" if complete and state == "complete" else (state if state != "complete" else "incomplete"),
          "build_id": B.build_id, "modes": modes, "seconds": round(time.time() - t0, 1), "batches": stats,
          "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **(extra or {})}
    _atomic_json(Path(out) / "run_status.json", st)
    return st


# ----------------------------------------------------------------------------- selections


def select(B: Build, how: str) -> pd.DataFrame:
    M = B.M
    if how == "all":
        return M[M.variant.isin(CORE_VARIANTS)]
    if how == "retokenized":
        return M[M.item.isin(retok_differing(B))]
    raise ValueError(how)


def retok_differing(B: Build) -> list[int]:
    M = B.M
    k = [*KEY, "fraction"]
    std = M[M.variant.isin(STANDARD_FAMILIES)].set_index(k).seq_hash.rename("std")
    rt = M[M.variant == "retokenized"].set_index(k).join(std, how="left")
    return [int(i) for i in rt.item[rt.seq_hash != rt["std"]]]


def smoke_selection(B: Build, seed: int = 2026, per_cell: tuple[int, int] = (5, 5), n_longest: int = 20) -> pd.DataFrame:
    """40 stratified responses (5 changed + 5 unchanged per pool/condition cell) with all their
    core inputs; every re-tokenized input that differs from the saved-id construction, with its
    counterpart; and the ``n_longest`` longest core inputs."""
    R = B.R[(B.R.eligibility == "eligible") & (B.R.status == "ok")]
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in R.groupby(["pool", "condition"]):
        for ch, n in zip((1, 0), per_cell, strict=True):
            sub = g[g.changed == ch]
            if len(sub):
                picks.append(sub.iloc[np.sort(rng.choice(len(sub), min(n, len(sub)), replace=False))][KEY])
    keys = pd.concat(picks) if picks else B.R.iloc[:0][KEY]
    core = B.M[B.M.variant.isin(CORE_VARIANTS)]
    a = core.merge(keys, on=KEY).assign(why="stratified_response")
    rt_items = retok_differing(B)
    rt = B.M[B.M.item.isin(rt_items)]
    counterpart = core[core.variant.isin(STANDARD_FAMILIES)].merge(rt[[*KEY, "fraction"]], on=[*KEY, "fraction"])
    b = pd.concat([rt, counterpart]).assign(why="retokenized_comparison")
    c = core.nlargest(n_longest, "n_tokens").assign(why="longest_input")
    return pd.concat([a, b, c]).drop_duplicates("item")


# ----------------------------------------------------------------------------- commands


def run(B: Build, out: Path, backend, identity: dict, *, how: str = "all", batch_size: int = 8,
        max_minutes: float | None = None, expect_sources: list[str] | None = None,
        head: str = "fp32_head") -> dict:
    if expect_sources is not None and sorted(B.manifest["params"].get("source_sets", [])) != sorted(expect_sources):
        raise RunMismatch(f"build {B.dir} holds source sets {B.manifest['params'].get('source_sets')}, "
                          f"not {expect_sources}")
    identity = {**identity, "readout_head": head}
    bind_output(out, B, identity, f"run:{how}")
    t0 = time.time()
    deadline = t0 + 60 * max_minutes if max_minutes else None
    wanted = select(B, how)
    stats: list = []
    state = read_items(backend, B, _pending(out, B, wanted, "run"), batch_size, "run", out, deadline, stats,
                       head=head, on_flush=lambda: _status(out, B, {"run": wanted}, "running", t0, stats))
    fanned = _fan_out(out, B, wanted, "run")
    return _status(out, B, {"run": wanted}, state, t0, stats, {"copied_from_identical_inputs": fanned})


def smoke(B: Build, out: Path, backend, identity: dict, *, batch_sizes=(4, 8, 16), max_minutes: float | None = None,
          seed: int = 2026, gates: dict | None = None, head: str = "fp32_head") -> dict:
    """Single-item reads of the smoke set, then the same inputs batched at each batch size; one
    deadline across all of it; then an explicit pass/fail report (``smoke_report``)."""
    identity = {**identity, "readout_head": head}
    bind_output(out, B, identity, "smoke")
    t0 = time.time()
    deadline = t0 + 60 * max_minutes if max_minutes else None
    S = smoke_selection(B, seed)
    S.to_csv(Path(out) / "smoke_selection.csv", index=False)
    wanted = {"single": S, **{f"batch{bs}": S for bs in batch_sizes}}
    stats, state = [], "complete"
    for mode, bs in [("single", 1), *[(f"batch{b}", b) for b in batch_sizes]]:
        state = read_items(backend, B, _pending(out, B, S, mode), bs, mode, out, deadline, stats, head=head,
                           on_flush=lambda: _status(out, B, wanted, "running", t0, stats))
        _fan_out(out, B, S, mode)
        if state != "complete":
            break  # time limit, interruption or OOM stops the WHOLE smoke run
    st = _status(out, B, wanted, state, t0, stats)
    rep = smoke_report(B, out, st, identity, gates or {})
    _atomic_json(Path(out) / "smoke_report.json", rep)
    return rep


DEFAULT_GATES = {"reproduction_argmax_min": 0.975, "reproduction_median_dlogp_max": 0.05,
                 "batched_argmax_min": 0.99, "batched_median_dlogp_max": 0.02, "batched_max_dlogp_max": 0.10,
                 "peak_gb_max": 38.0, "projected_core_minutes_max": 78.0}


def _cmp(a: list, b: list) -> dict:
    if not a:
        return {"n": 0}
    agree = np.mean([np.argmax(x) == np.argmax(y) for x, y in zip(a, b, strict=True)])
    d = np.array([abs(np.asarray(x)[np.argmax(y)] - np.asarray(y)[np.argmax(y)]) for x, y in zip(a, b, strict=True)])
    dall = np.array([np.max(np.abs(np.asarray(x) - np.asarray(y))) for x, y in zip(a, b, strict=True)])
    return {"n": len(a), "argmax_agreement": float(agree), "median_abs_dlogp_top": float(np.median(d)),
            "max_abs_dlogp_any": float(dall.max())}


def smoke_report(B: Build, out: Path, status: dict, identity: dict, gates: dict) -> dict:
    g = {**DEFAULT_GATES, **gates}
    rows = load_rows(out, B)
    S = pd.read_csv(Path(out) / "smoke_selection.csv")
    meta = B.M.set_index("item")
    R = B.R.set_index(KEY)
    checks = {}

    def check(name, value, ok, gated=True, note=""):
        checks[name] = {"value": value, "pass": ok, "gated": gated, **({"note": note} if note else {})}

    check("smoke_complete", {m: v["missing"] for m, v in status["modes"].items()},
          status["state"] == "complete")
    check("finite_and_normalised", int(len(rows)), True, note="every stored row passed validation on load")
    single = rows[rows["mode"] == "single"].set_index("item")

    def saved(item, col):
        k = tuple(meta.loc[item, KEY])
        v = json.loads(R.loc[k, col]) if k in R.index else None
        return v

    for name, variants, col in (("bridge_f0_vs_saved_exposure", ("bridge", "bridge_pre_opening"), "lp_exposure"),
                                ("exact_replay_vs_saved_before_stated_answer", ("exact_replay",), "lp_before_stated")):
        its = [i for i in S.item if i in single.index and meta.loc[i, "variant"] in variants
               and (meta.loc[i, "fraction"] in (0.0, None) or pd.isna(meta.loc[i, "fraction"]))]
        pairs = [(single.loc[i].logprobs, saved(i, col)) for i in its]
        pairs = [(a, b) for a, b in pairs if b is not None]
        c = _cmp([a for a, _ in pairs], [b for _, b in pairs])
        ok = c["n"] > 0 and c["argmax_agreement"] >= g["reproduction_argmax_min"] and \
            c["median_abs_dlogp_top"] <= g["reproduction_median_dlogp_max"]
        check(name, c, ok)
    rt = retok_differing(B)
    rts = [i for i in rt if i in single.index]
    if rts:
        k = [*KEY, "fraction"]
        std = B.M[B.M.variant.isin(STANDARD_FAMILIES)].set_index(k).item
        pairs = [(single.loc[i].logprobs, single.loc[std.loc[tuple(meta.loc[i, k])]].logprobs) for i in rts
                 if std.loc[tuple(meta.loc[i, k])] in single.index]
        check("retokenized_vs_saved_ids", _cmp([a for a, _ in pairs], [b for _, b in pairs]), None, gated=False,
              note="reported, not gated: the paper's construction differs from ours on these inputs")
    peak = {}
    thr = {}
    for mode in [m for m in status["modes"] if m.startswith("batch")]:
        bm = rows[rows["mode"] == mode].set_index("item")
        common = [i for i in bm.index if i in single.index]
        c = _cmp([bm.loc[i].logprobs for i in common], [single.loc[i].logprobs for i in common])
        ok = c["n"] > 0 and c["argmax_agreement"] >= g["batched_argmax_min"] and \
            c["median_abs_dlogp_top"] <= g["batched_median_dlogp_max"] and c["max_abs_dlogp_any"] <= g["batched_max_dlogp_max"]
        check(f"{mode}_vs_single", c, ok)
        b = pd.DataFrame([x for x in status["batches"] if x.get("mode") == mode and "seconds" in x])
        oom = any(x.get("error") == "out_of_memory" for x in status["batches"] if x.get("mode") == mode)
        if len(b):
            thr[mode] = {"passes_per_s": float(b.n.sum() / b.seconds.sum()), "tokens_per_s": float(b.tokens.sum() / b.seconds.sum())}
            peak[mode] = None if b.peak_gb.isna().all() else float(b.peak_gb.max())
        check(f"{mode}_memory", {"peak_gb": peak.get(mode), "out_of_memory": oom},
              (not oom) and (peak.get(mode) is None or peak[mode] <= g["peak_gb_max"]),
              note="not measured on this backend" if peak.get(mode) is None else "")
    core = B.M[B.M.variant.isin(CORE_VARIANTS)].drop_duplicates("seq_hash")
    ok_modes = [m for m in thr if checks.get(f"{m}_vs_single", {}).get("pass") and checks.get(f"{m}_memory", {}).get("pass")]
    best = max(ok_modes, key=lambda m: thr[m]["tokens_per_s"]) if ok_modes else None
    proj = (float(core.n_tokens.sum()) / thr[best]["tokens_per_s"] / 60) if best else None
    check("projected_core_run_minutes", {"batch_mode": best, "minutes": proj, "core_unique_inputs": int(len(core)),
                                         "core_tokens": int(core.n_tokens.sum()), "throughput": thr},
          proj is not None and proj <= g["projected_core_minutes_max"],
          note="tokens / measured tokens-per-second at the fastest batch size that passed; excludes setup")
    gated = [v["pass"] for v in checks.values() if v["gated"]]
    return {"overall": "PASS" if all(gated) else "FAIL", "identity": identity, "build_id": B.build_id,
            "gates": g, "checks": checks, "state": status["state"]}


def analyze(B: Build, rows_dir: Path, out: Path, *, partial_smoke: bool = False, allow_stub: bool = False,
            n_boot: int = 2000) -> dict:
    from mad.analysis import commitment as CM

    man = json.loads((Path(rows_dir) / "run_manifest.json").read_text())
    if man["build_id"] != B.build_id:
        raise RunMismatch("these rows were produced from a different build")
    if man["identity"].get("backend") == "stub" and not allow_stub:
        raise RunMismatch("these rows come from the stub backend")
    rows = load_rows(rows_dir, B)
    rows = rows[rows["mode"] == "run"].drop_duplicates("item")
    wanted = select(B, "all")
    missing = sorted(set(wanted.item) - set(rows.item))
    out = Path(out)
    label = ""
    X = B.M.merge(rows[["item", "logprobs"]], on="item")
    if missing:
        if not partial_smoke:
            raise IncompleteRun(f"{len(missing)} of {len(wanted)} expected items are missing; final analysis "
                                "requires complete coverage (use --partial-smoke for a labelled smoke analysis)")
        label = "PARTIAL_SMOKE"
        std = X[X.variant.isin(STANDARD_FAMILIES)]
        full = std.groupby(KEY).fraction.nunique()
        keep = full[full == len(C.FRACS)].index
        X = X.set_index(KEY).loc[lambda d: d.index.isin(keep)].reset_index()
    out.mkdir(parents=True, exist_ok=True)
    R = B.R
    X = X.merge(R[KEY + ["answer", "prev_answer", "gold", "kind", "changed", "reasoning_names_final_option",
                         "lp_exposure", "lp_before_stated"]], on=KEY)
    X["cluster"] = X.pool + ":" + X.question_id.astype(str)
    am = np.array([int(np.argmax(v)) for v in X.logprobs])
    X["is_eventual"] = (am == X.answer).astype(float)
    X["is_previous"] = (am == X.prev_answer).astype(float)
    X["is_gold"] = (am == X.gold).astype(float)
    X["p_eventual"] = [float(np.exp(v[a])) for v, a in zip(X.logprobs, X.answer, strict=True)]
    curves = []
    for fam in STANDARD_FAMILIES:
        std = X[X.variant == fam]
        for pool_lab, gp in (("pilot", std[std.pool == "pilot"]), ("val", std[std.pool == "val"]), ("pooled", std)):
            for (cond, f), g in gp.groupby(["condition", "fraction"]):
                for sub, gs in CM.subsets(g):
                    for m in ("is_eventual", "is_gold", "is_previous", "p_eventual"):
                        curves.append({"variant": fam, "pool": pool_lab, "condition": cond, "fraction": f,
                                       "subset": sub, "metric": m, **CM.cluster_boot(gs, m, n_boot=n_boot, seed=3)})
    std = X[X.variant == "standard"]
    paired = {str(f): CM.paired_gap(std[std.fraction == f], "debate", "self_reconsider", "is_eventual", seed=4)
              for f in C.FRACS}
    rep = {}
    for name, variants, col in (("bridge_f0_vs_saved_exposure", ("bridge", "bridge_pre_opening"), "lp_exposure"),
                                ("exact_replay_vs_saved_before_stated_answer", ("exact_replay",), "lp_before_stated")):
        sub = X[X.variant.isin(variants) & ((X.fraction == 0.0) | X.fraction.isna())]
        pairs = [(a, json.loads(b)) for a, b in zip(sub.logprobs, sub[col], strict=True) if json.loads(b) is not None]
        rep[name] = _cmp([a for a, _ in pairs], [b for _, b in pairs])
    s1 = X[X.variant.isin(STANDARD_FAMILIES) & (X.fraction == 1.0)]
    rep["standard_f1_argmax_equals_stated"] = float(s1.is_eventual.mean()) if len(s1) else None
    tag = f"_{label}" if label else ""
    cols = ["variant", "pool", "condition", "fraction", "subset", "metric", "n", "questions", "est", "lo", "hi"]
    pd.DataFrame(curves, columns=cols).to_csv(out / f"curves{tag}.csv", index=False)
    res = {"label": label or "FINAL", "build_id": B.build_id, "identity": man["identity"],
           "responses": int(X.drop_duplicates(KEY).shape[0]), "missing_items": len(missing),
           "paired_debate_minus_self_changed_is_eventual": paired, "reproduction": rep}
    CM.to_json(res, out / f"analysis{tag}.json")
    return res
