"""Batched-vs-single discrepancies in the Stage 1 curve smoke (CPU, from the downloaded rows).

    python scripts/smoke_discrepancy.py [--smoke runs/next_commit/smoke] [--build runs/next_commit/build_core]

The smoke rows do not record batch composition, so it is RECONSTRUCTED exactly as the runner
formed it (pending inputs deduplicated by sequence, sorted by length descending, consecutive
batches) and VERIFIED against the per-batch token sums and maximum lengths the runner logged.
For every unique input sequence and every batched mode it reports, against the single read:
max |dp|, total-variation distance, max |d log p| and the probability of the letter where that
occurs, top-two margins, argmax flips, and changes in p(eventual answer) and p(correct answer);
plus the left-padding amount and batch-mates. Item-level summaries are reported too (inputs
shared by several items count once per item there). Writes artifacts/next_phase/smoke/discrepancy_*.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import curve_runner as CR  # noqa: E402

OUT = ROOT / "artifacts/next_phase/smoke"
LETTERS = "ABCDEFGHIJ"


def reconstruct(B, S: pd.DataFrame, status: dict, mode: str, bs: int) -> pd.DataFrame:
    """Batch index, position, pad and batch-mates of every input read in ``mode``."""
    pend = S.drop_duplicates("seq_hash").sort_values("n_tokens", ascending=False)
    recs, logged = [], [b for b in status["batches"] if b.get("mode") == mode and "seconds" in b]
    for bi, start in enumerate(range(0, len(pend), bs)):
        ch = pend.iloc[start: start + bs]
        width = int(ch.n_tokens.max())
        if bi < len(logged):
            lg = logged[bi]
            if lg["n"] != len(ch) or lg["tokens"] != int(ch.n_tokens.sum()) or lg["max_len"] != width:
                raise SystemExit(f"{mode} batch {bi}: reconstruction does not match the logged batch")
        for r in ch.itertuples():
            recs.append({"seq_hash": r.seq_hash, f"{mode}_batch": bi, f"{mode}_pad": width - int(r.n_tokens),
                         f"{mode}_batch_lens": ",".join(map(str, ch.n_tokens.tolist()))})
    if len(logged) != -(-len(pend) // bs):
        raise SystemExit(f"{mode}: {len(logged)} logged batches, {-(-len(pend) // bs)} reconstructed")
    return pd.DataFrame(recs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="runs/next_commit/smoke")
    ap.add_argument("--build", default="runs/next_commit/build_core")
    a = ap.parse_args(argv)
    B = CR.load_build(ROOT / a.build)
    sm = ROOT / a.smoke
    rows = CR.load_rows(sm, B)
    status = json.loads((sm / "run_status.json").read_text())
    S = pd.read_csv(sm / "smoke_selection.csv", dtype={"question_id": str})
    meta = B.M.set_index("item")
    R = B.R.set_index(CR.KEY)
    modes = ["single", "batch4", "batch8", "batch16"]
    sizes = {"single": 1, "batch4": 4, "batch8": 8, "batch16": 16}
    # one probability vector per (mode, unique sequence): the row actually READ (not a copy)
    read = rows[rows.get("copied_from_identical_input").isna()] if "copied_from_identical_input" in rows else rows
    P = {m: {h: np.exp(np.asarray(v, float)) for h, v in zip(g.seq_hash, g.logprobs, strict=True)}
         for m, g in read.groupby("mode")}
    comp = None
    for m in modes[1:]:
        c = reconstruct(B, S, status, m, sizes[m])
        comp = c if comp is None else comp.merge(c, on="seq_hash", how="outer")
    # per unique sequence: identity from its first item, and all items sharing it
    U = S.sort_values("item").groupby("seq_hash").agg(item=("item", "first"), n_items=("item", "size"),
                                                         items=("item", lambda x: ",".join(map(str, x))),
                                                         why=("why", lambda x: ",".join(sorted(set(x))))).reset_index()
    U = U.merge(comp, on="seq_hash", how="left")
    rec = []
    for r in U.itertuples():
        m0 = meta.loc[r.item]
        key = tuple(m0[CR.KEY])
        resp = R.loc[key]
        ps = P["single"][r.seq_hash]
        top2 = np.sort(ps)[::-1][:2]
        row = {"seq_hash": r.seq_hash[:12], "item": r.item, "n_items_sharing": r.n_items, "items": r.items, "why": r.why,
               **{k: m0[k] for k in CR.KEY}, "variant": m0.variant, "fraction": m0.fraction, "n_tokens": int(m0.n_tokens),
               "eventual": LETTERS[int(resp.answer)], "gold": LETTERS[int(resp.gold)],
               "p_single": " ".join(f"{x:.4f}" for x in ps), "single_top_margin": float(top2[0] - top2[1]),
               "single_argmax": LETTERS[int(ps.argmax())]}
        for m in modes[1:]:
            pb = P[m][r.seq_hash]
            d = pb - ps
            dl = np.abs(np.log(np.maximum(pb, 1e-30)) - np.log(np.maximum(ps, 1e-30)))
            row.update({f"{m}_p": " ".join(f"{x:.4f}" for x in pb), f"{m}_max_abs_dp": float(np.abs(d).max()),
                        f"{m}_tv": float(0.5 * np.abs(d).sum()), f"{m}_max_abs_dlogp": float(dl.max()),
                        f"{m}_p_at_max_dlogp": float(ps[dl.argmax()]),
                        f"{m}_argmax_flip": bool(pb.argmax() != ps.argmax()),
                        f"{m}_dp_eventual": float(d[int(resp.answer)]), f"{m}_dp_gold": float(d[int(resp.gold)]),
                        f"{m}_pad": getattr(r, f"{m}_pad"), f"{m}_batch_lens": getattr(r, f"{m}_batch_lens")})
        rec.append(row)
    T = pd.DataFrame(rec)
    T["worst_max_abs_dp"] = T[[f"{m}_max_abs_dp" for m in modes[1:]]].max(1)
    T = T.sort_values("worst_max_abs_dp", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / "discrepancy_by_sequence.csv.gz", index=False)

    def summ(df, w=None):
        out = {}
        for m in modes[1:]:
            x = df[f"{m}_max_abs_dp"]
            wt = np.ones(len(df)) if w is None else w

            def share(c, wt=wt):
                return float(np.average(c, weights=wt))
            out[m] = {"n": int(wt.sum()), "share_max_abs_dp_gt_0.01": share(x > 0.01), "share_gt_0.05": share(x > 0.05),
                      "share_gt_0.10": share(x > 0.10), "max_abs_dp_max": float(x.max()),
                      "tv_median": float(df[f"{m}_tv"].median()), "tv_max": float(df[f"{m}_tv"].max()),
                      "argmax_flips": int((df[f"{m}_argmax_flip"] * wt).sum()),
                      "share_largest_dlogp_on_letter_p_lt_1e-3": share(df[f"{m}_p_at_max_dlogp"] < 1e-3),
                      "abs_dp_eventual_max": float(df[f"{m}_dp_eventual"].abs().max()),
                      "abs_dp_gold_max": float(df[f"{m}_dp_gold"].abs().max())}
        return out

    summary = {"unique_sequences": summ(T), "items": summ(T, T.n_items_sharing.to_numpy(float)),
               "reconstruction": "batch compositions reconstructed and verified against every logged batch"}
    # concentration: by variant/fraction, length tercile, pad amount
    T["length_bin"] = pd.qcut(T.n_tokens, 3, labels=["short", "middle", "long"])
    T["vf"] = T.variant + "@" + T.fraction.astype(str)
    conc = {}
    for m in modes[1:]:
        T[f"{m}_pad_bin"] = pd.cut(T[f"{m}_pad"], [-1, 0, 100, 400, 10_000], labels=["0", "1-100", "101-400", ">400"])
        conc[m] = {by: T.groupby(by, observed=True)[f"{m}_max_abs_dp"].agg(
            n="size", share_gt_005=lambda x: float((x > 0.05).mean()), max="max").round(4).to_dict("index")
            for by in ("vf", "length_bin", f"{m}_pad_bin")}
    summary["concentration"] = conc
    # agreement between batched modes themselves
    summary["between_batched_modes_share_max_abs_dp_gt_0.05"] = {
        f"{a}_vs_{b}": float(np.mean([np.abs(P[a][h] - P[b][h]).max() > 0.05 for h in U.seq_hash]))
        for a, b in (("batch4", "batch8"), ("batch8", "batch16"), ("batch4", "batch16"))}
    # effect on the curve metrics over the 40 fully covered responses (standard inputs)
    full = S[S.why == "stratified_response"]
    std = full[full.variant == "standard"].merge(B.R[CR.KEY + ["answer", "prev_answer", "gold", "changed"]], on=CR.KEY)
    eff = {}
    for m in modes:
        am = np.array([int(P[m][h].argmax()) for h in std.seq_hash])
        pe = np.array([P[m][h][a] for h, a in zip(std.seq_hash, std.answer, strict=True)])
        g = std.assign(is_eventual=(am == std.answer).astype(float), is_gold=(am == std.gold).astype(float), p_eventual=pe)
        eff[m] = g.groupby("fraction")[["is_eventual", "is_gold", "p_eventual"]].mean().round(4).to_dict("index")
    summary["curve_metrics_on_40_smoke_responses"] = {
        "note": "means over the 40 stratified smoke responses only; not an estimate for the full set", **eff}
    worst = T.head(15)
    stable = T[T.worst_max_abs_dp < 1e-3]
    controls = stable.iloc[np.linspace(0, len(stable) - 1, min(15, len(stable))).astype(int)] if len(stable) else stable
    pd.concat([worst.assign(group="worst"), controls.assign(group="stable_control")]).to_csv(
        OUT / "discrepancy_worst_and_controls.csv", index=False)
    (OUT / "discrepancy_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("unique_sequences", "items", "between_batched_modes_share_max_abs_dp_gt_0.05")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
