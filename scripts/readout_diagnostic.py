"""Targeted diagnostic of batch-dependent letter readouts (docs/NEXT_PHASE_CPU_RESULTS.md s11).

    python scripts/readout_diagnostic.py plan                      # CPU: fixed selection + batches
    python scripts/readout_diagnostic.py run  --out runs/next_commit/diag   # GPU (NOT run without approval)
    python scripts/readout_diagnostic.py report --out runs/next_commit/diag # CPU

Targets: the 12 worst sequences of the smoke discrepancy table and 8 stable controls (fixed by
``plan``, exact token ids from the core build). Every comparison changes ONE thing:

  D1 single x3        the target alone, three times            -> run-to-run variability
  D2 equal-length     [target x4]: batch shape 4, no padding    -> batch shape alone
  D3 original         the smoke's batch-of-4 composition        -> reproduce the original effect
  D4 co-batch         target with one partner: target longest (pad 0); target padded by ~50,
                      ~300 and ~1,500 tokens                    -> padding amount / batch-mate
  D5 positions        D4's padded cases with explicit position_ids from the mask
  D6 attention        D1 and D4 under attn_implementation="eager" (if the model supports a
                      runtime switch; skipped otherwise)
each under both readout heads (bf16 model logits, fp32 projection of the final hidden state).
Also writes an audit: eval mode, attention implementation, sliding window, soft-capping, logits
dtype, mask/pad conventions, option-letter token ids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mad import curve_runner as CR  # noqa: E402

PLAN = ROOT / "artifacts/next_phase/diagnostic/plan.json"
HEADS = ("bf16_logits", "fp32_head")


def plan(args) -> int:
    B = CR.load_build(ROOT / args.build)
    T = pd.read_csv(ROOT / "artifacts/next_phase/smoke/discrepancy_by_sequence.csv.gz")
    full = {h: i for i, h in zip(B.M.item, B.M.seq_hash, strict=True)}
    pick = lambda prefix: next(h for h in full if h.startswith(prefix))  # noqa: E731
    worst = [pick(h) for h in T.sort_values("worst_max_abs_dp", ascending=False).seq_hash.head(12)]
    stable = T[T.worst_max_abs_dp < 1e-3]
    controls = [pick(h) for h in stable.iloc[np.linspace(0, len(stable) - 1, 8).astype(int)].seq_hash]
    # the smoke's batch-of-4 compositions (reconstructed and verified in smoke_discrepancy.py)
    S = pd.read_csv(ROOT / args.smoke / "smoke_selection.csv")
    pend = S.drop_duplicates("seq_hash").sort_values("n_tokens", ascending=False)
    batches4 = [list(pend.seq_hash.iloc[i:i + 4]) for i in range(0, len(pend), 4)]
    original = {h: next(b for b in batches4 if h in b) for h in worst + controls}
    core = B.M[B.M.variant.isin(CR.CORE_VARIANTS)].drop_duplicates("seq_hash")
    length = B.M.drop_duplicates("seq_hash").set_index("seq_hash").n_tokens  # targets may be re-tokenized inputs

    def near(cand, n):
        return str(cand.iloc[(cand.n_tokens - n).abs().argsort().iloc[0]].seq_hash)

    partners = {}
    for t in worst + controls:
        L = int(length[t])
        cand = core[core.seq_hash != t]
        partners[t] = {"shorter": near(cand, max(L // 2, 50)), "pad50": near(cand, L + 50),
                       "pad300": near(cand, L + 300), "pad1500": near(cand, L + 1500)}
    out = {"targets": {"worst": worst, "stable": controls}, "original_batch4": original, "partners": partners,
           "build_id": B.build_id, "note": "fixed before any diagnostic GPU read"}
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(out, indent=1))
    n = sum(3 + 4 + 4 + 2 * 4 + 2 * 3 for _ in worst + controls) * len(HEADS)
    print(json.dumps({"targets": len(worst) + len(controls), "planned_passes_excluding_eager": n}, indent=1))
    return 0


def _jobs(P: dict) -> list[dict]:
    """(condition, target, batch of seq hashes, kwargs) for every planned read."""
    jobs = []
    for t in P["targets"]["worst"] + P["targets"]["stable"]:
        for r in range(3):
            jobs.append({"cond": "D1_single", "rep": r, "target": t, "batch": [t]})
        jobs.append({"cond": "D2_equal_length_x4", "target": t, "batch": [t] * 4})
        jobs.append({"cond": "D3_original_batch4", "target": t, "batch": P["original_batch4"][t]})
        for k, p in P["partners"][t].items():
            jobs.append({"cond": f"D4_partner_{k}", "target": t, "batch": [t, p]})
            if k != "shorter":
                jobs.append({"cond": f"D5_positions_{k}", "target": t, "batch": [t, p], "pos": True})
    return jobs


def run(args) -> int:
    from mad.backends.base import build_backend
    from mad.config import load_config

    P = json.loads(PLAN.read_text())
    B = CR.load_build(ROOT / args.build)
    if B.build_id != P["build_id"]:
        raise SystemExit("the plan was made for a different build")
    by = {h: i for i, h in zip(B.M.item, B.M.seq_hash, strict=True)}
    nopt = dict(zip(B.M.seq_hash, B.M.n_options, strict=True))
    seq = lambda h: B.seq(int(by[h]))  # noqa: E731
    if args.backend == "stub":
        from mad.backends.stub import StubBackend

        backend = StubBackend(load_config("configs/base.yaml", "configs/smoke_stub.yaml"), {})
    else:
        backend = build_backend(load_config("configs/base.yaml", args.config,
                                            **{"model.revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80"}))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = audit_backend(backend)
    (out / "audit.json").write_text(json.dumps(audit, indent=1, default=str))
    deadline = time.time() + 60 * args.max_minutes
    rows, attn_modes = [], ["default"]
    if audit.get("can_switch_attention") and not args.skip_eager:
        attn_modes.append("eager")
    for attn in attn_modes:
        if attn == "eager":
            backend.model.set_attn_implementation("eager")
        for job in _jobs(P):
            if attn == "eager" and not (job["cond"] == "D1_single" and job["rep"] == 0 or job["cond"].startswith("D4")):
                continue
            for head in HEADS:
                if time.time() > deadline:
                    break
                t = time.time()
                lp = backend.letter_logprobs_ids([seq(h) for h in job["batch"]], [nopt[h] for h in job["batch"]],
                                                 batch_size=len(job["batch"]), head=head,
                                                 position_ids_from_mask=job.get("pos", False))
                pos = job["batch"].index(job["target"])
                lens = [len(seq(h)) for h in job["batch"]]
                rows.append({**{k: v for k, v in job.items() if k != "batch"}, "attn": attn, "head": head,
                             "batch_hashes": ",".join(h[:12] for h in job["batch"]), "batch_lens": ",".join(map(str, lens)),
                             "target_pad": max(lens) - lens[pos], "logprobs": lp[pos].tolist(),
                             "seconds": round(time.time() - t, 3)})
        pd.DataFrame(rows).to_parquet(out / "diag_rows.parquet", index=False)  # flush after each mode
    (out / "status.json").write_text(json.dumps({"rows": len(rows), "complete": time.time() <= deadline,
                                                 "attn_modes": attn_modes}, indent=1))
    print(json.dumps({"rows": len(rows), "attn_modes": attn_modes}))
    return 0


def audit_backend(backend) -> dict:
    m = getattr(backend, "model", None)
    if m is None:
        return {"backend": "stub"}
    cfg = getattr(m.config, "text_config", m.config)
    tok = backend.tokenizer
    out = {"training": bool(m.training), "attn_implementation": getattr(m.config, "_attn_implementation", None),
           "text_attn_implementation": getattr(cfg, "_attn_implementation", None),
           "sliding_window": getattr(cfg, "sliding_window", None), "final_logit_softcapping": getattr(cfg, "final_logit_softcapping", None),
           "torch_dtype": str(next(m.parameters()).dtype), "padding_side_for_readout": "left (explicit in code)",
           "pad_token_id": tok.pad_token_id, "letter_ids": backend._letter_ids,
           "letters_decode": [tok.decode([i]) for i in backend._letter_ids],
           "can_switch_attention": hasattr(m, "set_attn_implementation")}
    import torch

    ids = torch.tensor([tok.encode("Answer: (", add_special_tokens=False)]).to(backend.device)
    with torch.no_grad():
        r = m(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1, use_cache=False, output_hidden_states=True)
    out["logits_dtype"] = str(r.logits.dtype)
    z = r.logits[0, -1, backend._letter_ids].float()
    h = backend.letter_logprobs_from_hidden(r.hidden_states[-1][0, -1, :], 10)
    out["letter_logit_scale"] = [float(z.min()), float(z.max())]
    out["bf16_logits_vs_fp32_head_max_abs_dlogp_on_probe"] = float(np.abs(torch.log_softmax(z, -1).cpu().numpy() - h).max())
    return out


RULE = "0 of 20 targets with max |dp| > 0.05 under fp32_head, D2-D4 vs D1 (default attention)"
THRESHOLD = 0.05
D2_D4 = ("D2_equal_length_x4", "D3_original_batch4", "D4_partner_shorter", "D4_partner_pad50",
         "D4_partner_pad300", "D4_partner_pad1500")


def verdict(D: pd.DataFrame, targets: list[str]) -> dict:
    """The fixed G1 decision (docs/NEXT_PHASE_CPU_RESULTS.md s13.1), from diagnostic rows with
    columns attn, head, cond, rep, target, p (probability vector).

    PASS        every target has D1 (rep 0) and all of D2-D4 under fp32_head / default attention,
                single reads repeat within THRESHOLD, and 0 of the targets exceed THRESHOLD.
    UNRESOLVED  coverage incomplete; or every exceedance disappears under explicit positions (D5)
                or eager attention (D6) -- a localized cause awaiting review.
    FAIL        any other exceedance, or single reads that do not repeat within THRESHOLD.
    """
    F = D[(D["head"] == "fp32_head") & (D.attn == "default")]
    d1 = F[(F.cond == "D1_single") & (F.rep == 0)].set_index("target").p
    missing = [t for t in targets if t not in d1.index
               or any(((F.cond == c) & (F.target == t)).sum() == 0 for c in D2_D4)]
    rep_dev = max([float(np.abs(r.p - d1[r.target]).max()) for r in F[(F.cond == "D1_single") & (F.rep > 0)].itertuples()
                   if r.target in d1.index] or [0.0])
    exceed = {}
    for r in F[F.cond.isin(D2_D4)].itertuples():
        if r.target in d1.index:
            dev = float(np.abs(r.p - d1[r.target]).max())
            if dev > THRESHOLD:
                exceed.setdefault(r.target, []).append((r.cond, round(dev, 4)))

    def gone(t, cond):  # does the same comparison stay within THRESHOLD under D5 or D6?
        k = cond.replace("D4_partner_", "")
        alts = []
        pos = D[(D["head"] == "fp32_head") & (D.attn == "default") & (D.cond == f"D5_positions_{k}") & (D.target == t)]
        eag = D[(D["head"] == "fp32_head") & (D.attn == "eager") & (D.cond == cond) & (D.target == t)]
        eag1 = D[(D["head"] == "fp32_head") & (D.attn == "eager") & (D.cond == "D1_single") & (D.rep == 0) & (D.target == t)]
        for r in pos.itertuples():
            alts.append(float(np.abs(r.p - d1[t]).max()) <= THRESHOLD)
        if len(eag) and len(eag1):
            alts.append(float(np.abs(eag.p.iloc[0] - eag1.p.iloc[0]).max()) <= THRESHOLD)
        return any(alts)

    explained = bool(exceed) and all(gone(t, c) for t, lst in exceed.items() for c, _ in lst if c.startswith("D4"))
    if missing:
        v = "UNRESOLVED"
    elif rep_dev > THRESHOLD:
        v = "FAIL"
    elif not exceed:
        v = "PASS"
    elif explained and all(c.startswith("D4") for lst in exceed.values() for c, _ in lst):
        v = "UNRESOLVED"
    else:
        v = "FAIL"
    return {"rule": RULE, "verdict": v, "targets": len(targets), "targets_exceeding": len(exceed),
            "exceedances": {t[:12]: lst for t, lst in exceed.items()}, "missing_targets": [t[:12] for t in missing],
            "single_read_repeat_max_abs_dp": rep_dev}


def report(args) -> int:
    out = Path(args.out)
    D = pd.read_parquet(out / "diag_rows.parquet")
    D["p"] = [np.exp(np.asarray(v)) for v in D.logprobs]
    D["rep"] = D["rep"].fillna(-1).astype(int) if "rep" in D else -1
    P = json.loads(PLAN.read_text())
    group = {h: "worst" for h in P["targets"]["worst"]} | {h: "stable" for h in P["targets"]["stable"]}
    base = D[(D.cond == "D1_single") & (D.rep == 0)].set_index(["attn", "head", "target"]).p
    res = []
    for (attn, head, cond), g in D.groupby(["attn", "head", "cond"]):
        for r in g.itertuples():
            ref = base.get((attn, head, r.target))
            if ref is None:
                continue
            d = r.p - ref
            res.append({"attn": attn, "head": head, "cond": cond, "group": group[r.target], "target": r.target[:12],
                        "pad": r.target_pad, "max_abs_dp": float(np.abs(d).max()), "tv": float(0.5 * np.abs(d).sum()),
                        "flip": bool(r.p.argmax() != ref.argmax())})
    R = pd.DataFrame(res)
    summ = R.groupby(["attn", "head", "cond", "group"]).agg(n=("tv", "size"), tv_median=("tv", "median"),
                                                          max_abs_dp_max=("max_abs_dp", "max"),
                                                          share_gt_005=("max_abs_dp", lambda x: float((x > 0.05).mean())),
                                                          flips=("flip", "sum")).reset_index()
    # head vs head at the same read
    hh = D.pivot_table(index=["attn", "cond", "target", "rep"], columns="head", values="p", aggfunc="first").dropna()
    head_gap = [float(np.abs(a - b).max()) for a, b in zip(hh["bf16_logits"], hh["fp32_head"], strict=True)] if len(hh) else []
    P_targets = P["targets"]["worst"] + P["targets"]["stable"]
    ver = verdict(D, P_targets)
    rep = {"verdict": ver, "summary": summ.round(6).to_dict("records"),
           "bf16_vs_fp32_head_same_read_max_abs_dp": {"median": float(np.median(head_gap)) if head_gap else None,
                                                      "max": float(np.max(head_gap)) if head_gap else None},
           "audit": json.loads((out / "audit.json").read_text())}
    (out / "diag_report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(summ.to_string(index=False))
    print(f"\nRULE: {ver['rule']}\nVERDICT: {ver['verdict']}  ({ver['targets_exceeding']} of {ver['targets']} "
          f"targets exceed; single-read repeat max |dp| {ver['single_read_repeat_max_abs_dp']:.2e}; "
          f"missing {len(ver['missing_targets'])})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--build", default="runs/next_commit/build_core")
    p.add_argument("--smoke", default="runs/next_commit/smoke")
    r = sub.add_parser("run")
    r.add_argument("--build", default="runs/next_commit/build_core")
    r.add_argument("--out", required=True)
    r.add_argument("--backend", choices=["hf", "stub"], default="hf")
    r.add_argument("--config", default="configs/colab_12b.yaml")
    r.add_argument("--max-minutes", type=float, default=20)
    r.add_argument("--skip-eager", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"plan": plan, "run": run, "report": report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
