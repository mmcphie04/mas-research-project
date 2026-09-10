"""Command line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from mad.config import load_config
from mad.data import (
    load_mc_dataset,
    load_mmlu_pro,
    make_splits,
    save_splits,
    stratified_sample,
    synthetic_questions,
)
from mad.debate import ROWS_FILE, run_debate


def _questions(cfg):
    if cfg.data.dataset == "synthetic":
        pool = synthetic_questions(max(cfg.data.n_questions * 2, 50), seed=cfg.data.sample_seed)
    elif "MMLU-Pro" in cfg.data.dataset:
        pool = load_mmlu_pro(cfg)
    else:
        pool = load_mc_dataset(cfg.data.dataset, cfg.data.dataset_revision, cfg.data.split)
    return stratified_sample(pool, cfg.data.n_questions, cfg.data.sample_seed, cfg.data.subjects)


def cmd_collect(args) -> None:
    from mad.collect import collect_states

    cfg = load_config(*args.config, **_overrides(args.set))
    questions = _questions(cfg)
    result = collect_states(cfg, questions, n_variants=args.variants)
    acc = result.rows["correct"].mean()
    print(f"{result.n_states} states from {len(questions)} questions x {args.variants} variants")
    print(f"accuracy {acc:.4f}   run dir: {result.run_dir}")


def _overrides(pairs):
    raw = dict(kv.split("=", 1) for kv in pairs or [])
    return {
        k: json.loads(v) if v[:1] in '[{"0123456789-' or v in ("true", "false", "null") else v
        for k, v in raw.items()
    }


def cmd_run(args) -> None:
    cfg = load_config(*args.config, **_overrides(args.set))
    questions = _questions(cfg)
    result = run_debate(cfg, questions)
    print(f"wrote {result.n_states} activation states and {len(result.rows)} agent-state rows")
    print(f"run dir: {result.run_dir}")


def cmd_splits(args) -> None:
    cfg = load_config(*args.config)
    questions = _questions(cfg)
    assignment = make_splits(questions, cfg.data.split_fracs, cfg.data.split_seed)
    path = save_splits(
        assignment,
        args.out,
        meta={
            "fracs": list(cfg.data.split_fracs),
            "seed": cfg.data.split_seed,
            "n": len(questions),
        },
    )
    counts = pd.Series(assignment).value_counts().to_dict()
    print(f"wrote {path}  {counts}")


def cmd_report(args) -> None:
    from mad.analysis.report import format_report, pilot_report, save_report

    frames = [pd.read_parquet(Path(r) / ROWS_FILE) for r in args.run]
    rows = pd.concat(frames, ignore_index=True)
    report = pilot_report(rows, probe_auroc=args.probe_auroc)
    print(format_report(report))
    if args.out:
        print(f"\nwrote {save_report(report, args.out)}")


def cmd_power(args) -> None:
    from mad.analysis.power_sim import auroc_curve, expected_labels, power_curve

    print("expected labels per", args.expect_n, "questions:")
    print(
        json.dumps(
            expected_labels(args.expect_n, args.p, args.r_c, args.r_i, args.s),
            indent=2,
            default=float,
        )
    )
    print()
    kw = {"p": args.p, "r_C": args.r_c, "r_I": args.r_i, "s": args.s}
    if args.by_auroc:
        curve = auroc_curve(args.by_auroc, n_questions=args.expect_n, n_reps=args.reps, **kw)
    else:
        curve = power_curve(args.n, probe_auroc=args.auroc, n_reps=args.reps, **kw)
    print(curve.to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mad", description="multi-agent debate revision probes")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run a debate collection")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--set", nargs="*", help="dotted overrides, e.g. data.n_questions=50")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("collect", help="readout-only state collection (no generation)")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--set", nargs="*")
    p.add_argument("--variants", type=int, default=3, help="permutation variants per question")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("splits", help="freeze the grouped train/val/test split")
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--out", default="configs/splits.json")
    p.set_defaults(func=cmd_splits)

    p = sub.add_parser("report", help="Stage-1 pilot report")
    p.add_argument("--run", nargs="+", required=True)
    p.add_argument("--probe-auroc", type=float, default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("power", help="synthetic power analysis")
    p.add_argument("--n", nargs="+", type=int, default=[300, 600, 900, 1500])
    p.add_argument("--auroc", type=float, default=0.65)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument(
        "--by-auroc",
        nargs="*",
        type=float,
        default=None,
        help="sweep probe AUROC at --expect-n instead of sweeping N",
    )
    p.add_argument("--expect-n", type=int, default=3000)
    p.add_argument("--p", type=float, default=0.42)
    p.add_argument("--r-c", type=float, default=0.14)
    p.add_argument("--r-i", type=float, default=0.28)
    p.add_argument("--s", type=float, default=0.45)
    p.set_defaults(func=cmd_power)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
