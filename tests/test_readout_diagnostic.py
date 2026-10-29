"""The fixed G1 verdict (scripts/readout_diagnostic.py: PASS / UNRESOLVED / FAIL)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("readout_diagnostic", ROOT / "scripts" / "readout_diagnostic.py")
RD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(RD)

BASE = np.array([0.6, 0.4, 0.0, 0.0])
T = [f"t{i}" for i in range(3)]


def rows(shift=None, rep_shift=0.0, d5_fixes=False, eager_fixes=False, drop=None):
    """shift: {(target, cond): delta} applied to fp32_head default reads."""
    shift = shift or {}
    out = []
    for t in T:
        for attn in ("default", "eager"):
            for head in ("fp32_head", "bf16_logits"):
                def add(cond, rep=-1, p=BASE):
                    out.append({"attn": attn, "head": head, "cond": cond, "rep": rep, "target": t, "p": p})
                add("D1_single", 0)
                add("D1_single", 1, BASE + np.array([rep_shift, -rep_shift, 0, 0]))
                for c in RD.D2_D4:
                    if drop == (t, c):
                        continue
                    dv = shift.get((t, c), 0.0) if head == "fp32_head" and (attn == "default" or not eager_fixes) else 0.0
                    add(c, p=BASE + np.array([dv, -dv, 0, 0]))
                for k in ("pad50", "pad300", "pad1500"):
                    dv = 0.0 if d5_fixes else shift.get((t, f"D4_partner_{k}"), 0.0)
                    add(f"D5_positions_{k}", p=BASE + np.array([dv, -dv, 0, 0]))
    return pd.DataFrame(out)


def test_pass_when_no_target_moves():
    assert RD.verdict(rows(), T)["verdict"] == "PASS"


def test_fail_on_an_unexplained_shift():
    v = RD.verdict(rows({("t1", "D2_equal_length_x4"): 0.2}), T)
    assert v["verdict"] == "FAIL" and v["targets_exceeding"] == 1


def test_unresolved_when_positions_or_eager_remove_a_padding_shift():
    s = {("t0", "D4_partner_pad300"): 0.2}
    assert RD.verdict(rows(s, d5_fixes=True), T)["verdict"] == "UNRESOLVED"
    assert RD.verdict(rows(s, eager_fixes=True), T)["verdict"] == "UNRESOLVED"
    assert RD.verdict(rows(s), T)["verdict"] == "FAIL"


def test_fail_on_unstable_single_reads_and_unresolved_on_missing_coverage():
    assert RD.verdict(rows(rep_shift=0.1), T)["verdict"] == "FAIL"
    assert RD.verdict(rows(drop=("t2", "D3_original_batch4")), T)["verdict"] == "UNRESOLVED"
