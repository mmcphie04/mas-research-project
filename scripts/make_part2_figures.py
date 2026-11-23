"""Part 2 figures (Experiments A and B), regenerated from committed artifacts only, in the style of
the midterm figures (branch ``midterm-figures``, scripts/make_midterm_figures.py).

    uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/make_part2_figures.py

(matplotlib is not a project dependency; the command above uses a throwaway environment.)

Reads only the committed analysis outputs in SOURCES (no runs/, no activations, no GPU). Writes
figures/part2/fig{A1,A2,A3,B1,B2,B3}_*.pdf (vector) and .png (200 dpi preview), and
figures/part2/README.md (per figure: sources, every plotted value with its interval, a draft caption).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "artifacts/expab/analysis"
SOURCES = {
    "expa": AN / "expa_results.json",
    "expb": AN / "expb_results.json",
    "panel": AN / "expb_panel_effect.json",
}

# The midterm palette (same hexes). Experiment A colors by CHOOSER MODEL; Experiment B by ARM
# (peer arms in the debate blue, reconsider-alone in the reconsideration orange, as in the midterm).
C = {
    "gemma": "#199e70",
    "qwen_nt": "#3b3b98",
    "qwen_t": "#b8429a",
    "peer": "#2a78d6",
    "alone": "#eb6834",
    "neutral": "#898781",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
WIDE, NARROW = 6.5, 3.2

# Experiment A arms: (chooser model, pool kind, label "chooser <- candidates")
ARMS = {
    "A1_qt5": ("qwen_t", "own", "Thinking ← own ×5"),
    "A1_qnt5": ("qwen_nt", "own", "Non-thinking ← own ×5"),
    "A1_qnt10": ("qwen_nt", "own", "Non-thinking ← own ×10"),
    "A1_g5": ("gemma", "own", "Gemma ← own ×5"),
    "A1_g10": ("gemma", "own", "Gemma ← own ×10"),
    "A2_5": ("qwen_t", "other", "Thinking ← non-thinking ×5"),
    "A2_10": ("qwen_t", "other", "Thinking ← non-thinking ×10"),
    "A3": ("qwen_t", "mixed", "Thinking ← 5 Gemma + 5 non-thinking"),
}
MODE_LABEL = {"gemma": "Gemma-3-12B", "qwen_nt": "Qwen3-8B non-thinking", "qwen_t": "Qwen3-8B thinking"}
B_ARMS = [("debate", "Debate"), ("dissent", "Dissent"), ("anon", "Anon"), ("alone", "Alone")]
B_MODES = [("qwen_nt", "Non-thinking"), ("qwen_t", "Thinking")]


CAPTIONS = {
    "figA1_choosing_gains": (
        "Experiment A, pre-registered (held-out, 2,900 MMLU-Pro questions). Each row is a chooser arm: a model (color) reads "
        "a pool of candidate answers and picks one; three candidate orderings, plurality of the picks; the chooser is called only "
        "when the pool disagrees. (a) Accuracy gain over the plurality vote of the same pool (P1). (b) Gain over the chooser model "
        "voting with its own samples at the same measured GPU time (P2). Points: means; bars: 95% paired question-bootstrap "
        "intervals. Hollow: the chooser picks among its own samples. The headline arm (A2_10) was fixed as the largest development "
        "P1 gain before held-out was scored. In (b), questions where the arm costs more than the chooser's 10-sample voting curve "
        "are compared with that curve's 10-sample value (A2_10: 1,344 of 2,900 questions; A1_qnt10 and A1_g10: all), so for "
        "A1_qnt10 and A1_g10 panel (b) equals panel (a). Eight arms × three primary measures, no multiplicity correction."),
    "figA2_recognize_vs_produce": (
        "Experiment A, pre-registered P3, the verification–generation gap (held-out). On questions whose candidates include both "
        "the right answer and a wrong one: accuracy of the chooser's pick (recognize, solid) against the chooser model's own "
        "expected single-answer accuracy on the same questions, from its bank samples (produce, hatched). Bars: 95% paired "
        "question-bootstrap intervals. Right: recognize − produce in percentage points with its interval; bold where the interval "
        "excludes zero."),
    "figA3_selfvoting_curves": (
        "Experiment A (held-out). Lines: each model voting with itself — exact expected plurality accuracy of 1–10 of its own "
        "samples (all subsets), plotted at n times its measured GPU-seconds per sample (no intervals). Points: chooser arms at "
        "their mean measured GPU-seconds per question (the pool's samples plus three chooser calls when the pool disagrees), with "
        "95% question-bootstrap intervals; color = chooser model, shape = whose samples it picks among. Arms to the right of their "
        "chooser's curve are compared with the curve's end value in figure A1(b)."),
    "figB1_helpful_harmful": (
        "Experiment B (held-out full run; Qwen3-8B, one revision round, disputed 3-agent panels). Share of agents whose revision "
        "turned a wrong answer right (up) or a right answer wrong (down, hatched); numbers are the pre-registered counts. Peer arms "
        "(blue) read the other two panel members' answers: debate (labelled by panel member, fixed order), dissent (plus an "
        "instruction to change only on finding a specific error), anon (labelled Response A/B, order shuffled). Alone (orange) "
        "reconsiders without peers. Peer arms: 440 non-thinking / 393 thinking panels; alone: 2,353 / 1,539 (the same panels plus "
        "more). Bars: 95% question-bootstrap intervals (descriptive; shares among agents with a parsed revision)."),
    "figB2_answer_prediction": (
        "Experiment B, pre-registered answer-prediction curves (held-out; thinking mode). Each revision's thinking trace is cut "
        "after 0–100% of its tokens and closed, and the answer readout is taken; the y-axis is the probability on the agent's final "
        "stated answer. Solid: agents who changed their answer; dashed: agents who kept it. Bands: 95% question-bootstrap "
        "intervals (descriptive; the plan specifies means). In debate the final answer is usually one already shown in the peers' "
        "text, which by itself can raise the readout at 0%."),
    "figB3_order_effect": (
        "Experiment B, pre-registered order effect (held-out). Among changed agents whose two shown peers gave different answers "
        "and whose new answer is one of them: the share that took the answer of the peer shown last. Points: means; bars: 95% "
        "question-bootstrap intervals; dashed line: no order preference. Debate and dissent show the peers in a fixed order; anon "
        "shuffles the order per agent, so its share reflects position rather than which panel member spoke."),
}

HEADER = """# Part 2 figures (Experiments A and B)

Generated by `scripts/make_part2_figures.py`; do not edit by hand. Regenerate from the repo root with

```
uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/make_part2_figures.py
```

(matplotlib is not a project dependency, so this runs in a throwaway environment.) The script reads only the committed analysis
outputs in `artifacts/expab/analysis/` (`expa_results.json`, `expb_results.json`, `expb_panel_effect.json`); no activations,
nothing under `runs/`, no GPU. Every value in the tables below is read from those files, and the figures plot the same values.
Each figure is a vector PDF plus a 200-dpi PNG, cropped to exactly 6.5 in or 3.2 in wide (9 pt text at that width), in the
style of the midterm figures (branch `midterm-figures`).

**Colors** use the midterm palette's hexes. Experiment A colors by *chooser model*: Gemma-3-12B `#199e70`, Qwen3-8B non-thinking
`#3b3b98`, Qwen3-8B thinking `#b8429a`. Experiment B colors by *arm*: the peer arms (debate, dissent, anon) in the debate blue
`#2a78d6` and reconsider-alone in the reconsideration orange `#eb6834`, as in the midterm; modes are separate panels or marker
groups. The two mappings never share a figure.

"Pre-registered" means fixed in `docs/EXPAB_PLAN.md` before the data were analysed; intervals marked descriptive were added for
the figures (`scripts/expb_panel_effect.py`, exploratory).

---
"""


def tint(hex_color: str, amount: float) -> str:
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * amount):02x}" for c in rgb)


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": C["axis"], "axes.linewidth": 0.8, "axes.labelcolor": C["ink"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.grid": False, "grid.color": C["grid"], "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5,
        "hatch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 200,
    })


def ygrid(ax, axis="y") -> None:
    ax.grid(axis=axis)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str, width: float) -> None:
    """Tight crop exactly `width` inches wide, so fonts print at their set size (as the midterm)."""
    pad = 0.02
    for _ in range(6):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        miss = width - (bb.width + 2 * pad)
        if abs(miss) < 0.005:
            break
        fig.set_size_inches(fig.get_figwidth() + miss, fig.get_figheight())
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    crop = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    fig.savefig(out / f"{name}.pdf", bbox_inches=crop, metadata={"CreationDate": None})
    fig.savefig(out / f"{name}.png", bbox_inches=crop, dpi=200)
    plt.close(fig)


def load(key: str):
    return json.loads(SOURCES[key].read_text())


def ci(d: dict, k: int = 3) -> str:
    return f"{d['mean']:.{k}f} [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"


def err(d: dict) -> list[list[float]]:
    return [[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]]


# ---------------------------------------------------------------------------------------------
def figA1(out: Path, md: list[str]) -> None:
    r = load("expa")
    h, head = r["heldout"], r["headline_arm"]
    order = ["A2_10", "A2_5", "A3", "A1_qt5", "A1_qnt5", "A1_qnt10", "A1_g5", "A1_g10"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 3.9), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, key, title in ((a1, "P1_minus_pool_plurality", "(a) Gain over its pool's\nplurality vote"),
                           (a2, "P2_minus_sc_matched_seconds", "(b) Gain over self-voting\nat equal GPU time")):
        for y, arm in enumerate(order):
            d, col = h[arm][key], C[ARMS[arm][0]]
            hollow = ARMS[arm][1] == "own"
            ax.errorbar([100 * d["mean"]], [y], xerr=[[100 * (d["mean"] - d["lo"])], [100 * (d["hi"] - d["mean"])]],
                        color=col, marker="o", mfc="white" if hollow else col, mec=col, mew=1.3, ms=6,
                        elinewidth=1.0, capsize=0, ls="none")
        ax.axvline(0, color=C["ink"], lw=0.8)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Percentage points")
        ygrid(ax, "x")
    a1.set_yticks(range(len(order)), [f"{a}" + ("  (headline)" if a == head else "") + f"\n{ARMS[a][2]}" for a in order])
    a1.invert_yaxis()
    for lab in a1.get_yticklabels():
        if "(headline)" in lab.get_text():
            lab.set_fontweight("bold")
    for ax in (a1, a2):
        ax.set_xlim(-6, 7)
    handles = [Line2D([], [], color=C[m], marker="o", ls="none", ms=6, label=f"chooser: {MODE_LABEL[m]}")
               for m in ("qwen_t", "qwen_nt", "gemma")]
    handles.append(Line2D([], [], color=C["ink2"], marker="o", mfc="white", ls="none", ms=6, label="hollow: picks among its own samples"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, handletextpad=0.3, columnspacing=1.2)
    save(fig, out, "figA1_choosing_gains", WIDE)

    md += ["## figA1_choosing_gains", "", "Held-out (2,900 questions), 95% paired question bootstrap (10,000 resamples, seed 0). "
           "Source: `artifacts/expab/analysis/expa_results.json` (`heldout.<arm>.P1_minus_pool_plurality`, "
           "`heldout.<arm>.P2_minus_sc_matched_seconds`, `P2_flags`). Plotted in percentage points.", "",
           "| arm | P1 (vs pool vote) | P2 (vs self-voting, matched GPU-seconds) | P2 questions beyond the 10-sample curve |",
           "|---|---|---|---|"]
    for arm in order:
        v = h[arm]
        beyond = v["P2_flags"]["seconds"].get("beyond_curve", 0)
        md.append(f"| {arm}{' (headline)' if arm == head else ''} | {ci(v['P1_minus_pool_plurality'], 4)} | "
                  f"{ci(v['P2_minus_sc_matched_seconds'], 4)} | {beyond} of {v['questions']} |")
    md.append("")


def figA2(out: Path, md: list[str]) -> None:
    h = load("expa")["heldout"]
    order = ["A1_qt5", "A2_5", "A2_10", "A3", "A1_qnt5", "A1_qnt10", "A1_g5", "A1_g10"]
    fig, ax = plt.subplots(figsize=(WIDE, 4.0))
    w = 0.36
    for i, arm in enumerate(order):
        p, col = h[arm]["P3_gap"], C[ARMS[arm][0]]
        rec, prod, dif = p["chooser_accuracy"], p["own_single_accuracy"], p["difference"]
        for y, d, fc, hatch in ((i - w / 2 - 0.01, rec, col, None), (i + w / 2 + 0.01, prod, tint(col, 0.6), "////")):
            ax.barh(y, d["mean"], height=w, color=fc, lw=0, hatch=hatch, edgecolor=col if hatch else None)
            ax.errorbar([d["mean"]], [y], xerr=err(d), color=C["ink"], elinewidth=0.9, capsize=0, ls="none")
        sig = dif["lo"] > 0 or dif["hi"] < 0
        ax.text(0.82, i, f"{100 * dif['mean']:+.1f} pts [{100 * dif['lo']:+.1f}, {100 * dif['hi']:+.1f}]",
                va="center", fontsize=8, color=C["ink"] if sig else C["ink2"], fontweight="bold" if sig else "normal")
    ax.text(0.82, -0.85, "recognize − produce", fontsize=8, color=C["ink2"], va="center")
    ax.set_yticks(range(len(order)), [f"{a}\n{ARMS[a][2]}" for a in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("Accuracy on questions whose candidates include a right and a wrong answer")
    ygrid(ax, "x")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["ink2"], label="recognize: picks the right candidate"),
                       plt.Rectangle((0, 0), 1, 1, fc=tint(C["ink2"], 0.6), ec=C["ink2"], hatch="////", lw=0,
                                     label="produce: the chooser model's own single answer")],
              loc="lower center", bbox_to_anchor=(0.4, 1.02), ncol=1, handlelength=1.2, borderaxespad=0.2)
    save(fig, out, "figA2_recognize_vs_produce", WIDE)

    md += ["## figA2_recognize_vs_produce", "", "P3 (verification–generation gap), held-out questions whose candidates include "
           "the correct answer and a wrong one; 95% paired question bootstrap. Source: `expa_results.json` (`heldout.<arm>.P3_gap`).", "",
           "| arm | questions | recognize (chooser accuracy) | produce (own single-answer accuracy) | difference |", "|---|---|---|---|---|"]
    for arm in order:
        p = h[arm]["P3_gap"]
        md.append(f"| {arm} | {p['questions']} | {ci(p['chooser_accuracy'])} | {ci(p['own_single_accuracy'])} | {ci(p['difference'])} |")
    md.append("")


def figA3(out: Path, md: list[str]) -> None:
    r = load("expa")
    sc, h = r["self_consistency_heldout"], r["heldout"]
    fig, ax = plt.subplots(figsize=(WIDE, 3.3))
    for m in ("gemma", "qwen_nt", "qwen_t"):
        s = sc[m]
        x = [n * s["seconds_per_sample"] for n in s["n"]]
        ax.plot(x, s["accuracy"], color=C[m], lw=1.6, marker=".", ms=5, zorder=2)
        ax.text(x[0] / 1.12, s["accuracy"][0], f"{MODE_LABEL[m]}\nvoting with itself, 1–10", color=C[m], fontsize=8,
                va="center", ha="right", linespacing=1.0)
    marker = {"own": "o", "other": "s", "mixed": "D"}
    for arm, (m, pool, _lab) in ARMS.items():
        v = h[arm]
        x, acc = v["mean_cost"]["seconds"], v["accuracy"]
        ax.errorbar([x], [acc["mean"]], yerr=err(acc), color=C[m], marker=marker[pool], mfc="white" if pool == "own" else C[m],
                    mec=C[m], mew=1.3, ms=6.5, elinewidth=0.9, capsize=0, ls="none", zorder=3)
        dx, dy, ha = {"A2_5": (-7, -3, "right"), "A1_qnt5": (-7, 5, "right"), "A3": (7, -10, "left"),
                      "A1_qnt10": (6, -9, "left"), "A1_g10": (6, -9, "left")}.get(arm, (6, 3, "left"))
        ax.annotate(arm, (x, acc["mean"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=C["ink2"], ha=ha)
    ax.set_xscale("log")
    ax.set_xlabel("Measured GPU-seconds per question (log scale)")
    ax.set_ylabel("Held-out accuracy")
    ax.set_xlim(0.012, 10)
    ygrid(ax)
    handles = [Line2D([], [], color=C["ink2"], marker="o", mfc="white", ls="none", ms=6, label="chooser picks among its own samples"),
               Line2D([], [], color=C["ink2"], marker="s", ls="none", ms=6, label="among non-thinking samples"),
               Line2D([], [], color=C["ink2"], marker="D", ls="none", ms=6, label="among 5 Gemma + 5 non-thinking")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.01), ncol=2, handletextpad=0.3, columnspacing=1.0)
    save(fig, out, "figA3_selfvoting_curves", WIDE)

    md += ["## figA3_selfvoting_curves", "", "Self-voting curves: exact expected plurality accuracy of n of the mode's bank samples "
           "(all subsets), held-out, at n × the measured GPU-seconds per sample (no intervals in the source). Chooser arms: "
           "held-out accuracy (95% question bootstrap) at their mean measured GPU-seconds per question. "
           "Source: `expa_results.json` (`self_consistency_heldout`, `heldout.<arm>.accuracy`, `heldout.<arm>.mean_cost`).", "",
           "| mode | s/sample | n=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ("gemma", "qwen_nt", "qwen_t"):
        s = sc[m]
        md.append(f"| {m} | {s['seconds_per_sample']:.3f} | " + " | ".join(f"{a:.4f}" for a in s["accuracy"]) + " |")
    md += ["", "| arm | GPU-s per question | accuracy |", "|---|---|---|"]
    for arm in ARMS:
        md.append(f"| {arm} | {h[arm]['mean_cost']['seconds']:.3f} | {ci(h[arm]['accuracy'], 4)} |")
    md.append("")


# ---------------------------------------------------------------------------------------------
def figB1(out: Path, md: list[str]) -> None:
    rates, summ = load("panel")["agent_rates"], load("expb")["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (mode, mlab) in zip(axes, B_MODES, strict=True):
        for i, (arm, _) in enumerate(B_ARMS):
            col = C["alone"] if arm == "alone" else C["peer"]
            hp, hm = rates[f"{mode} {arm}"]["helpful"], rates[f"{mode} {arm}"]["harmful"]
            ax.bar(i, 100 * hp["mean"], width=0.66, color=col, lw=0)
            ax.bar(i, -100 * hm["mean"], width=0.66, color=tint(col, 0.55), lw=0, hatch="////", edgecolor=col)
            ax.errorbar([i], [100 * hp["mean"]], yerr=[[100 * (hp["mean"] - hp["lo"])], [100 * (hp["hi"] - hp["mean"])]],
                        color=C["ink"], elinewidth=0.9, capsize=0, ls="none")
            ax.errorbar([i], [-100 * hm["mean"]], yerr=[[100 * (hm["hi"] - hm["mean"])], [100 * (hm["mean"] - hm["lo"])]],
                        color=C["ink"], elinewidth=0.9, capsize=0, ls="none")
            s = summ[f"{mode} {arm}"]
            ax.text(i, 100 * hp["hi"] + 0.8, str(s["helpful"]), ha="center", va="bottom", fontsize=8, color=C["ink"])
            ax.text(i, -100 * hm["hi"] - 0.8, str(s["harmful"]), ha="center", va="top", fontsize=8, color=C["ink"])
        ax.axhline(0, color=C["ink"], lw=0.8)
        ax.set_xticks(range(len(B_ARMS)), [lab for _, lab in B_ARMS])
        ax.tick_params(axis="x", length=0)
        ax.set_title(f"({'ab'[B_MODES.index((mode, mlab))]}) {mlab}", fontsize=9, loc="left")
        ygrid(ax)
    axes[0].set_ylim(-17, 26)
    axes[0].set_yticks([-15, -10, -5, 0, 5, 10, 15, 20, 25], ["15", "10", "5", "0", "5", "10", "15", "20", "25"])
    axes[0].set_ylabel("Share of agents (%)\n← harmful | helpful →")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["peer"], label="peer arms (read the other two)"),
                        plt.Rectangle((0, 0), 1, 1, color=C["alone"], label="reconsider alone"),
                        plt.Rectangle((0, 0), 1, 1, fc=tint(C["ink2"], 0.55), ec=C["ink2"], hatch="////", lw=0,
                                      label="harmful (lighter, hatched)")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, borderaxespad=0)
    save(fig, out, "figB1_helpful_harmful", WIDE)

    md += ["## figB1_helpful_harmful", "", "Held-out full run. Bars: share of agents with a parsed revision that changed wrong→right "
           "(helpful) or right→wrong (harmful), 95% question bootstrap (`expb_panel_effect.json` `agent_rates`, descriptive). "
           "Counts above/below the bars: `expb_results.json` `summary`.", "",
           "| mode × arm | agents | helpful (count) | harmful (count) | helpful share | harmful share |", "|---|---|---|---|---|---|"]
    for mode, _ in B_MODES:
        for arm, _ in B_ARMS:
            s, rr = summ[f"{mode} {arm}"], rates[f"{mode} {arm}"]
            md.append(f"| {mode} {arm} | {s['agents']} | {s['helpful']} | {s['harmful']} | {ci(rr['helpful'])} | {ci(rr['harmful'])} |")
    md.append("")


def figB2(out: Path, md: list[str]) -> None:
    cv = load("panel")["thinking_curve_intervals"]
    fig, ax = plt.subplots(figsize=(NARROW, 2.9))
    for arm, col in (("debate", C["peer"]), ("alone", C["alone"])):
        for kind, ls in (("changed", "-"), ("unchanged", "--")):
            d = cv[f"qwen_t {arm} {kind}"]
            x, ys = d["fractions"], d["p_final_answer"]
            ax.fill_between(x, [y["lo"] for y in ys], [y["hi"] for y in ys], color=col, alpha=0.13, lw=0)
            ax.plot(x, [y["mean"] for y in ys], color=col, ls=ls, marker="o", ms=3.5)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of the thinking trace read")
    ax.set_ylabel("Probability on the final answer")
    ax.set_ylim(0, 1.02)
    ygrid(ax)
    handles = [Line2D([], [], color=C["peer"], lw=1.6, label="Debate"), Line2D([], [], color=C["alone"], lw=1.6, label="Reconsider alone"),
               Line2D([], [], color=C["ink2"], lw=1.6, label="changed answer"),
               Line2D([], [], color=C["ink2"], lw=1.6, ls="--", label="kept answer")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=2, borderaxespad=0, handlelength=1.8)
    save(fig, out, "figB2_answer_prediction", NARROW)

    md += ["## figB2_answer_prediction", "", "Qwen3-8B thinking, held-out full run. The trace is cut at each share of its tokens, closed, "
           "and read out; probability on the agent's final stated answer; changed = helpful + harmful + wrong→wrong. 95% question "
           "bootstrap (`expb_panel_effect.json` `thinking_curve_intervals`, descriptive; the pre-registered by-kind means are in "
           "`expb_results.json` `curves`).", "",
           "| arm, agents | n | 0% | 10% | 25% | 50% | 75% | 100% |", "|---|---|---|---|---|---|---|---|"]
    for arm in ("debate", "alone"):
        for kind in ("changed", "unchanged"):
            d = cv[f"qwen_t {arm} {kind}"]
            md.append(f"| {arm}, {kind} | {d['n']} | " + " | ".join(ci(y) for y in d["p_final_answer"]) + " |")
    md.append("")


def figB3(out: Path, md: list[str]) -> None:
    oe = load("expb")["order_effect"]
    fig, ax = plt.subplots(figsize=(NARROW, 2.7))
    arms = [("debate", "o", "Debate"), ("dissent", "D", "Dissent"), ("anon", "^", "Anonymous (order shuffled)")]
    for gi, (mode, _mlab) in enumerate(B_MODES):
        for j, (arm, mk, _) in enumerate(arms):
            d, x = oe[f"{mode} {arm}"], gi + (j - 1) * 0.22
            ax.errorbar([x], [d["mean"]], yerr=err(d), color=C["peer"], marker=mk, mec="white", mew=0.8, ms=6.5,
                        elinewidth=0.9, capsize=0, ls="none")
    ax.axhline(0.5, color=C["neutral"], lw=0.9, ls="--")
    ax.text(1.45, 0.505, "no order\npreference", fontsize=8, color=C["ink2"], va="bottom", ha="right", linespacing=1.0)
    ax.set_xticks([0, 1], [lab for _, lab in B_MODES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0.4, 0.75)
    ax.set_ylabel("Changed agents taking the\nlast-shown peer's answer")
    ygrid(ax)
    ax.legend(handles=[Line2D([], [], color=C["peer"], marker=mk, ls="none", ms=6, label=lab) for _, mk, lab in arms],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, borderaxespad=0.2, handletextpad=0.3)
    save(fig, out, "figB3_order_effect", NARROW)

    md += ["## figB3_order_effect", "", "Changed agents whose two shown peers stated different answers and whose new answer is one of "
           "them; share adopting the last-shown peer, 95% question bootstrap. Source: `expb_results.json` (`order_effect`).", "",
           "| mode × arm | agents | share last-shown |", "|---|---|---|"]
    for mode, _ in B_MODES:
        for arm, _, _ in arms:
            d = oe[f"{mode} {arm}"]
            md.append(f"| {mode} {arm} | {d['n']} | {ci(d)} |")
    md.append("")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures/part2"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    sections = []
    for f in (figA1, figA2, figA3, figB1, figB2, figB3):
        md: list[str] = []
        f(out, md)
        name = md[0].removeprefix("## ").split(" ")[0]
        sections += md + [f"Draft caption: *{CAPTIONS[name]}*", ""]
    (out / "README.md").write_text(HEADER + "\n" + "\n".join(sections) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
