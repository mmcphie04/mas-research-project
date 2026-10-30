"""Midterm-report figures, regenerated from committed artifacts only.

    uv run --no-project --with matplotlib==3.9.2 --with numpy python scripts/make_midterm_figures.py

(matplotlib is not a project dependency; the command above uses a throwaway environment.)

Reads only the small committed reports listed in SOURCES; no activations, nothing under runs/, no
GPU, and nothing from main250's reserved test split beyond what those reports already state.
Every plotted number is also looked up, as rendered text, in the matching Markdown report. A
number that cannot be found there is listed as a discrepancy and the script exits non-zero.

Writes figures/midterm/fig{2..9}_*.pdf (vector) and .png (200 dpi preview), and
figures/midterm/plotted_values.md (every plotted number, plus the result of the Markdown checks).
fig1_pipeline.tex (TikZ) is written by hand and is not touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
SOURCES = {
    "curves": A / "next_phase/commit_curves/core_bf16_bs1/curves.csv",
    "curves_analysis": A / "next_phase/commit_curves/core_bf16_bs1/analysis.json",
    "curves_md": ROOT / "docs/NEXT_PHASE_CPU_RESULTS.md",
    "primary": A / "main250/analysis/primary.json",
    "primary_md": A / "main250/analysis/primary.md",
    "exploratory": A / "main250/analysis/exploratory.json",
    "exploratory_md": A / "main250/analysis/exploratory.md",
    "signals": A / "main250/analysis/selection_signals.json",
    "signals_md": A / "main250/analysis/selection_signals.md",
    "gate": A / "revision_gate/feasibility.json",
    "gate_md": A / "revision_gate/feasibility.md",
    "shifts": A / "val12b/investigation/shifts.json",
    "shifts_md": ROOT / "docs/R1_INVESTIGATION.md",
    "exposure": A / "next_phase/exposure_commit/summary.json",
}

# One color per concept, in every figure. Hues from the dataviz skill's reference palette, except
# structured (#b8429a) and gates (#3b3b98), chosen so that every set of concepts drawn together
# clears the skill's checks (protan/deutan OKLab dE >= 8, normal-vision dE >= 15, >= 3:1 on white).
C = {
    "debate": "#2a78d6",
    "reconsideration": "#eb6834",
    "independent": "#199e70",
    "structured": "#b8429a",
    "gate": "#3b3b98",
    "neutral": "#898781",  # baselines and fixed rules
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
}
ARMS = {  # condition -> (concept, label)
    "debate3": ("debate", "Debate, 3 agents"),
    "structured3": ("structured", "Structured, 3 agents"),
    "debate5": ("debate", "Debate, 5 agents"),
    "structured5": ("structured", "Structured, 5 agents"),
}
WIDE, NARROW = 6.5, 3.2


def tint(hex_color: str, amount: float) -> str:
    """Mix a color with white (amount = share of white)."""
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


def ygrid(ax) -> None:
    ax.grid(axis="y")
    ax.set_axisbelow(True)


# Mismatches found and explained; reported in plotted_values.md but not treated as failures.
KNOWN = {
    "| reconsideration, unchanged | 0.95 | 0.88 | 0.89 | 0.88 | 0.93 | 1.00 |":
        "§12 of the report predates tie-aware scoring (§15.1): the 10% cell was 0.874 (0.87) in 7da85eb's "
        "curves.csv and is 0.876 (0.88) in the committed one; §15.1's 'unchanged at two decimals' misses this cell.",
}


class Checks:
    """Look up each plotted number, as the report renders it, in that report's text."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []
        self._text: dict[str, str] = {}

    def expect(self, fig: str, key: str, s: str) -> None:
        if key not in self._text:
            self._text[key] = SOURCES[key].read_text()
        self.rows.append((fig, SOURCES[key].relative_to(ROOT).as_posix(), s, s in self._text[key]))

    def failures(self) -> list[tuple[str, str, str, bool]]:
        return [r for r in self.rows if not r[3] and r[2] not in KNOWN]

    def known(self) -> list[tuple[str, str, str, bool]]:
        return [r for r in self.rows if not r[3] and r[2] in KNOWN]


def load(key: str):
    return json.loads(SOURCES[key].read_text())


def f3(x: float) -> str:
    return f"{x:.3f}"


def ci3(d: dict) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def save(fig, out: Path, name: str, width: float) -> None:
    """Save with a tight crop whose width is exactly `width` inches, so fonts print at their set size.

    The overhang of labels outside the axes is (nearly) fixed in inches, so a few resizes converge.
    """
    for _ in range(6):
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        if abs(bb.width - width) < 0.005:
            break
        fig.set_size_inches(fig.get_figwidth() + width - bb.width, fig.get_figheight())
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})  # no timestamp, so reruns give identical files
    fig.savefig(out / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
def fig2(out: Path, chk: Checks, md: list[str]) -> None:
    rows = [r for r in csv.DictReader(SOURCES["curves"].open())
            if r["pool"] == "pooled" and r["metric"] == "is_eventual" and r["variant"] == "standard"]
    series = {}
    for cond in ("debate", "self_reconsider"):
        for sub in ("changed", "unchanged"):
            rs = sorted((r for r in rows if r["condition"] == cond and r["subset"] == sub),
                        key=lambda r: float(r["fraction"]))
            series[cond, sub] = rs
    gap = load("curves_analysis")["paired_debate_minus_self_changed_is_eventual"]["0.0"]

    # The report (§12) prints the n for changed answers, the f = 0 interval for changed answers,
    # and every point estimate to two decimals.
    names = {("debate", "changed"): "debate, changed (n = 251)",
             ("self_reconsider", "changed"): "reconsideration, changed (n = 308)",
             ("debate", "unchanged"): "debate, unchanged",
             ("self_reconsider", "unchanged"): "reconsideration, unchanged"}
    for k, rs in series.items():
        cells = [f"{float(r['est']):.2f}" for r in rs]
        if k[1] == "changed":
            cells[0] = f"**{cells[0]}** [{float(rs[0]['lo']):.2f}, {float(rs[0]['hi']):.2f}]"
        chk.expect("fig2", "curves_md", f"| {names[k]} | " + " | ".join(cells) + " |")
    chk.expect("fig2", "curves_md", f"| 0% | +{gap['mean_diff']:.2f} [{gap['lo']:.2f}, {gap['hi']:.2f}] |")

    fig, ax = plt.subplots(figsize=(NARROW, 2.9))
    for (cond, sub), rs in series.items():
        col = C["debate"] if cond == "debate" else C["reconsideration"]
        ls = "-" if sub == "changed" else (0, (4, 2))
        x = [100 * float(r["fraction"]) for r in rs]
        ax.fill_between(x, [float(r["lo"]) for r in rs], [float(r["hi"]) for r in rs],
                        color=col, alpha=0.13, lw=0)
        ax.plot(x, [float(r["est"]) for r in rs], color=col, ls=ls, marker="o",
                mfc=col if sub == "changed" else "white", mec=col, mew=1.2, ms=4.5)
    d0, r0 = float(series["debate", "changed"][0]["est"]), float(series["self_reconsider", "changed"][0]["est"])
    ax.annotate("", xy=(-3.5, d0), xytext=(-3.5, r0),
                arrowprops=dict(arrowstyle="|-|,widthA=0.25,widthB=0.25", color=C["ink2"], lw=0.8,
                                shrinkA=0, shrinkB=0))
    ax.text(102, 0.04, f"Bracket: changed answers at 0%,\ndebate − reconsideration\n"
                       f"= +{gap['mean_diff']:.2f} [{gap['lo']:.2f}, {gap['hi']:.2f}]\n"
                       f"(paired, {gap['questions']} questions)",
            fontsize=9, color=C["ink2"], va="bottom", ha="right")
    ax.set_xlim(-7, 103)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 10, 25, 50, 75, 100])
    ax.set_xlabel("Share of own revision written (%)")
    ax.set_ylabel("Forced answer already equals\nthe eventual answer (share)")
    ygrid(ax)
    handles = [Line2D([], [], color=C["debate"], marker="o", ms=4.5, label="Debate, changed"),
               Line2D([], [], color=C["debate"], ls=(0, (4, 2)), marker="o", mfc="white", ms=4.5,
                      label="Debate, unchanged"),
               Line2D([], [], color=C["reconsideration"], marker="o", ms=4.5, label="Reconsideration, changed"),
               Line2D([], [], color=C["reconsideration"], ls=(0, (4, 2)), marker="o", mfc="white", ms=4.5,
                      label="Reconsideration, unchanged")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=1,
               handlelength=2.2, borderaxespad=0)
    save(fig, out, "fig2_commitment_curves", NARROW)

    md += ["## fig2_commitment_curves", "",
           "Pooled pilot + validation, standardized prefix `Answer: (`, metric `is_eventual`; question-cluster "
           "bootstrap 95% intervals from `curves.csv`.", "",
           "| series | n responses | questions | " + " | ".join(f"{int(100 * float(r['fraction']))}%"
                                                         for r in series["debate", "changed"]) + " |",
           "|---|---|---|" + "---|" * 6]
    for (cond, sub), rs in series.items():
        lab = ("debate" if cond == "debate" else "reconsideration") + f", {sub}"
        md.append(f"| {lab} | {rs[0]['n']} | {rs[0]['questions']} | " + " | ".join(
            f"{float(r['est']):.3f} [{float(r['lo']):.3f}, {float(r['hi']):.3f}]" for r in rs) + " |")
    md += ["", f"Paired gap at 0% (changed answers, debate − reconsideration): {gap['mean_diff']:+.3f} "
               f"[{gap['lo']:.3f}, {gap['hi']:.3f}], {gap['questions']} questions, "
               f"{gap['favour_a']} favour debate / {gap['favour_b']} favour reconsideration.", ""]


# ---------------------------------------------------------------------------------------------
def fig3(out: Path, chk: Checks, md: list[str]) -> None:
    ex = load("exploratory")
    cov = ex["1_coverage_vs_selection"]["pool"]
    vc = ex["3_correlated_errors"]["voting_curve_vs_independent"]
    icc = ex["3_correlated_errors"]["icc_correctness"]
    neff = ex["3_correlated_errors"]["n_eff"]["10"]
    prim = load("primary")["independent_curve"]
    ks = list(range(3, 11))
    for k in ks:
        c, o, p = cov[str(k)]["coverage"], vc[str(k)]["observed"], vc[str(k)]["predicted_independent"]
        assert abs(o["mean"] - prim[str(k)]["mean"]) < 1e-12 and [o["lo"], o["hi"]] == prim[str(k)]["ci95"]
        chk.expect("fig3", "exploratory_md", f"| {k} | {ci3(c)} | {ci3(o)} |")
        chk.expect("fig3", "exploratory_md", f"| {k} | {ci3(o)} | {ci3(p)} |")
        chk.expect("fig3", "primary_md", f"| {k} | {o['mean']:.3f} | [{o['lo']:.3f}, {o['hi']:.3f}] |")
    chk.expect("fig3", "exploratory_md", f"ICC(1)): {ci3(icc)}")
    chk.expect("fig3", "exploratory_md", f"k=10: {neff['n_eff']:.2f} [{neff['lo']:.2f}, {neff['hi']:.2f}]")

    fig, ax = plt.subplots(figsize=(NARROW, 3.0))
    spec = [("coverage", "At least one of the k answers correct", C["ink2"], "-",
             [cov[str(k)]["coverage"] for k in ks]),
            ("predicted", "Plurality vote, predicted if errors\nwere independent", C["independent"], (0, (4, 2)),
             [vc[str(k)]["predicted_independent"] for k in ks]),
            ("observed", "Plurality vote, observed", C["independent"], "-",
             [vc[str(k)]["observed"] for k in ks])]
    handles = []
    for _, lab, col, ls, vals in spec:
        ax.fill_between(ks, [v["lo"] for v in vals], [v["hi"] for v in vals], color=col, alpha=0.13, lw=0)
        ax.plot(ks, [v["mean"] for v in vals], color=col, ls=ls, marker="o", ms=4,
                mfc="white" if ls != "-" else col, mec=col, mew=1.2)
        handles.append(Line2D([], [], color=col, ls=ls, marker="o", ms=4,
                              mfc="white" if ls != "-" else col, label=lab))
    ax.text(10, 0.405, f"Intra-class correlation of\ncorrectness: {icc['mean']:.2f} [{icc['lo']:.2f}, {icc['hi']:.2f}]\n"
                      f"Effective independent answers\nat k = 10: {neff['n_eff']:.2f} [{neff['lo']:.2f}, {neff['hi']:.2f}]",
            ha="right", va="bottom", fontsize=9, color=C["ink2"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C["axis"], lw=0.6))
    ax.set_xticks(ks)
    ax.set_xlim(2.7, 10.3)
    ax.set_ylim(0.39, 1.0)
    ax.set_xlabel("Number of independent answers (k)")
    ax.set_ylabel("Share of questions")
    ygrid(ax)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=1,
               handlelength=2.2, borderaxespad=0)
    save(fig, out, "fig3_voting_curve", NARROW)

    md += ["## fig3_voting_curve (exploratory)", "",
           "Means with 95% question-bootstrap intervals (10,000 resamples, seed 0), 200 questions.", "",
           "| k | at least one correct | plurality, observed | plurality, predicted if independent |",
           "|---|---|---|---|"]
    md += [f"| {k} | {ci3(cov[str(k)]['coverage'])} | {ci3(vc[str(k)]['observed'])} | "
           f"{ci3(vc[str(k)]['predicted_independent'])} |" for k in ks]
    md += ["", f"ICC(1) of correctness {ci3(icc)}; n_eff at k = 10: {neff['n_eff']:.3f} "
               f"[{neff['lo']:.3f}, {neff['hi']:.3f}].", ""]


# ---------------------------------------------------------------------------------------------
def fig4(out: Path, chk: Checks, md: list[str]) -> None:
    pr = load("primary")
    ex = load("exploratory")
    cost = pr["cost"]
    r0_resp = cost["conditions"]["pool"]["responses"]
    per_r0 = cost["round0_stage_s_total"] / r0_resp
    chk.expect("fig4", "primary_md", f"{cost['round0_stage_s_total']:,.0f} s in total")
    chk.expect("fig4", "primary_md", f"| pool | {r0_resp} |")
    ind = []
    for k in range(3, 11):
        c = pr["independent_curve"][str(k)]
        ind.append((k, k * per_r0, c["mean"], c["ci95"][0], c["ci95"][1]))
    arms = []
    for cond, (concept, lab) in ARMS.items():
        cc = cost["conditions"][cond]
        k = pr["revision_events"][cond]["k"]
        per_rev = (cc["generation_s"] + cc["extraction_s"]) / cc["revisions"]
        acc = ex["1_coverage_vs_selection"]["debate"][cond]["plurality_after"]
        assert abs(acc["mean"] - pr["revision_events"][cond]["panel_acc_after"]) < 1e-12
        arms.append((cond, concept, lab, k, k * per_r0 + k * per_rev, acc["mean"], acc["lo"], acc["hi"],
                     cc["generation_s"], cc["extraction_s"], cc["revisions"]))
        chk.expect("fig4", "primary_md", f"| {cond} | {cc['revisions']} | {cc['input_tokens']:,} | "
                                         f"{cc['output_tokens']:,} | {cc['generation_s']:,.0f} |")
        chk.expect("fig4", "primary_md", f"| {cc['extraction_s']:,.0f} |")
        chk.expect("fig4", "exploratory_md", f"| {cond} | ")
        chk.expect("fig4", "exploratory_md", f"| {ci3(acc)} | ")

    fig, ax = plt.subplots(figsize=(WIDE, 3.1))
    ax.errorbar([r[1] for r in ind], [r[2] for r in ind], yerr=[[r[2] - r[3] for r in ind], [r[4] - r[2] for r in ind]],
                color=C["independent"], ecolor=tint(C["independent"], 0.45), marker="o", ms=5, lw=1.6,
                elinewidth=1.0, capsize=0, zorder=3)
    # Where each arm's label goes: (anchor y, dx points, dy points, ha, va), placed clear of other marks.
    place = {"debate3": ("hi", 2, 3, "right", "bottom"), "structured3": ("lo", 3, -3, "left", "top"),
             "debate5": ("lo", 0, -3, "center", "top"), "structured5": ("hi", 0, 3, "center", "bottom")}
    for cond, concept, lab, k, sec, m, lo, hi, *_ in arms:
        ax.errorbar([sec], [m], yerr=[[m - lo], [hi - m]], color=C[concept], marker="o" if k == 3 else "s",
                    ms=7, elinewidth=1.3, capsize=0, ls="none", mec="white", mew=1.0, zorder=4)
        anchor, dx, dy, ha, va = place[cond]
        ax.annotate(lab.replace(", ", ",\n"), (sec, hi if anchor == "hi" else lo), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va, fontsize=9, color=C["ink"], linespacing=1.0)
    top = ax.secondary_xaxis("top")
    top.set_xticks([r[1] for r in ind], [str(r[0]) for r in ind])
    top.set_xlabel("Independent answers voted (k), placed at their cost", color=C["ink2"])
    top.tick_params(colors=C["independent"], labelcolor=C["ink2"])
    top.spines["top"].set_visible(False)
    ax.set_xlim(10, 90)
    ax.set_ylim(0.55, 0.775)
    ax.set_xlabel("Measured seconds per question (generation + answer readouts)")
    ax.set_ylabel("Panel accuracy")
    ygrid(ax)
    handles = [Line2D([], [], color=C["independent"], marker="o", ms=5, label="Independent voting"),
               Line2D([], [], color=C["debate"], marker="o", ls="none", ms=6, label="Debate"),
               Line2D([], [], color=C["structured"], marker="o", ls="none", ms=6, label="Structured debate"),
               Line2D([], [], color=C["ink2"], marker="o", ls="none", mfc="white", ms=6, label="3 agents"),
               Line2D([], [], color=C["ink2"], marker="s", ls="none", mfc="white", ms=6, label="5 agents")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, -0.03), ncol=5, handletextpad=0.4,
               columnspacing=1.2, borderaxespad=0)
    save(fig, out, "fig4_accuracy_vs_compute", WIDE)

    md += ["## fig4_accuracy_vs_compute", "",
           f"Round-0 seconds per response = {cost['round0_stage_s_total']} / {r0_resp} = {per_r0:.4f} s "
           "(generation and readouts of the pool's answers; not separable in the records). "
           "Independent k: k × that. Debate arm: k × that + k × (generation s + extraction s) / revisions. "
           "Accuracy intervals: 95% question bootstrap (10,000, seed 0); independent from "
           "`primary.json` (`independent_curve`), debate arms from `exploratory.json` "
           "(`1_coverage_vs_selection.debate.<arm>.plurality_after`, same per-question scores and "
           "bootstrap; the point estimates equal `primary.json`'s `panel_acc_after`).", "",
           "| arm | seconds per question | panel accuracy [95%] |", "|---|---|---|"]
    md += [f"| independent, k = {k} | {s:.1f} | {m:.3f} [{lo:.3f}, {hi:.3f}] |" for k, s, m, lo, hi in ind]
    md += [f"| {cond} | {sec:.1f} = {k} × {per_r0:.3f} + {k} × ({g} + {e}) / {n} | {m:.3f} [{lo:.3f}, {hi:.3f}] |"
           for cond, _, _, k, sec, m, lo, hi, g, e, n in arms]
    md.append("")


# ---------------------------------------------------------------------------------------------
def fig5(out: Path, chk: Checks, md: list[str]) -> None:
    pr = load("primary")["revision_events"]
    ex = load("exploratory")["4_consensus"]
    for cond in ARMS:
        a = pr[cond]["per_agent"]
        chk.expect("fig5", "primary_md", f"| {cond} | ")
        chk.expect("fig5", "primary_md", f"| {a['revisions']} | {a['changed']} | {a['helpful']} | {a['harmful']} |")
        e = ex[cond]
        chk.expect("fig5", "exploratory_md", f"| {cond} | {ci3(e['wrong_unanimity_before'])} | "
                                             f"{ci3(e['wrong_unanimity_after'])} | {ci3(e['change'])} |")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.9), gridspec_kw={"wspace": 0.38})
    xpos = {"debate3": 0, "structured3": 1, "debate5": 2.5, "structured5": 3.5}  # grouped by panel size
    for cond, (concept, _) in ARMS.items():
        a, i = pr[cond]["per_agent"], xpos[cond]
        a1.bar(i, a["helpful"], width=0.7, color=C[concept], lw=0)
        a1.bar(i, -a["harmful"], width=0.7, color=tint(C[concept], 0.55), lw=0, hatch="////", edgecolor=C[concept])
        a1.text(i, a["helpful"] + 2, str(a["helpful"]), ha="center", va="bottom", fontsize=9, color=C["ink"])
        a1.text(i, -a["harmful"] - 2, str(a["harmful"]), ha="center", va="top", fontsize=9, color=C["ink"])
    a1.axhline(0, color=C["ink"], lw=0.8)
    a1.set_ylim(-118, 118)
    a1.set_yticks([-80, -40, 0, 40, 80], ["80", "40", "0", "40", "80"])
    a1.set_ylabel("Revised answers (count)")
    a1.text(-0.5, 116, "Helpful: wrong → right", va="top", fontsize=9, color=C["ink2"])
    a1.text(-0.5, -116, "Harmful: right → wrong", va="bottom", fontsize=9, color=C["ink2"])
    a1.text(-0.2, 1.02, "(a)", transform=a1.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ygrid(a1)

    for cond, (concept, _) in ARMS.items():
        e, i = ex[cond], xpos[cond]
        b, af, ch = e["wrong_unanimity_before"], e["wrong_unanimity_after"], e["change"]
        a2.plot([i - 0.18, i + 0.18], [b["mean"], af["mean"]], color=C["axis"], lw=1, zorder=1)
        a2.errorbar([i - 0.18], [b["mean"]], yerr=[[b["mean"] - b["lo"]], [b["hi"] - b["mean"]]],
                    color=C["neutral"], marker="o", mfc="white", mec=C["neutral"], mew=1.2, ms=5.5,
                    elinewidth=0.9, capsize=0, zorder=3)
        a2.errorbar([i + 0.18], [af["mean"]], yerr=[[af["mean"] - af["lo"]], [af["hi"] - af["mean"]]],
                    color=C[concept], marker="o", mec="white", mew=1.0, ms=6.5, elinewidth=0.9, capsize=0, zorder=3)
        if ch["lo"] > 0 or ch["hi"] < 0:  # label only the paired changes whose interval excludes 0
            a2.text(i + 0.18, af["hi"] + 0.006, f"Δ {ch['mean']:+.3f}\n[{ch['lo']:+.3f}, {ch['hi']:+.3f}]",
                    ha="center", va="bottom", fontsize=9, color=C["ink2"], linespacing=1.05)
    a2.set_xlim(-0.6, 4.1)
    a2.set_ylim(0, 0.32)
    a2.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25])
    a2.set_ylabel("Questions where every agent\ngives the same wrong answer")
    a2.text(-0.27, 1.02, "(b)", transform=a2.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ygrid(a2)
    for ax in (a1, a2):
        ax.set_xticks([0.5, 3.0], ["3 agents", "5 agents"])
        ax.tick_params(axis="x", length=0)
    a1.set_xlim(-0.6, 4.1)
    a2.legend(handles=[Line2D([], [], color=C["neutral"], marker="o", mfc="white", ls="none", ms=5.5,
                              label="before revision"),
                       Line2D([], [], color=C["ink2"], marker="o", ls="none", ms=6.5, label="after revision")],
              loc="upper left", handletextpad=0.3, borderaxespad=0.2)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["debate"], label="Debate"),
                        plt.Rectangle((0, 0), 1, 1, color=C["structured"], label="Structured debate"),
                        plt.Rectangle((0, 0), 1, 1, fc=tint(C["ink2"], 0.55), ec=C["ink2"], hatch="////", lw=0,
                                      label="harmful (lighter, hatched)")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, borderaxespad=0)
    save(fig, out, "fig5_what_debate_does", WIDE)

    md += ["## fig5_what_debate_does", "",
           "(a) Per-agent revision events (`primary.json` `revision_events.<arm>.per_agent`). "
           "(b) Wrong unanimity (exploratory §4), 95% question bootstrap.", "",
           "| arm | revisions | changed | helpful | harmful | wrong unanimity before | after | change (paired) |",
           "|---|---|---|---|---|---|---|---|"]
    for cond in ARMS:
        a, e = pr[cond]["per_agent"], ex[cond]
        md.append(f"| {cond} | {a['revisions']} | {a['changed']} | {a['helpful']} | {a['harmful']} | "
                  f"{ci3(e['wrong_unanimity_before'])} | {ci3(e['wrong_unanimity_after'])} | {ci3(e['change'])} |")
    md.append("")


# ---------------------------------------------------------------------------------------------
SIGNALS = [("ii_response_logprob_mean", "(ii) response mean log-prob (as exp)", "(a) Mean log-prob"),
           ("iii_verbalized_confidence", "(iii) verbalized confidence", "(b) Stated confidence"),
           ("iv_peer_exposure", "(iv) peer-exposure readout", "(c) Peer-exposure readout")]
SETS = [("pool first 10", "independent", "10 independent answers"),
        ("debate3 post-debate", "debate", "Debate, 3 agents"),
        ("structured3 post-debate", "structured", "Structured, 3 agents"),
        ("debate5 post-debate", "debate", "Debate, 5 agents"),
        ("structured5 post-debate", "structured", "Structured, 5 agents")]


def fig6(out: Path, chk: Checks, md: list[str]) -> None:
    wq = load("signals")["within_question"]
    md += ["## fig6_signal_auroc (exploratory)", "",
           "AUROC of each signal for the correctness of the answer it accompanies. Ordinary: all states. "
           "Within-question: every correct answer paired with every incorrect answer to the same "
           "(non-unanimous) question. 95% question bootstrap.", "",
           "| answers | signal | ordinary, all states | within-question | pairs (questions) |", "|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (sk, slab, title) in zip(axes, SIGNALS, strict=True):
        assert any(sk in wq[setk] for setk, _, _ in SETS), f"no within-question entries for {sk}"
        for row, (setk, concept, _) in enumerate(SETS):
            if sk not in wq[setk]:
                continue
            d = wq[setk][sk]
            o, w = d["ordinary_auroc_all_states"], d["within_question"]
            y = len(SETS) - 1 - row
            chk.expect("fig6", "signals_md", f"| {setk} | {slab} | {ci3(w)} | {d['questions_with_pairs']} | "
                                             f"{d['pairs']:,} |")
            chk.expect("fig6", "signals_md", f"| {ci3(o)} | {ci3(d['ordinary_auroc_non_unanimous_states'])} |")
            md.append(f"| {setk} | {slab} | {ci3(o)} | {ci3(w)} | {d['pairs']} ({d['questions_with_pairs']}) |")
            col = C[concept]
            ax.plot([o["mean"], w["mean"]], [y + 0.14, y - 0.14], color=tint(col, 0.5), lw=1, zorder=1)
            ax.plot([o["lo"], o["hi"]], [y + 0.14] * 2, color=col, lw=0.9)
            ax.plot([w["lo"], w["hi"]], [y - 0.14] * 2, color=col, lw=0.9)
            ax.plot(o["mean"], y + 0.14, "o", ms=5.5, mfc="white", mec=col, mew=1.3, zorder=3)
            ax.plot(w["mean"], y - 0.14, "o", ms=5.5, color=col, mec="white", mew=0.8, zorder=3)
        ax.axvline(0.5, color=C["ink2"], ls=(0, (3, 2)), lw=0.8, zorder=0)
        ax.set_xlim(0.35, 0.9)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=9, loc="left", pad=4)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(SETS)), [s[2] for s in SETS][::-1])
    axes[1].set_xlabel("AUROC for the correctness of the answer (0.5 = chance)")
    handles = [Line2D([], [], color=C["ink2"], marker="o", mfc="white", ls="none", ms=5.5,
                      label="ordinary AUROC (all answers)"),
               Line2D([], [], color=C["ink2"], marker="o", ls="none", ms=5.5,
                      label="within-question AUROC")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.55, 0.98), ncol=2, borderaxespad=0)
    save(fig, out, "fig6_signal_auroc", WIDE)
    md.append("")


# ---------------------------------------------------------------------------------------------
def fig7(out: Path, chk: Checks, md: list[str]) -> None:
    g = load("gate")
    sim = g["simulation"]["all panels"]
    auc = {m: g["models"][m]["auroc"] for m in ("logistic", "boosted_tree")}
    rows = [("reject_all", "reject all", "Reject every revision", "neutral"),
            ("a_new_is_peer_majority", "(a) accept iff new = peer majority", "(a) Peer-majority rule", "neutral"),
            ("b_conviction_reject_if_conf_before_ge_90", "(b) conviction: reject if confidence before ≥ 90",
             "(b) Conviction rule", "neutral"),
            ("c_mad_conformist_keep_if_old_is_prev_majority", "(c) MAD-Conformist: keep if old = previous majority",
             "(c) MAD-Conformist rule", "neutral"),
            ("logistic", "logistic gate (CV)", f"Logistic gate (AUROC {auc['logistic']['mean']:.2f})", "gate"),
            ("boosted_tree", "boosted-tree gate (CV)", f"Boosted-tree gate (AUROC {auc['boosted_tree']['mean']:.2f})", "gate"),
            ("perfect", "perfect gate (oracle)", "Perfect gate (oracle)", "gate")]
    chk.expect("fig7", "gate_md", f"### all panels ({sim['panels']} panels)")
    for m, lab in (("logistic", "logistic"), ("boosted_tree", "boosted_tree")):
        chk.expect("fig7", "gate_md", f"| {lab} | {ci3(auc[m])} |")
    acc_all = sim["accept_all"]
    md += ["## fig7_revision_gate (exploratory)", "",
           f"Pooled over all {sim['panels']} panels ({g['data']['questions']} questions; Gemma-3-12B datasets "
           "except main250 test), simulated with the gate deciding every changed answer, out-of-fold. "
           "95% question-grouped bootstrap.", "",
           f"Gate AUROC (helpful vs harmful, out-of-fold): logistic {ci3(auc['logistic'])}, "
           f"boosted tree {ci3(auc['boosted_tree'])}.", "",
           f"Accept-all panel accuracy: {ci3(acc_all['panel'])}.", "",
           "| policy | panel accuracy | gain over accept-all |", "|---|---|---|"]
    vals = []
    for key, mdlab, lab, concept in rows:
        p, d = sim[key]["panel"], sim[key]["panel_minus_accept_all"]
        chk.expect("fig7", "gate_md", f"| {mdlab} | {ci3(sim[key]['agent'])} | {ci3(p)} | {ci3(d)} |")
        md.append(f"| {lab} | {ci3(p)} | {ci3(d)} |")
        vals.append((lab, concept, key == "perfect", d))
    md.append("")

    fig, ax = plt.subplots(figsize=(WIDE, 2.5))
    for i, (_, concept, hollow, d) in enumerate(vals):
        y = len(vals) - 1 - i
        col = C[concept]
        ax.plot([d["lo"], d["hi"]], [y, y], color=col, lw=1.2)
        ax.plot(d["mean"], y, "o", ms=6.5, color=col, mfc="white" if hollow else col, mec=col if hollow else "white",
                mew=1.4 if hollow else 1.0, zorder=3)
        ax.text(d["hi"] + 0.003, y, f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]".replace("-", "−"),
                va="center", fontsize=9, color=C["ink2"])
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(range(len(vals)), [v[0] for v in vals][::-1])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.03, 0.15)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
    ax.set_ylim(-0.6, len(vals) - 0.4)
    ax.set_xlabel(f"Gain in panel accuracy over accepting every revision (accept-all = {acc_all['panel']['mean']:.3f})")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save(fig, out, "fig7_revision_gate", WIDE)


# ---------------------------------------------------------------------------------------------
def fig8(out: Path, chk: Checks, md: list[str]) -> None:
    s = load("shifts")["activation_validation"]
    deb = s["debate"]
    groups = [("unanimous", "Unanimous"), ("two_one", "Two against one")]
    kept = [(f"{g}|unchanged|first_right", g, lab, True) for g, lab in groups] + \
           [(f"{g}|unchanged|first_wrong", g, lab, False) for g, lab in groups]
    # The report rounds the kept-answer table to two decimals and the all-agent shifts to three.
    for key, g, _, right in kept:
        name = {"unanimous": "unanimous", "two_one": "two-against-one"}[g] + (", right" if right else ", wrong")
        sign = f"{deb[key]['mean']:+.2f}".replace("-", "−")
        chk.expect("fig8", "shifts_md", f"| {name} | {sign} |")
    dmu = {c: s[c]["disputed_minus_unanimous"] for c in ("debate", "self_reconsider", "independent")}
    chk.expect("fig8", "shifts_md", f"disputed minus unanimous: {dmu['debate']['value']:+.2f} "
                                    f"[{dmu['debate']['ci95'][0]:+.2f}, {dmu['debate']['ci95'][1]:+.2f}]".replace("-", "−"))
    chk.expect("fig8", "shifts_md", f"disputed minus unanimous is {dmu['independent']['value']:+.2f} "
                                    f"[{dmu['independent']['ci95'][0]:+.2f}, {dmu['independent']['ci95'][1]:+.2f}]".replace("-", "−"))
    chk.expect("fig8", "shifts_md", f"(difference {dmu['self_reconsider']['value']:+.2f} "
                                    f"[{dmu['self_reconsider']['ci95'][0]:+.2f}, {dmu['self_reconsider']['ci95'][1]:+.2f}])".replace("-", "−"))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDE, 2.6), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.5})
    for gi, (g, _) in enumerate(groups):
        for off, right in ((-0.2, True), (0.2, False)):
            d = deb[f"{g}|unchanged|{'first_right' if right else 'first_wrong'}"]
            x = gi + off
            a1.plot([x, x], d["ci95"], color=C["debate"], lw=1.0)
            a1.plot(x, d["mean"], "o", ms=6.5, color=C["debate"], mfc=C["debate"] if right else "white",
                    mec="white" if right else C["debate"], mew=1.0 if right else 1.4, zorder=3)
            a1.annotate(f"n={d['n']}", (x, d["ci95"][0]), xytext=(0, -3), textcoords="offset points",
                        ha="center", va="top", fontsize=9, color=C["ink2"])
    a1.axhline(0, color=C["ink"], lw=0.8)
    a1.set_xticks(range(len(groups)), [lab for _, lab in groups])
    a1.tick_params(axis="x", length=0)
    a1.set_xlim(-0.5, 2.0)
    a1.set_ylim(-1.3, 1.0)
    a1.set_ylabel("Probe score change,\nrevised − first answer")
    a1.set_xlabel("Initial agreement in the 3-agent panel")
    a1.legend(handles=[Line2D([], [], color=C["debate"], marker="o", ls="none", ms=6.5, label="kept a right answer"),
                       Line2D([], [], color=C["debate"], marker="o", mfc="white", ls="none", ms=6.5,
                              label="kept a wrong answer")], loc="upper right", handletextpad=0.3)
    a1.text(-0.22, 1.02, "(a)", transform=a1.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ygrid(a1)

    ctrl = [("debate", "Debate", "debate"), ("self_reconsider", "Recon-\nsideration", "reconsideration"),
            ("independent", "Fresh\nsample", "independent")]
    for i, (c, _, concept) in enumerate(ctrl):
        d = dmu[c]
        a2.plot([i, i], d["ci95"], color=C[concept], lw=1.0)
        a2.plot(i, d["value"], "o", ms=6.5, color=C[concept], mec="white", mew=1.0, zorder=3)
    a2.axhline(0, color=C["ink"], lw=0.8)
    a2.set_xticks(range(len(ctrl)), [c[1] for c in ctrl])
    a2.tick_params(axis="x", length=0)
    a2.set_xlim(-0.5, 2.5)
    a2.set_ylim(-1.5, 0.5)
    a2.set_ylabel("Disputed minus unanimous\n(mean shift, all agents)")
    a2.text(-0.3, 1.02, "(b)", transform=a2.transAxes, fontsize=9, fontweight="bold", va="bottom")
    ygrid(a2)
    save(fig, out, "fig8_probe_agreement", WIDE)

    md += ["## fig8_probe_agreement (exploratory)", "",
           "Frozen R1 activation probe, validation tranche (val12b, 200 questions, 3 agents). Shift = revised-state "
           "score minus first-state score. 95% question bootstrap (10,000). Source: "
           "`artifacts/val12b/investigation/shifts.json` (`activation_validation`).", "",
           "| (a) debate, kept answer | n agents (questions) | mean shift [95%] |", "|---|---|---|"]
    for key, _, lab, right in kept:
        d = deb[key]
        md.append(f"| {lab}, kept {'right' if right else 'wrong'} | {d['n']} ({d['questions']}) | "
                  f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    md += ["", "| (b) condition | disputed − unanimous [95%] |", "|---|---|"]
    md += [f"| {c} | {dmu[c]['value']:+.3f} [{dmu[c]['ci95'][0]:+.3f}, {dmu[c]['ci95'][1]:+.3f}] |" for c, _, _ in ctrl]
    md += ["", "Not plotted: three-way splits with a kept answer (5 agents, all wrong at first; mean "
               f"{deb['three_way|unchanged|first_wrong']['mean']:+.3f}).", ""]


# ---------------------------------------------------------------------------------------------
def fig9(out: Path, chk: Checks, md: list[str]) -> None:
    e = load("exposure")["echo_check"]
    pools = [("val", "Validation"), ("pilot", "Pilot")]
    for p, lab in pools:
        d = e[p]
        chk.expect("fig9", "curves_md", f"| {lab.lower()} ({d['revisions_with_two_different_peer_answers']} revisions) | "
                                        f"{d['exposure_argmax_is_last_shown_peer']} | {d['exposure_argmax_is_first_shown_peer']} | "
                                        f"{d['final_stated_answer_is_last_shown_peer']} | "
                                        f"{d['final_stated_answer_is_first_shown_peer']} |")
    fig, ax = plt.subplots(figsize=(NARROW, 2.5))
    w = 0.32
    for i, (p, _) in enumerate(pools):
        d = e[p]
        for off, key, col, hatch in ((-w / 2 - 0.01, "final_stated_answer_is_last_shown_peer", C["debate"], None),
                                     (w / 2 + 0.01, "final_stated_answer_is_first_shown_peer", tint(C["debate"], 0.55), "////")):
            ax.bar(i + off, d[key], width=w, color=col, lw=0, hatch=hatch, edgecolor=C["debate"] if hatch else None)
            ax.text(i + off, d[key] + 1.5, str(d[key]), ha="center", va="bottom", fontsize=9, color=C["ink"])
    ax.set_xticks(range(len(pools)), [f"{lab}\n({e[p]['revisions_with_two_different_peer_answers']} revisions)"
                                      for p, lab in pools])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Revisions (count)")
    ygrid(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C["debate"], label="adopted the peer shown last"),
                       plt.Rectangle((0, 0), 1, 1, fc=tint(C["debate"], 0.55), ec=C["debate"], hatch="////", lw=0,
                                     label="adopted the peer shown first")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, borderaxespad=0.2)
    save(fig, out, "fig9_peer_order", NARROW)

    md += ["## fig9_peer_order (exploratory)", "",
           "Debate revisions whose two shown peers stated different answers; peer order randomized per agent. "
           "Source: `artifacts/next_phase/exposure_commit/summary.json` (`echo_check`).", "",
           "| pool | revisions | final answer = last-shown peer | = first-shown peer | neither | "
           "exposure readout argmax = last | = first |", "|---|---|---|---|---|---|---|"]
    for p, lab in pools:
        d = e[p]
        n = d["revisions_with_two_different_peer_answers"]
        last, first = d["final_stated_answer_is_last_shown_peer"], d["final_stated_answer_is_first_shown_peer"]
        md.append(f"| {lab} | {n} | {last} | {first} | {n - last - first} | {d['exposure_argmax_is_last_shown_peer']} | "
                  f"{d['exposure_argmax_is_first_shown_peer']} |")
    md.append("")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures/midterm"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    chk = Checks()
    md = ["# Plotted values (generated by scripts/make_midterm_figures.py; do not edit)", ""]
    for f in (fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f(out, chk, md)
    bad = chk.failures()
    md += ["## Checks against the Markdown reports", "",
           f"{len(chk.rows)} rendered values looked up; {len(bad)} not found.", ""]
    md += [f"- NOT FOUND ({fig}, `{src}`): `{s}`" for fig, src, s, _ in bad]
    md += [f"- Known discrepancy ({fig}, `{src}`): plotted `{s}`. {KNOWN[s]}" for fig, src, s, _ in chk.known()]
    (out / "plotted_values.md").write_text("\n".join(md) + "\n")
    print(f"checks: {len(chk.rows)} looked up, {len(bad)} not found")
    for fig, src, s, _ in bad:
        print(f"  NOT FOUND {fig} {src}: {s}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
