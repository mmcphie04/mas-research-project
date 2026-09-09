"""The Stage-1 pilot report: the thirteen statistics and the eight go/no-go gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mad.analysis.labels import build_events, label_counts, revision_rates
from mad.analysis.oracles import evaluate_all, headroom
from mad.analysis.stats import auroc

GATES = {
    "G1_base_accuracy": (0.25, 0.70),
    "G2_non_unanimous_round0": (0.35, None),
    "G3_revision_rate": (0.10, None),
    "G4_harmful_share": (0.08, None),
    "G4_beneficial_share": (0.15, None),
    "G5_panel_headroom": (0.04, None),
    "G7_probe_auroc": (0.60, None),
}


def agreement_distribution(events: pd.DataFrame, column: str = "pre_answer") -> dict[str, float]:
    """P(3-0), P(2-1), P(1-1-1) over the round-0 answers."""
    counts = {"unanimous": 0, "majority": 0, "split": 0}
    for _, grp in events.groupby("question_id", sort=False):
        vals = grp[column].tolist()
        top = max(vals.count(v) for v in set(vals))
        counts["unanimous" if top == len(vals) else ("majority" if top > 1 else "split")] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def conformity_signature(events: pd.DataFrame) -> dict[str, float]:
    peers_disagree = events[events["n_peers_agree_pre"] == 0]
    peers_agree = events[events["n_peers_agree_pre"] > 0]
    switched = events[events["revised"] == 1]
    return {
        "p_switch_given_peers_disagree": float(peers_disagree["revised"].mean())
        if len(peers_disagree)
        else np.nan,
        "p_switch_given_peers_agree": float(peers_agree["revised"].mean())
        if len(peers_agree)
        else np.nan,
        "p_lands_on_peer_majority_given_switch": (
            float((switched["post_answer"] == switched["peer_majority"]).mean())
            if len(switched)
            else np.nan
        ),
    }


def parsing_health(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "freetext_parse_failure_rate": float(1 - rows["freetext_parse_ok"].mean()),
        "freetext_readout_disagreement_rate": float(
            1 - rows.loc[rows["freetext_parse_ok"] == 1, "freetext_agrees_readout"].mean()
        ),
    }


def confidence_auroc(events: pd.DataFrame) -> dict[str, float]:
    return {
        "readout_conf_auroc_pre": auroc(events["pre_correct"], events["pre_confidence"]),
        "readout_conf_auroc_post": auroc(events["post_correct"], events["post_confidence"]),
    }


def pilot_report(rows: pd.DataFrame, probe_auroc: float | None = None) -> dict:
    events = build_events(rows)
    debate = events[events["condition"] == "debate"]
    control = events[events["condition"] == "self_reconsider"]
    counts = label_counts(debate)
    n_rev = sum(counts[k] for k in ("beneficial", "harmful", "unhelpful")) or 1
    head = headroom(debate)
    rates = revision_rates(debate)

    report = {
        "n_questions": int(debate["question_id"].nunique()),
        "n_agent_states": int(len(rows)),
        "revision_rates": rates,
        "label_counts": counts,
        "label_shares": {k: counts[k] / n_rev for k in ("beneficial", "harmful", "unhelpful")},
        "agreement_round0": agreement_distribution(debate),
        "conformity": conformity_signature(debate),
        "parsing": parsing_health(rows),
        "confidence_auroc": confidence_auroc(debate),
        "headroom": head,
        "policies": evaluate_all(debate).to_dict(orient="records"),
        "probe_auroc_pre_correctness": probe_auroc,
    }
    if len(control):
        report["control"] = {
            "revision_rates": revision_rates(control),
            "label_counts": label_counts(control),
        }
    report["gates"] = evaluate_gates(report)
    return report


def evaluate_gates(report: dict) -> dict:
    agree = report["agreement_round0"]
    values = {
        "G1_base_accuracy": report["revision_rates"]["base_accuracy"],
        "G2_non_unanimous_round0": 1.0 - agree["unanimous"],
        "G3_revision_rate": report["revision_rates"]["revision_rate"],
        "G4_harmful_share": report["label_shares"]["harmful"],
        "G4_beneficial_share": report["label_shares"]["beneficial"],
        "G5_panel_headroom": report["headroom"]["panel_headroom_optimal"],
        "G7_probe_auroc": report.get("probe_auroc_pre_correctness"),
    }
    out = {}
    for name, (lo, hi) in GATES.items():
        v = values.get(name)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[name] = {"value": v, "pass": None, "threshold": [lo, hi]}
            continue
        ok = (lo is None or v >= lo) and (hi is None or v <= hi)
        out[name] = {"value": float(v), "pass": bool(ok), "threshold": [lo, hi]}
    return out


def save_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=float))
    return path


def format_report(report: dict) -> str:
    lines = ["=" * 68, "STAGE 1 PILOT REPORT", "=" * 68]
    r = report["revision_rates"]
    lines += [
        f"questions {report['n_questions']}   agent-states {report['n_agent_states']}",
        "",
        f"round-0 accuracy p     {r['base_accuracy']:.3f}",
        f"revision rate          {r['revision_rate']:.3f}   r_C {r['r_C']:.3f}   r_I {r['r_I']:.3f}",
        f"switch-to-correct s    {r['s_switch_to_correct']:.3f}",
        "",
        "labels  " + "  ".join(f"{k} {v}" for k, v in report["label_counts"].items()),
        "shares  " + "  ".join(f"{k} {v:.3f}" for k, v in report["label_shares"].items()),
    ]
    a = report["agreement_round0"]
    lines += [
        "",
        f"round-0 agreement      3-0 {a['unanimous']:.3f}  2-1 {a['majority']:.3f}  1-1-1 {a['split']:.3f}",
    ]
    h = report["headroom"]
    lines += [
        "",
        "PANEL ACCURACY",
        f"  round-0 plurality / self-consistency@3  {h['panel_round0']:.4f}",
        f"  standard MAD                            {h['panel_mad']:.4f}   (debate gain {h['debate_gain_over_round0']:+.4f})",
        f"  oracle: reject harmful                  {h['panel_oracle_reject_harmful']:.4f}",
        f"  oracle: full (per-agent optimal)        {h['panel_oracle_full']:.4f}",
        f"  oracle: panel-optimal (true ceiling)    {h['panel_oracle_panel_optimal']:.4f}",
        f"  PANEL HEADROOM (G5, vs true ceiling)    {h['panel_headroom_optimal']:+.4f}",
        f"  oracle panel-flip rate f                {h['panel_flip_rate_f']:.4f}",
        "",
        "AGENT LEVEL",
        f"  standard MAD {h['agent_mad']:.4f}   oracle {h['agent_oracle_full']:.4f}   headroom {h['agent_headroom_full']:+.4f}",
    ]
    c = report["conformity"]
    lines += [
        "",
        f"P(switch | peers disagree) {c['p_switch_given_peers_disagree']:.3f}"
        f"   P(switch | peers agree) {c['p_switch_given_peers_agree']:.3f}",
        f"P(lands on peer majority | switch) {c['p_lands_on_peer_majority_given_switch']:.3f}",
    ]
    p = report["parsing"]
    ca = report["confidence_auroc"]
    lines += [
        "",
        f"parse failures {p['freetext_parse_failure_rate']:.4f}   freetext/readout disagreement {p['freetext_readout_disagreement_rate']:.4f}",
        f"readout-confidence AUROC   pre {ca['readout_conf_auroc_pre']:.3f}   post {ca['readout_conf_auroc_post']:.3f}",
    ]
    if "control" in report:
        cr = report["control"]["revision_rates"]
        lines += [
            "",
            f"SELF-RECONSIDERATION CONTROL   revision rate {cr['revision_rate']:.3f} "
            f"(debate {r['revision_rate']:.3f})",
        ]
    lines += ["", "GATES"]
    for name, g in report["gates"].items():
        mark = "?" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        val = "n/a" if g["value"] is None else f"{g['value']:.4f}"
        lines.append(f"  {mark:<4} {name:<28} {val}  threshold {g['threshold']}")
    lines.append("=" * 68)
    return "\n".join(lines)
