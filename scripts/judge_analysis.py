"""Analysis of the cross-model judge run, exactly as committed in docs/REVISION_JUDGE_PLAN.md
(exploratory). Question-grouped 95% intervals, 10,000 resamples, seed 0.

    python scripts/judge_analysis.py --runs runs/judge_full --out artifacts/revision_gate/judge

Per judge: accuracy on helpful, on harmful and pooled (accept iff P(revision) > 1/2; exactly 1/2
scores 1/2); AUROC on pooled helpful/harmful; mean P(revision) (all, helpful, harmful,
wrong->wrong); position bias; the decision bar (a) accuracy >= 0.75, (b) clearly beats the
out-of-fold logistic AND boosted-tree gates on the same events, (c) clearly beats the own-answer
rule; simulated gated panels (judge alone, combined with the logistic gate) against accept-all,
reject-all, the fixed rules, the out-of-fold gates and the perfect gate. Thinking: on its
subsample next to non-thinking Qwen3. Cross-model: correlation of correctness, same-wrong-answer
rate and mixed 3-answer panels from the own-answer runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from revision_gate import auroc, cluster_mean, cluster_stat, rule_accept  # noqa: E402

from mad.analysis import revision_events as R  # noqa: E402
from mad.analysis.actions import expected_vote  # noqa: E402

JUDGES = ["gemma", "qwen3", "qwen3_think", "llama"]
LABEL = "EXPLORATORY"
N_BOOT = 10000


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in open(p)]


def find(runs: Path, name: str) -> list[Path]:
    return sorted(runs.rglob(name))


def load_judge(runs: Path, model: str) -> pd.DataFrame | None:
    files = find(runs, f"{model}_judge.jsonl")
    if not files:
        return None
    rows = [r for f in files for r in _jsonl(f)]
    d = pd.DataFrame(rows).pivot_table(index="event_id", columns="order", values=["p1", "p2"])
    out = pd.DataFrame({"p_rev": (d[("p2", "original_first")] + d[("p1", "revised_first")]) / 2,
                        "pos_bias": (d[("p1", "original_first")] + d[("p1", "revised_first")]) / 2 - 0.5})
    return out


def load_own(runs: Path, model: str, root: Path) -> dict[str, list[int]]:
    """question_id -> the model's own answers (-1 = unparsed). Gemma: fresh samples plus pool
    agents 5-7 of main250/pilot250 for the reuse list."""
    own: dict[str, list[int]] = {}
    for f in find(runs, f"{model}_own.jsonl"):
        for r in _jsonl(f):
            own.setdefault(str(r["question_id"]), []).append(int(r["answer"]))
    if model == "gemma":
        reuse = json.loads((root / "artifacts/revision_gate/judge/gemma_reuse.json").read_text())["questions"]
        views = {"main250": root / "runs/main250/main250-pool", "pilot250": root / "runs/pilot250/out/pilot250-pool"}
        pools = {k: pd.read_parquet(v / "agent_states.parquet", columns=["question_id", "agent_id", "phase", "answer"])
                 for k, v in views.items()}
        for q in reuse:
            p = pools[q["source"]]
            p = p[(p.phase == "pre") & (p.question_id.astype(str) == q["question_id"]) & p.agent_id.isin(q["pool_agents"])]
            own[q["question_id"]] = p.sort_values("agent_id").answer.astype(int).tolist()
    return own


def majority(answers) -> int:
    c = Counter(a for a in answers if a >= 0).most_common()
    if not c or (len(c) > 1 and c[0][1] == c[1][1]):
        return -1
    return c[0][0]


def events_table(root: Path) -> pd.DataFrame:
    ev = R.load_events(root)["events"]
    ev = ev.assign(event_id=ev.dataset + "|" + ev.condition + "|" + ev.question_id + "|" + ev.agent_id.astype(str))
    sample = {json.loads(x)["event_id"]: json.loads(x)["thinking_subsample"]
              for x in open(root / "artifacts/revision_gate/judge/events.jsonl")}
    oof = pd.read_csv(root / "artifacts/revision_gate/oof_gate_decisions.csv", dtype={"question_id": str})
    t = ev[ev.event_id.isin(set(sample))].merge(
        oof[["event_id", "p_logistic", "accept_logistic", "p_boosted_tree", "accept_boosted_tree"]], on="event_id", how="left")
    t["thinking_subsample"] = t.event_id.map(sample)
    if len(t) != len(sample):
        raise SystemExit(f"{len(sample) - len(t)} sampled events are missing from the event table")
    return t


def decision_score(p: np.ndarray, helpful: np.ndarray) -> np.ndarray:
    """Accuracy of 'accept iff p > 1/2' on helpful/harmful events (p == 1/2 scores 1/2)."""
    acc = np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, 0.5))
    return np.where(helpful == 1, acc, 1 - acc)


def own_rule_score(t: pd.DataFrame, own: dict) -> np.ndarray:
    """Side with the response whose answer equals the judge's own majority answer; else 1/2."""
    s = []
    for r in t.itertuples(index=False):
        m = majority(own.get(r.question_id, []))
        side_rev = 1.0 if m == r.after else 0.0 if m == r.before else 0.5
        s.append(side_rev if r.kind == "helpful" else 1 - side_rev)
    return np.array(s)


def own_accept_prob(t: pd.DataFrame, own: dict) -> np.ndarray:
    out = []
    for r in t.itertuples(index=False):
        m = majority(own.get(r.question_id, []))
        out.append(1.0 if m == r.after else 0.0 if m == r.before else 0.5)
    return np.array(out)


def paired(a: np.ndarray, b: np.ndarray, qs: np.ndarray) -> dict:
    x = cluster_mean(a - b, qs)
    x["clearly_beats"] = bool(x.get("lo", -1) > 0)
    return x


def per_judge(t: pd.DataFrame, j: pd.DataFrame, own: dict) -> dict:
    t = t.join(j, on="event_id", how="inner")
    hh = t[t.kind.isin(["helpful", "harmful"])]
    y, qs = (hh.kind == "helpful").to_numpy(int), hh.question_id.to_numpy()
    p = hh.p_rev.to_numpy()
    sc = decision_score(p, y)
    lr = np.where(hh.accept_logistic.to_numpy() == y, 1.0, 0.0)
    gb = np.where(hh.accept_boosted_tree.to_numpy() == y, 1.0, 0.0)
    ow = own_rule_score(hh, own)
    comb = decision_score((p + hh.p_logistic.to_numpy()) / 2, y)
    res = {
        "events": int(len(t)), "helpful_harmful": int(len(hh)),
        "accuracy": {"pooled": cluster_mean(sc, qs), "helpful": cluster_mean(sc[y == 1], qs[y == 1]),
                     "harmful": cluster_mean(sc[y == 0], qs[y == 0])},
        "auroc": cluster_stat(auroc, qs, p, y),
        "mean_p_revision": {"all": cluster_mean(t.p_rev.to_numpy(), t.question_id.to_numpy()),
                            "helpful": cluster_mean(p[y == 1], qs[y == 1]), "harmful": cluster_mean(p[y == 0], qs[y == 0]),
                            "wrong_wrong": cluster_mean(t[t.kind == "wrong_wrong"].p_rev.to_numpy(),
                                                        t[t.kind == "wrong_wrong"].question_id.to_numpy())},
        "position_bias": cluster_mean(t.pos_bias.to_numpy(), t.question_id.to_numpy()),
        "comparators_accuracy": {"logistic_gate": cluster_mean(lr, qs), "boosted_tree_gate": cluster_mean(gb, qs),
                                 "own_answer_rule": cluster_mean(ow, qs), "combined_judge_logistic": cluster_mean(comb, qs)},
        "judge_minus": {"logistic_gate": paired(sc, lr, qs), "boosted_tree_gate": paired(sc, gb, qs),
                        "own_answer_rule": paired(sc, ow, qs)},
        "own_answers_cover": float(np.mean([r in own for r in hh.question_id])),
    }
    m = res["judge_minus"]
    res["decision_bar"] = {"a_accuracy_ge_0.75": bool(res["accuracy"]["pooled"]["mean"] >= 0.75),
                           "b_clearly_beats_both_oof_gates": bool(m["logistic_gate"]["clearly_beats"] and m["boosted_tree_gate"]["clearly_beats"]),
                           "c_clearly_beats_own_answer_rule": bool(m["own_answer_rule"]["clearly_beats"])}
    res["decision_bar"]["met"] = all(res["decision_bar"].values())
    res["by_source"] = {f"{d} {c}": {"helpful_harmful": int(len(g)), "accuracy": cluster_mean(
        decision_score(g.p_rev.to_numpy(), (g.kind == "helpful").to_numpy(int)), g.question_id.to_numpy())}
        for (d, c), g in hh.groupby(["dataset", "condition"])}
    return res


GATES = ["accept_all", "reject_all", "a_new_is_peer_majority", "b_conviction_reject_if_conf_before_ge_90",
         "c_mad_conformist_keep_if_old_is_prev_majority", "logistic", "boosted_tree", "judge", "combined", "own_answer",
         "perfect"]


def simulate(t: pd.DataFrame, root: Path, judges: dict, owns: dict) -> dict:
    """Complete sampled panels: every changed answer decided by the gate; unchanged answers kept."""
    ev = R.load_events(root)["events"]
    ev = ev[ev.question_id.isin(set(t.question_id))]
    ev = ev[ev.apply(lambda r: (r.dataset, r.condition) in R.TRAIN_SOURCES, axis=1)]
    tid = t.set_index("event_id")
    out = {}
    for name, j in judges.items():
        jp = j.p_rev.to_dict()
        oap = dict(zip(t.event_id, own_accept_prob(t, owns[name]), strict=True))
        judged_q = {e.split("|")[2] for e in jp}  # only the questions this judge saw (thinking: its subsample)
        rows = []
        for (d, c, q), g in ev[ev.question_id.isin(judged_q)].groupby(["dataset", "condition", "question_id"]):
            g = g.sort_values("agent_id")
            ids = [f"{d}|{c}|{q}|{a}" for a in g.agent_id]
            ch = [i for i, (b, a) in enumerate(zip(g.before, g.after, strict=True)) if b != a]
            if any(ids[i] not in jp for i in ch):
                continue  # this judge did not see every changed answer of the panel (thinking subsample)
            b, a, gold = g.before.tolist(), g.after.tolist(), int(g.gold.iloc[0])
            row = {"dataset": d, "condition": c, "question_id": q}
            for gate in GATES:
                if gate == "perfect":
                    row[gate] = R.perfect_gate(b, a, gold)[1]
                    continue
                p = []
                for i, e in enumerate(ids):
                    if i not in ch:
                        p.append(0.0)
                        continue
                    r = tid.loc[e]
                    if gate == "judge":
                        p.append(1.0 if jp[e] > 0.5 else 0.0 if jp[e] < 0.5 else 0.5)
                    elif gate == "combined":
                        v = (jp[e] + r.p_logistic) / 2
                        p.append(1.0 if v > 0.5 else 0.0 if v < 0.5 else 0.5)
                    elif gate == "own_answer":
                        p.append(oap[e])
                    elif gate in ("logistic", "boosted_tree"):
                        p.append(float(r[f"accept_{gate}"]))
                    else:
                        p.append(float(rule_accept(r)[gate]))
                row[gate] = R.panel_expectation(b, a, gold, p)[1]
            rows.append(row)
        s = pd.DataFrame(rows)
        qs = s.question_id.to_numpy()
        base = s.accept_all.to_numpy()
        out[name] = {"panels": len(s), **{g: {"panel": cluster_mean(s[g].to_numpy(), qs),
                                              "minus_accept_all": cluster_mean(s[g].to_numpy() - base, qs)} for g in GATES}}
    return out


def cross_model(t: pd.DataFrame, owns: dict) -> dict:
    gold = t.drop_duplicates("question_id").set_index("question_id").gold.to_dict()
    nopt = t.drop_duplicates("question_id").set_index("question_id").n_options.to_dict()
    out = {"own_accuracy": {}, "vs_gemma": {}, "mixed_panels": {}}
    for m, own in owns.items():
        acc = [np.mean([a == gold[q] for a in v]) for q, v in own.items() if q in gold and v]
        qs = [q for q, v in own.items() if q in gold and v]
        out["own_accuracy"][m] = cluster_mean(np.array(acc), np.array(qs))
        out["own_accuracy"][m]["parse_rate"] = float(np.mean([a >= 0 for v in own.values() for a in v]))
    G = owns["gemma"]
    for m in ("qwen3", "llama", "qwen3_think"):
        X = owns.get(m, {})
        both = [q for q in gold if G.get(q) and X.get(q)]
        a_g = np.array([np.mean([a == gold[q] for a in G[q]]) for q in both])
        a_x = np.array([np.mean([a == gold[q] for a in X[q]]) for q in both])
        same, pairs, expect = [], [], []
        for q in both:
            wg = [a for a in G[q] if a >= 0 and a != gold[q]]
            wx = [a for a in X[q] if a >= 0 and a != gold[q]]
            n = len(wg) * len(wx)
            same.append(sum(a == b for a in wg for b in wx))
            pairs.append(n)
            expect.append(n / (nopt[q] - 1))
        same, pairs, expect = np.array(same, float), np.array(pairs, float), np.array(expect, float)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(both), (N_BOOT, len(both)))
        corr = [np.corrcoef(a_g[i], a_x[i])[0, 1] for i in idx]
        rate = same[idx].sum(1) / np.maximum(pairs[idx].sum(1), 1)
        out["vs_gemma"][m] = {
            "questions": len(both),
            "correlation_of_per_question_accuracy": {"mean": float(np.corrcoef(a_g, a_x)[0, 1]),
                                                     "lo": float(np.nanquantile(corr, .025)), "hi": float(np.nanquantile(corr, .975))},
            "same_wrong_answer_rate": {"mean": float(same.sum() / max(pairs.sum(), 1)), "lo": float(np.quantile(rate, .025)),
                                       "hi": float(np.quantile(rate, .975)), "wrong_pairs": int(pairs.sum())},
            "uniform_expected_rate": float(expect.sum() / max(pairs.sum(), 1))}
    Q, L = owns.get("qwen3", {}), owns.get("llama", {})
    qs = [q for q in gold if len(G.get(q, [])) >= 3 and len(Q.get(q, [])) >= 3 and len(L.get(q, [])) >= 3]

    def avg(panels):
        return float(np.mean([expected_vote(list(p), gold_q) for p, gold_q in panels]))

    scores = {k: [] for k in ("GGG", "GGQ", "GGL", "GQL")}
    for q in qs:
        g, qq, ll, gd = G[q][:3], Q[q][:3], L[q][:3], gold[q]
        scores["GGG"].append(expected_vote(g, gd))
        scores["GGQ"].append(avg([((*gg, x), gd) for gg in itertools.combinations(g, 2) for x in qq]))
        scores["GGL"].append(avg([((*gg, x), gd) for gg in itertools.combinations(g, 2) for x in ll]))
        scores["GQL"].append(avg([((a, b, c), gd) for a in g for b in qq for c in ll]))
    qa = np.array(qs)
    out["mixed_panels"] = {"questions": len(qs), **{k: cluster_mean(np.array(v), qa) for k, v in scores.items()},
                           **{f"{k}_minus_GGG": cluster_mean(np.array(scores[k]) - np.array(scores["GGG"]), qa)
                              for k in ("GGQ", "GGL", "GQL")}}
    return out


def analyze(runs: Path, root: Path) -> dict:
    t = events_table(root)
    judges = {m: j for m in JUDGES if (j := load_judge(runs, m)) is not None}
    owns = {m: load_own(runs, m, root) for m in JUDGES}
    res = {"label": LABEL, "plan": "docs/REVISION_JUDGE_PLAN.md", "judges": {}, "missing": [m for m in JUDGES if m not in judges]}
    for m, j in judges.items():
        res["judges"][m] = per_judge(t, j, owns[m])
    if "qwen3_think" in judges and "qwen3" in judges:
        sub = t[t.thinking_subsample]
        res["thinking_vs_nonthinking_same_events"] = {
            "qwen3_think": per_judge(sub, judges["qwen3_think"], owns["qwen3_think"])["accuracy"]["pooled"],
            "qwen3": per_judge(sub, judges["qwen3"], owns["qwen3"])["accuracy"]["pooled"]}
    res["panels"] = simulate(t, root, judges, owns)
    if owns.get("gemma") and owns.get("qwen3") and owns.get("llama"):
        res["cross_model"] = cross_model(t, owns)
    return res


def ci(x, nd=3):
    return "—" if not x or "mean" not in x else f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"


def markdown(r: dict) -> str:
    L = [f"# Cross-model judge of revisions: results ({LABEL})", "",
         f"As committed in `{r['plan']}`. Question-grouped 95% intervals (10,000 resamples). Limitations: possible "
         "MMLU-Pro contamination in Qwen/Llama; events from Gemma's revisions only; the revision is identifiable from "
         "its content (98.4% mention the panel); the 149 questions skew hard and disputed; events mix 150- and 250-word "
         "protocols." + (f" Missing judges: {r['missing']}." if r["missing"] else ""), "",
         "## Per judge (helpful/harmful events)", "",
         "| judge | events | pooled accuracy | on helpful | on harmful | AUROC | mean P(revision): all / helpful / harmful / wrong→wrong | position bias |",
         "|---|---|---|---|---|---|---|---|"]
    for m, v in r["judges"].items():
        mp = v["mean_p_revision"]
        L.append(f"| {m} | {v['helpful_harmful']} | {ci(v['accuracy']['pooled'])} | {ci(v['accuracy']['helpful'])} | "
                 f"{ci(v['accuracy']['harmful'])} | {ci(v['auroc'])} | {mp['all']['mean']:.3f} / {mp['helpful']['mean']:.3f} / "
                 f"{mp['harmful']['mean']:.3f} / {mp['wrong_wrong']['mean']:.3f} | {ci(v['position_bias'])} |")
    L += ["", "## Decision bar", "",
          "| judge | (a) accuracy ≥ 0.75 | judge − logistic gate | judge − boosted-tree gate | judge − own-answer rule | (b) | (c) | met |",
          "|---|---|---|---|---|---|---|---|"]
    for m, v in r["judges"].items():
        d, jm = v["decision_bar"], v["judge_minus"]
        L.append(f"| {m} | {'yes' if d['a_accuracy_ge_0.75'] else 'no'} ({v['accuracy']['pooled']['mean']:.3f}) | "
                 f"{ci(jm['logistic_gate'])} | {ci(jm['boosted_tree_gate'])} | {ci(jm['own_answer_rule'])} | "
                 f"{'yes' if d['b_clearly_beats_both_oof_gates'] else 'no'} | {'yes' if d['c_clearly_beats_own_answer_rule'] else 'no'} | "
                 f"**{'MET' if d['met'] else 'not met'}** |")
    L += ["", "Comparator accuracy on the same events: " + "; ".join(
        f"{m}: logistic {v['comparators_accuracy']['logistic_gate']['mean']:.3f}, boosted {v['comparators_accuracy']['boosted_tree_gate']['mean']:.3f}, "
        f"own-answer {v['comparators_accuracy']['own_answer_rule']['mean']:.3f}, combined {v['comparators_accuracy']['combined_judge_logistic']['mean']:.3f}"
        for m, v in r["judges"].items()) + "."]
    if "thinking_vs_nonthinking_same_events" in r:
        x = r["thinking_vs_nonthinking_same_events"]
        L += ["", f"Thinking subsample (same events): Qwen3 thinking {ci(x['qwen3_think'])}, Qwen3 non-thinking {ci(x['qwen3'])}."]
    L += ["", "## Simulated gated panels (complete sampled panels; panel accuracy − accept-all)", "",
          "| gate | " + " | ".join(r["panels"]) + " |", "|---|" + "---|" * len(r["panels"])]
    for g in GATES:
        L.append(f"| {g} | " + " | ".join(ci(v[g]["minus_accept_all"]) for v in r["panels"].values()) + " |")
    L.append("| panels | " + " | ".join(str(v["panels"]) for v in r["panels"].values()) + " |")
    if "cross_model" in r:
        c = r["cross_model"]
        L += ["", "## Cross-model (own answers, 250-word protocol)", "",
              "Own accuracy on these questions: " + "; ".join(f"{m} {ci(v)} (parse {v['parse_rate']:.2f})" for m, v in c["own_accuracy"].items()) + ".", "",
              "| vs Gemma | questions | correlation of per-question accuracy | same-wrong-answer rate | uniform expectation |", "|---|---|---|---|---|"]
        for m, v in c["vs_gemma"].items():
            L.append(f"| {m} | {v['questions']} | {ci(v['correlation_of_per_question_accuracy'])} | "
                     f"{ci(v['same_wrong_answer_rate'])} ({v['same_wrong_answer_rate']['wrong_pairs']} pairs) | {v['uniform_expected_rate']:.3f} |")
        mp = c["mixed_panels"]
        L += ["", f"Mixed 3-answer panels ({mp['questions']} questions; exact expected accuracy averaged over sample choices): "
              + ", ".join(f"{k} {ci(mp[k])}" for k in ("GGG", "GGQ", "GGL", "GQL")) + "; differences from GGG: "
              + ", ".join(f"{k} {ci(mp[k + '_minus_GGG'])}" for k in ("GGQ", "GGL", "GQL")) + "."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    r = analyze(Path(a.runs), ROOT)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(r, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    (out / "results.md").write_text(markdown(r))
    print(markdown(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
