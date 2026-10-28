"""Reproducible transcript sample for the prompt overhaul (validation tranche, seed 2026).

    python scripts/transcript_sample.py   # writes artifacts/val12b/transcript_review/sample.md and sample_index.csv

Six cases each: helpful debate changes (wrong -> right), harmful changes (right -> wrong),
unchanged correct, unchanged wrong (debate), and truncated responses (hit the 768-token cap;
any state). Debate cases show the agent's first response, the full peer text it was shown,
and the revised response. The hand coding of the sample is in coding.csv; codebook in
docs/PROBE_ACTION_TRANSCRIPT_REVIEW.md.

Also writes population_stats.json: counts over ALL 600 debate revisions that can be computed
from the data without hand coding (answer adoption, capped peers, confidence changes, and
lexical markers -- markers measure wording, not whether a check was performed).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/val12b/transcript_review"
L = "ABCDEFGHIJ"
SEED, N = 2026, 6


def main():
    qs = {str(json.loads(line)["question_id"]): json.loads(line) for line in open(ROOT / "runs/val12b/val12b-questions.jsonl")}
    T = {}
    for c in ("debate", "independent"):
        for line in open(ROOT / f"runs/val12b/val12b-{c}/transcripts.jsonl"):
            r = json.loads(line)
            T[(str(r["question_id"]), r["agent_id"], r["phase"], "round0" if r["phase"] == "pre" else c)] = r
    st = pd.concat([pd.read_parquet(ROOT / f"runs/val12b/val12b-{c}/agent_states.parquet") for c in ("debate", "independent")])
    st = st[st.phase.isin(["pre", "post"])].copy()
    st["q"] = st.question_id.astype(str)
    st["cond"] = st.condition.where(st.phase == "post", "round0")
    st = st.drop_duplicates(["q", "agent_id", "phase", "cond"])
    pre = st[st.cond == "round0"].set_index(["q", "agent_id"])
    deb = st[st.cond == "debate"].copy()
    deb["pre_answer"] = [pre.loc[(q, a)].answer for q, a in zip(deb.q, deb.agent_id, strict=True)]
    deb["pre_correct"] = [pre.loc[(q, a)].correct for q, a in zip(deb.q, deb.agent_id, strict=True)]
    ch = deb.answer != deb.pre_answer
    cats = {"helpful": deb[ch & (deb.correct == 1)], "harmful": deb[ch & (deb.pre_correct == 1) & (deb.correct == 0)],
            "unchanged_correct": deb[~ch & (deb.correct == 1)], "unchanged_wrong": deb[~ch & (deb.correct == 0)]}
    stop = {k: T[k]["stop_reason"] for k in T}
    trunc = st[[stop[(q, a, p, c)] == "length" for q, a, p, c in zip(st.q, st.agent_id, st.phase, st.cond, strict=True)]]
    cats["truncated"] = trunc
    pop = {k: len(v) for k, v in cats.items()}
    rows, md = [], [f"# Transcript sample (seed {SEED}; {N} per category)\n", f"Population sizes: {pop}\n"]
    for cat, df in cats.items():
        for i, (_, r) in enumerate(df.sample(min(N, len(df)), random_state=SEED).iterrows(), 1):
            q = qs[r.q]
            perm = [int(x) for x in r.permutation.split(",")]
            opts = [q["options"][j] for j in perm]
            key = f"{cat}-{i}"
            t = T[(r.q, r.agent_id, r.phase, r.cond)]
            rows.append({"case": key, "category": cat, "question_id": r.q, "agent_id": int(r.agent_id), "phase": r.phase,
                         "condition": r.cond, "subject": r.subject, "answer": L[int(r.answer)] if r.answer >= 0 else "?",
                         "gold": L[int(r.gold_display)], "tokens": len(t["token_ids"]),
                         "answer_source": r.answer_source, "verbalized_confidence": r.verbalized_confidence})
            if cat != "truncated":
                f0 = T[(r.q, r.agent_id, "pre", "round0")]
                peers = [int(x) for x in str(r.shown_peer_sources).replace("agent", "").split(",")]
                rows[-1].update({
                    "first_answer": L[int(r.pre_answer)], "first_confidence": pre.loc[(r.q, r.agent_id)].verbalized_confidence,
                    "first_capped": int(f0["stop_reason"] == "length"),
                    "peer_answers": ",".join(L[int(pre.loc[(r.q, p)].answer)] for p in peers),
                    "peers_capped": sum(T[(r.q, p, "pre", "round0")]["stop_reason"] == "length" for p in peers),
                    "answer_equals_a_peer": int(any(int(pre.loc[(r.q, p)].answer) == int(r.answer) for p in peers))})
            md.append(f"\n---\n## {key} | {r.subject} | q{r.q} agent {r.agent_id} | {r.cond} {r.phase} | "
                      f"answer {rows[-1]['answer']} | correct {rows[-1]['gold']} | {rows[-1]['tokens']} tokens\n")
            md.append("**Question:** " + q["question"].strip()[:600] + "\n\n" + "\n".join(f"- {L[j]}. {o[:120]}" for j, o in enumerate(opts)))
            if cat != "truncated":
                first = T[(r.q, r.agent_id, "pre", "round0")]["text"].strip()
                md.append(f"\n\n**First response** (answer {L[int(r.pre_answer)]}):\n\n> " + first.replace("\n", "\n> "))
                user = [m["content"] for m in t["messages"] if m["role"] == "user"][-1]
                peer = user[user.find("Panel member"):] if "Panel member" in user else user[-1500:]
                md.append("\n\n**Shown to the agent (peers, full text):**\n\n> " + peer.replace("\n", "\n> "))
            md.append("\n\n**Response:**\n\n> " + t["text"].strip().replace("\n", "\n> "))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.md").write_text("\n".join(md) + "\n")
    pd.DataFrame(rows).to_csv(OUT / "sample_index.csv", index=False)
    stats = population_stats(deb, pre, T, pop)
    (OUT / "population_stats.json").write_text(json.dumps(stats, indent=1))
    print(pop, len(rows))
    print(json.dumps(stats, indent=1))


PRAISE = re.compile(r"compelling|reinforces|valid point|crucial point|i concede|concede to|raises a|correctly (?:identif|point|highlight|calculat|derive)|"
                    r"is sound|are sound|strengthens|solidif", re.I)
EQUATION = re.compile(r"=\s*[-+(]?\$?\d")  # at least one "= <number>": the response writes out some arithmetic


def population_stats(deb, pre, T, pop):
    """Data-derived counts over all debate revisions (no hand coding)."""
    d = deb.copy()
    shown = [[int(x) for x in str(s).split(",")] for s in d.shown_peer_answers]
    d["n_visible"] = [sum(a >= 0 for a in s) for s in shown]
    d["to_visible_peer"] = [int(a in [x for x in s if x >= 0]) for a, s in zip(d.answer, shown, strict=True)]
    d["changed"] = (d.answer != d.pre_answer).astype(int)
    d["both_visible_agree_against"] = [int(len(s) == 2 and s[0] >= 0 and s[0] == s[1] and s[0] != p) for s, p in zip(shown, d.pre_answer, strict=True)]
    d["split_visible"] = [int(len(s) == 2 and min(s) >= 0 and s[0] != s[1] and p in s) for s, p in zip(shown, d.pre_answer, strict=True)]
    d["both_visible_agree_with"] = [int(len(s) == 2 and s[0] >= 0 and s[0] == s[1] == p) for s, p in zip(shown, d.pre_answer, strict=True)]
    text = [T[(q, a, "post", "debate")]["text"] for q, a in zip(d.q, d.agent_id, strict=True)]
    d["praise"] = [int(bool(PRAISE.search(t))) for t in text]
    d["equation"] = [int(bool(EQUATION.search(t))) for t in text]
    d["pre_conf"] = [pre.loc[(q, a)].verbalized_confidence for q, a in zip(d.q, d.agent_id, strict=True)]
    ch = d[d.changed == 1]
    kind = np.where(d.changed == 0, np.where(d.correct == 1, "unchanged_correct", "unchanged_wrong"),
                    np.where(d.correct == 1, "helpful", np.where(d.pre_correct == 1, "harmful", "wrong_to_wrong")))
    d["kind"] = kind
    out = {"population_sizes_sampled_categories": pop,
           "debate_revisions": int(len(d)),
           "changed": int(len(ch)),
           "changed_to_an_answer_a_visible_peer_held": f"{int(ch.to_visible_peer.sum())}/{len(ch)}",
           "revisions_with_a_capped_peer_no_answer_shown": f"{int((d.n_visible < 2).sum())}/{len(d)}",
           "switch_rate_both_peers_agree_against_you": f"{int(d.changed[d.both_visible_agree_against == 1].sum())}/{int(d.both_visible_agree_against.sum())}",
           "switch_rate_peers_split_one_agrees_with_you": f"{int(d.changed[d.split_visible == 1].sum())}/{int(d.split_visible.sum())}",
           "switch_rate_both_peers_agree_with_you": f"{int(d.changed[d.both_visible_agree_with == 1].sum())}/{int(d.both_visible_agree_with.sum())}",
           "by_kind": {}}
    for k, g in d.groupby("kind"):
        out["by_kind"][k] = {"n": int(len(g)), "praise_or_deference_wording": f"{int(g.praise.sum())}/{len(g)}",
                             "writes_out_arithmetic": f"{int(g.equation.sum())}/{len(g)}"}
    for lab, m in (("both_peers_agree_against_you", d.both_visible_agree_against == 1), ("peers_split", d.split_visible == 1)):
        g = d[m & (d.changed == 1)]
        out[f"switches_when_{lab}"] = {"n": int(len(g)), "to_a_visible_peer_answer": int(g.to_visible_peer.sum()),
                                       **{k: int((g.kind == k).sum()) for k in ("helpful", "harmful", "wrong_to_wrong")}}
    for lab, m in (("both_peers_agree_with_you", d.both_visible_agree_with == 1), ("peers_split", d.split_visible == 1),
                   ("both_peers_agree_against_you", d.both_visible_agree_against == 1)):
        g = d[m & d.pre_conf.notna() & d.verbalized_confidence.notna()]
        out[f"confidence_change_{lab}"] = {"n": int(len(g)), "mean_post_minus_first": round(float((g.verbalized_confidence - g.pre_conf).mean()), 2),
                                            "share_increased": round(float((g.verbalized_confidence > g.pre_conf).mean()), 3)}
    calc = d.subject.isin(["math", "physics", "chemistry", "engineering"])
    out["calculation_subjects"] = {"n": int(calc.sum()), "writes_out_arithmetic": f"{int(d.equation[calc].sum())}/{int(calc.sum())}"}
    return out


if __name__ == "__main__":
    main()
