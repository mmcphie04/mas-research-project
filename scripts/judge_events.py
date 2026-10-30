"""Fixed, outcome-blind event sample for the cross-model judge (revision gate, section 5).

    python scripts/judge_events.py --out artifacts/revision_gate/judge

Population: every CHANGED answer in the gate's sources (revisions after seeing peers, all
Gemma-3-12B datasets, main250 test excluded; ``revision_events.TRAIN_SOURCES``).

Rule (fixed before any judge output exists): QUESTIONS are drawn, not events, so every sampled
panel is complete (all its changed answers are judged) and the gated panel can be simulated.
``n_questions`` = round(400 / (helpful+harmful events per question in the population)), drawn by
numpy.random.default_rng(seed).permutation over the sorted question ids; every changed answer of
every source on a drawn question is included. The thinking subsample: the drawn questions in a
second permutation (seed + 1), taken in order until they hold at least ``thinking_events``
events (counts of events only). Selection never reads a judge output, and no per-question
outcome is used.

Writes events.jsonl (texts the judge sees; NO labels), questions.jsonl (question blocks for the
own-answer runs) and manifest.json (rule, seeds, counts, sha256 of both files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from mad.analysis import revision_events as R  # noqa: E402
from mad.prompts import strip_confidence  # noqa: E402

PREFIX = {s[0] + "|" + c: (s[1], s[2]) for s in R.SOURCES for c in s[4]}


def _transcripts(root: Path, dataset: str, cond: str) -> dict:
    prefix, job = PREFIX[f"{dataset}|{cond}"]
    out = {}
    for line in open(root / prefix / f"{job}-{cond}" / "transcripts.jsonl"):
        r = json.loads(line)
        if r["phase"] in ("pre", "post"):
            out[(str(r["question_id"]), int(r["agent_id"]), r["phase"])] = r
    return out


def build(root: Path, seed: int, target_hh: int, thinking_events: int) -> dict:
    ev = R.load_events(root)["events"]
    pop = ev[ev.apply(lambda r: (r.dataset, r.condition) in R.TRAIN_SOURCES, axis=1) & (ev.changed == 1)]
    qs = np.array(sorted(pop.question_id.unique()))
    rate = pop.kind.isin(["helpful", "harmful"]).sum() / len(qs)
    n_q = int(round(target_hh / rate))
    drawn = qs[np.random.default_rng(seed).permutation(len(qs))[:n_q]]
    sample = pop[pop.question_id.isin(set(drawn.tolist()))].sort_values(["question_id", "dataset", "condition", "agent_id"])
    per_q = sample.groupby("question_id").size()
    thinking_q, n_ev = set(), 0
    for q in drawn[np.random.default_rng(seed + 1).permutation(len(drawn))]:
        if n_ev >= thinking_events:
            break
        thinking_q.add(q)
        n_ev += int(per_q[q])
    cache, events, questions = {}, [], {}
    for r in sample.itertuples(index=False):
        key = (r.dataset, r.condition)
        if key not in cache:
            cache[key] = _transcripts(root, *key)
        t = cache[key]
        pre, post = t[(r.question_id, r.agent_id, "pre")], t[(r.question_id, r.agent_id, "post")]
        block = next(m["content"] for m in pre["messages"] if m["role"] == "user")
        events.append({"event_id": f"{r.dataset}|{r.condition}|{r.question_id}|{r.agent_id}", "dataset": r.dataset,
                       "condition": r.condition, "question_id": r.question_id, "agent_id": int(r.agent_id),
                       "n_options": int(r.n_options), "protocol": int(r.protocol), "question_block": block,
                       "original": strip_confidence(pre["text"]), "revised": strip_confidence(post["text"]),
                       "thinking_subsample": r.question_id in thinking_q})
        questions.setdefault(r.question_id, {"question_id": r.question_id, "n_options": int(r.n_options),
                                             "question_block": block, "thinking_subsample": r.question_id in thinking_q})
    counts = sample.groupby(["dataset", "condition"]).kind.value_counts().unstack(fill_value=0)
    return {"events": events, "questions": list(questions.values()), "sample": sample, "n_q": n_q, "rate": rate,
            "counts": counts, "population_questions": len(qs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--target-hh", type=int, default=400)
    ap.add_argument("--thinking-events", type=int, default=150)
    a = ap.parse_args(argv)
    b = build(ROOT, a.seed, a.target_hh, a.thinking_events)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ef, qf = out / "events.jsonl", out / "questions.jsonl"
    ef.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in b["events"]))
    qf.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in b["questions"]))
    s = b["sample"]
    manifest = {
        "label": "EXPLORATORY",
        "rule": __doc__.split("Rule (fixed before any judge output exists): ")[1].split("\n\nWrites")[0].replace("\n", " "),
        "seed": a.seed, "thinking_seed": a.seed + 1, "target_helpful_harmful": a.target_hh,
        "population": {"questions": b["population_questions"], "helpful_harmful_per_question": round(float(b["rate"]), 4)},
        "n_questions": b["n_q"], "n_events": len(b["events"]),
        "events_by_kind": s.kind.value_counts().to_dict(),
        "thinking_subsample": {"questions": int(sum(q["thinking_subsample"] for q in b["questions"])),
                               "events": int(sum(e["thinking_subsample"] for e in b["events"]))},
        "events_by_source": {f"{d} {c}": int(n) for (d, c), n in s.groupby(["dataset", "condition"]).size().items()},
        "files": {ef.name: hashlib.sha256(ef.read_bytes()).hexdigest(), qf.name: hashlib.sha256(qf.read_bytes()).hexdigest()},
        "deviation": "questions (complete panels) are sampled rather than ~400 helpful/harmful + ~100 wrong->wrong "
                     "events, so the gated panel can be simulated; wrong->wrong events are therefore ~250, not ~100",
        "limitation_identifiable_revision": "share of responses mentioning panel/peer/member: revised "
            f"{sum(bool(re.search(r'panel|peer|member', e['revised'], re.I)) for e in b['events']) / len(b['events']):.3f}, "
            f"original {sum(bool(re.search(r'panel|peer|member', e['original'], re.I)) for e in b['events']) / len(b['events']):.3f}: "
            "without labels the revision is still usually identifiable from its content",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_questions", "n_events", "events_by_kind", "thinking_subsample", "files")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
