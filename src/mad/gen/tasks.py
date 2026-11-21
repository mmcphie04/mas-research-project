"""Request builders for Experiments A and B and the judge confirmation (docs/EXPAB_PLAN.md).

Prompts reuse the project's protocol: the round-0 system prompt and persona, the 250-word
brevity line, the per-question option permutation (``data.permutation_seed`` 12345), the peer
block. Every request's seed is ``engine.seed_for(question, agent, condition)``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from mad import prompts as P
from mad.data import Question
from mad.gen import prompts_ab as PA
from mad.gen.engine import Request, seed_for
from mad.job import question_sha256
from mad.permute import LETTERS, make_permutation

PERSONA = "You are a careful, methodical reasoner."
PERMUTATION_SEED = 12345


# ----------------------------------------------------------------------------- questions


def load_questions(plan_dir, ext_dir=None, which: str = "all") -> tuple[list[Question], dict, dict]:
    """The committed question snapshots, each checked against its manifest: the original 1,500
    (``plan_dir``) and, when present, the 2,000-question held-out extension (``ext_dir``, default
    ``<plan_dir>/../plan_ext``). ``which``: all | original | extension. Returns (questions,
    assignment {qid: dev|heldout} over ALL questions, manifest with both sets' hashes)."""
    plan_dir = Path(plan_dir)
    ext_dir = Path(ext_dir) if ext_dir else plan_dir.parent / "plan_ext"
    qs, assign, m = _load_one(plan_dir)
    sets = {"original": qs}
    info = {"original": {k: m[k] for k in ("question_ids_sha256", "assignment_sha256", "snapshot_sha256")}}
    if (ext_dir / "manifest.json").exists():
        eq, ea, em = _load_one(ext_dir)
        if em["extends"]["question_ids_sha256"] != m["question_ids_sha256"] or set(ea) & set(assign):
            raise SystemExit("the extension does not extend this question set")
        sets["extension"] = eq
        assign = {**assign, **ea}
        info["extension"] = {k: em[k] for k in ("question_ids_sha256", "assignment_sha256", "snapshot_sha256")}
    chosen = [q for name, v in sets.items() if which in ("all", name) for q in v]
    if which != "all" and which not in sets:
        raise SystemExit(f"no question set {which!r}")
    combined = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    return chosen, assign, {**m, "question_ids_sha256": combined, "sets": info, "which": which}


def _load_one(plan_dir: Path) -> tuple[list[Question], dict, dict]:
    m = json.loads((plan_dir / "manifest.json").read_text())
    snap = plan_dir / "questions.jsonl"
    if hashlib.sha256(snap.read_bytes()).hexdigest() != m["snapshot_sha256"]:
        raise SystemExit("question snapshot does not match its manifest")
    qs = [Question(**{**d, "options": tuple(d["options"])}) for d in map(json.loads, snap.read_text().splitlines())]
    if [q.question_id for q in qs] != m["question_ids"]:
        raise SystemExit("question order differs from the manifest")
    for q in qs:
        if question_sha256(q) != m["question_content_sha256"][q.question_id]:
            raise SystemExit(f"{q.question_id}: content hash differs from the manifest")
    return qs, m["assignment"], m


def gold_display(q: Question) -> int:
    return make_permutation(q.n_options, q.question_id, PERMUTATION_SEED).original_to_display(q.gold_index)


def round0(q: Question, visible_rationale: bool = False) -> list[dict]:
    """The main collection's round-0 prompt. ``visible_rationale`` (thinking mode only, a recorded
    protocol choice) appends prompts_ab.VISIBLE_RATIONALE to the system prompt."""
    perm = make_permutation(q.n_options, q.question_id, PERMUTATION_SEED)
    msgs = P.round0_messages(persona=PERSONA, subject=q.subject, question=q.question, options=list(q.options),
                             perm=perm, ask_confidence=True, brevity=P.BREVITY_250)
    if visible_rationale:
        msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + PA.VISIBLE_RATIONALE}
    return msgs


def question_block(q: Question) -> str:
    return round0(q)[1]["content"]


# ----------------------------------------------------------------------------- Experiment A: the bank

BANK = {"gemma": 10, "qwen_nt": 10, "qwen_t": 10}  # thinking: 10 (review of Stop 2; was 5, then 8)


def bank_requests(qs: list[Question], mode: str, n: int | None = None, visible_rationale: bool = False) -> list[Request]:
    n = n or BANK[mode]
    vis = visible_rationale and mode == "qwen_t"
    return [Request(f"{q.question_id}|{a}", q.question_id, a, f"bank/{mode}", q.n_options, messages=round0(q, vis),
                    meta={"gold": gold_display(q)}) for q in qs for a in range(n)]


# ----------------------------------------------------------------------------- Experiment A: choosers

# arm -> (chooser mode, pool: [(bank mode, first k samples)])
ARMS = {
    "A1_qt5": ("qwen_t", [("qwen_t", 5)]),
    "A1_qnt5": ("qwen_nt", [("qwen_nt", 5)]),
    "A1_qnt10": ("qwen_nt", [("qwen_nt", 10)]),
    "A1_g5": ("gemma", [("gemma", 5)]),
    "A1_g10": ("gemma", [("gemma", 10)]),
    "A2_5": ("qwen_t", [("qwen_nt", 5)]),
    "A2_10": ("qwen_t", [("qwen_nt", 10)]),
    "A3": ("qwen_t", [("gemma", 5), ("qwen_nt", 5)]),
}
RUNS = 3
BEST_RE = re.compile(r"Best candidate\W{0,3}:?\W{0,3}(\d{1,2})", re.IGNORECASE)


def pool_rows(bank: dict, qid: str, spec) -> list[dict]:
    """bank: {(mode, qid): [rows sorted by agent]} -> the pool's samples (parsed or not)."""
    out = []
    for mode, k in spec:
        out += [dict(r, pool_mode=mode) for r in bank.get((mode, qid), [])[:k]]
    return out


def candidates(pool: list[dict], qid: str, arm: str) -> list[dict]:
    """One representative rationale per distinct parsed answer letter, drawn by seed among the
    samples with that letter. The text shown is the post-thinking answer text with the confidence
    line removed. Vote counts are never shown."""
    by = {}
    for r in pool:
        if r["parsed"]:
            by.setdefault(int(r["answer"]), []).append(r)
    rng = np.random.default_rng(seed_for(qid, "rep", arm))
    out = []
    for letter in sorted(by):
        group = sorted(by[letter], key=lambda r: (r["pool_mode"], int(r["agent"])))
        pick = group[int(rng.integers(0, len(group)))]
        out.append({"answer": letter, "text": P.strip_confidence(pick["text"]), "source": f"{pick['pool_mode']}:{pick['agent']}",
                    "votes": len(group)})
    return out


def chooser_requests(q: Question, arm: str, cands: list[dict]) -> list[Request]:
    """RUNS requests, each with the candidates in its own seeded order; the answer is an index."""
    m = len(cands)
    reqs = []
    for run in range(RUNS):
        order = np.random.default_rng(seed_for(q.question_id, f"order{run}", arm)).permutation(m).tolist()
        body = "\n\n".join(PA.CANDIDATE_ENTRY.format(i=i + 1, text=cands[j]["text"]) for i, j in enumerate(order))
        user = PA.CHOOSER_USER.format(question_block=question_block(q), m=m, candidates=body)
        reqs.append(Request(f"{q.question_id}|{arm}|{run}", q.question_id, run, f"choose/{arm}", q.n_options,
                            messages=[{"role": "system", "content": PA.CHOOSER_SYSTEM}, {"role": "user", "content": user}],
                            meta={"arm": arm, "run": run, "order_answers": json.dumps([cands[j]["answer"] for j in order]),
                                  "m": m, "gold": gold_display(q)}))
    return reqs


def parse_pick(text: str, m: int) -> int:
    """The LAST 'Best candidate: N' with 1 <= N <= m, as a 1-based position; -1 if none."""
    for mt in reversed(list(BEST_RE.finditer(text or ""))):
        n = int(mt.group(1))
        if 1 <= n <= m:
            return n
    return -1


def disagreeing(pool: list[dict]) -> bool:
    return len({int(r["answer"]) for r in pool if r["parsed"]}) >= 2


# ----------------------------------------------------------------------------- Experiment B: debate arms

B_ARMS = ("debate", "alone", "dissent", "anon")


TRIPLES = (0, 1, 2)  # disjoint sample triples 0-2, 3-5, 6-8 (review of Stop 2: panel supply)


def panel(bank: dict, mode: str, qid: str, triple: int = 0) -> list[dict] | None:
    """Bank samples 3t..3t+2 of this mode; None unless all three parsed and not unanimous
    (parse status and answers only -- no gold answer is used)."""
    rows = bank.get((mode, qid), [])[3 * triple: 3 * triple + 3]
    if len(rows) < 3 or not all(r["parsed"] for r in rows):
        return None
    return rows if len({int(r["answer"]) for r in rows}) >= 2 else None


def b_requests(q: Question, mode: str, rows: list[dict], arm: str, visible_rationale: bool = False,
               triple: int = 0) -> list[Request]:
    """One request per agent. Own previous response: its (post-thinking) answer text, as an
    assistant turn. Peers: the other two agents' answer texts, confidence lines removed.
    debate/dissent: fixed agent order, labelled 'Panel member <agent number>'; anon: seeded random
    order, labelled 'Response A/B'; alone: no peers."""
    base = round0(q, visible_rationale and mode == "qwen_t")  # the same round-0 prompt the bank used
    out = []
    pos = {int(p["agent"]): i for i, p in enumerate(rows)}  # panel position 0-2 of each bank sample
    for r in rows:
        a = int(r["agent"])  # the bank sample index (distinct across triples: distinct seeds)
        peers = [p for p in rows if int(p["agent"]) != a]
        cond = f"b/{mode}/{arm}"
        if arm == "anon":
            order = np.random.default_rng(seed_for(q.question_id, a, cond + "/order")).permutation(len(peers)).tolist()
            peers = [peers[i] for i in order]
            entries = [PA.PEER_ENTRY_ANON.format(label=LETTERS[i], text=P.strip_confidence(p["text"]))
                       for i, p in enumerate(peers)]
        else:
            entries = [PA.PEER_ENTRY_NAMED.format(num=pos[int(p["agent"])] + 1, text=P.strip_confidence(p["text"]))
                       for p in peers]
        fmt = {"brevity": P.BREVITY_250, "ending": P.ENDING_CONF}
        if arm == "alone":
            block = P.SELF_BLOCK.format(**fmt)
        elif arm == "dissent":
            block = PA.DISSENT_BLOCK.format(peers="\n\n".join(entries), **fmt)
        else:
            block = P.PEER_BLOCK.format(peers="\n\n".join(entries), **fmt)
        msgs = [*base, {"role": "assistant", "content": r["text"].strip()}, {"role": "user", "content": block}]
        out.append(Request(f"{q.question_id}|{mode}|{arm}|{triple}|{a}", q.question_id, a, cond, q.n_options, messages=msgs,
                           meta={"arm": arm, "triple": triple, "before": int(r["answer"]), "gold": gold_display(q),
                                 "shown_sources": json.dumps([] if arm == "alone" else [int(p["agent"]) for p in peers]),
                                 "shown_answers": json.dumps([] if arm == "alone" else [int(p["answer"]) for p in peers])}))
    return out


def select_panels(bank: dict, qs: list[Question], assignment: dict, split: str, mode: str, n: int | None,
                  seed: int) -> list[tuple[str, int]]:
    """Disputed panels (question, triple) of ``mode`` on ``split`` questions, over every disjoint
    sample triple, in a seeded order; the first n."""
    ids = [(q.question_id, t) for q in qs if assignment[q.question_id] == split for t in TRIPLES
           if panel(bank, mode, q.question_id, t)]
    ids = [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]
    return ids if n is None else ids[:n]


# ----------------------------------------------------------------------------- judge confirmation


def judge_confirm_requests(events: list[dict], questions: list[dict]) -> list[Request]:
    from mad import judge as J

    reqs = []
    for e in events:
        for order, msgs in J.both_orders(e).items():
            reqs.append(Request(f"{e['event_id']}|{order}", e["question_id"], order, "judge_confirm/judge", e["n_options"],
                                messages=msgs, meta={"event_id": e["event_id"], "order": order}))
    for q in questions:
        for s in range(3):
            reqs.append(Request(f"{q['question_id']}|own|{s}", q["question_id"], s, "judge_confirm/own", q["n_options"],
                                messages=J.own_messages(q["question_block"]), meta={"sample": s}))
    return reqs


def bank_index(rows) -> dict:
    """{(mode, qid): [row dicts sorted by agent]} from bank rows (DataFrame or list)."""
    recs = rows.to_dict("records") if hasattr(rows, "to_dict") else list(rows)
    out: dict = {}
    for r in recs:
        out.setdefault((r["mode"], r["question_id"]), []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r["agent"]))
    return out


def plurality(answers) -> Counter:
    return Counter(int(a) for a in answers if int(a) >= 0)

