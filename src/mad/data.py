"""Benchmark loading, sampling, and the frozen grouped split.

The split key is the UNDERLYING QUESTION ID. A question appears once per agent, again under any
permutation replicate, and again in the self-reconsideration control; all of those rows must land
on the same side of the split or the probe has seen its own test questions
(docs/RISKS.md risk 5). ``assert_no_leakage`` enforces this rather than trusting convention.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Question:
    question_id: str
    subject: str
    question: str
    options: tuple[str, ...]
    gold_index: int  # index into ``options``

    @property
    def n_options(self) -> int:
        return len(self.options)


def _clean_options(options: list[str], gold_index: int) -> tuple[list[str], int]:
    """Drop MMLU-Pro's 'N/A' placeholder options and remap the gold index."""
    keep = [i for i, o in enumerate(options) if o is not None and str(o).strip().upper() != "N/A"]
    if gold_index not in keep:
        raise ValueError("gold option was filtered out as a placeholder")
    return [options[i] for i in keep], keep.index(gold_index)


def load_mmlu_pro(cfg) -> list[Question]:
    """Load MMLU-Pro via ``datasets``. Requires the ``data`` extra and network access."""
    from datasets import load_dataset

    ds = load_dataset(cfg.data.dataset, revision=cfg.data.dataset_revision, split=cfg.data.split)
    out: list[Question] = []
    for row in ds:
        gold = row.get("answer_index")
        if gold is None:
            gold = "ABCDEFGHIJ".index(str(row["answer"]).strip().upper())
        try:
            options, gold = _clean_options(list(row["options"]), int(gold))
        except ValueError:
            continue
        if len(options) < 2:
            continue
        out.append(
            Question(
                question_id=str(row["question_id"]),
                subject=str(row.get("category", "unknown")),
                question=str(row["question"]),
                options=tuple(options),
                gold_index=gold,
            )
        )
    return out


def synthetic_questions(
    n: int, subjects: list[str] | None = None, n_options: int = 10, seed: int = 0
) -> list[Question]:
    """Offline stand-in for MMLU-Pro so tier-0 and the test suite need no network."""
    subjects = subjects or ["math", "physics", "law", "biology", "history"]
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        subj = subjects[i % len(subjects)]
        k = int(rng.integers(4, n_options + 1))
        out.append(
            Question(
                question_id=f"syn-{i:06d}",
                subject=subj,
                question=f"[{subj}] Synthetic reasoning item #{i}.",
                options=tuple(f"option {j}" for j in range(k)),
                gold_index=int(rng.integers(0, k)),
            )
        )
    return out


def stratified_sample(
    questions: list[Question], n: int, seed: int, subjects: list[str] | None = None
) -> list[Question]:
    """Sample ``n`` questions with as-equal-as-possible subject representation."""
    pool = [q for q in questions if subjects is None or q.subject in subjects]
    if n >= len(pool):
        return sorted(pool, key=lambda q: q.question_id)
    by_subject: dict[str, list[Question]] = defaultdict(list)
    for q in pool:
        by_subject[q.subject].append(q)
    rng = np.random.default_rng(seed)
    for qs in by_subject.values():
        qs.sort(key=lambda q: q.question_id)
        rng.shuffle(qs)

    chosen: list[Question] = []
    subject_order = sorted(by_subject)
    cursor = {s: 0 for s in subject_order}
    while len(chosen) < n:
        progressed = False
        for s in subject_order:
            if len(chosen) >= n:
                break
            i = cursor[s]
            if i < len(by_subject[s]):
                chosen.append(by_subject[s][i])
                cursor[s] += 1
                progressed = True
        if not progressed:
            break
    return sorted(chosen, key=lambda q: q.question_id)


def make_splits(
    questions: list[Question],
    fracs: tuple[float, float, float] = (0.5, 0.2, 0.3),
    seed: int = 20260909,
) -> dict[str, str]:
    """Assign each question id to train/val/test, stratified by subject.

    Returns a ``{question_id: split}`` mapping. Because the key is the question id, every derived
    row (agent, phase, permutation replicate, condition) inherits the same split by construction.
    """
    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"split fractions must sum to 1, got {fracs}")
    rng = np.random.default_rng(seed)
    by_subject: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by_subject[q.subject].append(q.question_id)

    assignment: dict[str, str] = {}
    for subject in sorted(by_subject):
        ids = sorted(by_subject[subject])
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(fracs[0] * n))
        n_val = int(round(fracs[1] * n))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        for i, qid in enumerate(ids):
            assignment[qid] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assignment


def save_splits(assignment: dict[str, str], path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta or {}, "assignment": assignment}, indent=2, sort_keys=True)
    )
    return path


def load_splits(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())["assignment"]


def assert_no_leakage(
    rows: pd.DataFrame,
    assignment: dict[str, str] | None = None,
    id_col: str = "question_id",
    split_col: str = "split",
) -> None:
    """Fail loudly if any underlying question id appears in more than one split.

    Two modes. With ``assignment``, every row must have a question id present in the map (this
    catches rows built from a different question set). Without it, ``rows`` must already carry a
    split column -- which is the case that actually bites: a split drawn per ROW rather than per
    question puts the same question's three agent rows on both sides of the boundary.
    """
    if id_col not in rows.columns:
        raise KeyError(f"{id_col} not in rows")
    if assignment is not None:
        tagged = rows.assign(_split=rows[id_col].map(assignment))
        missing = tagged["_split"].isna()
        if missing.any():
            raise ValueError(
                f"{int(missing.sum())} rows have question ids absent from the split map"
            )
    else:
        if split_col not in rows.columns:
            raise KeyError(f"pass an assignment or give rows a {split_col!r} column")
        tagged = rows.rename(columns={split_col: "_split"})
    spread = tagged.groupby(id_col)["_split"].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise AssertionError(
            f"{len(bad)} question ids span multiple splits, e.g. {list(bad.index[:5])}"
        )


def questions_to_frame(questions: list[Question]) -> pd.DataFrame:
    return pd.DataFrame([asdict(q) for q in questions])
