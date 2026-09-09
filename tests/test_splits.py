import pandas as pd
import pytest

from mad.data import assert_no_leakage, make_splits, stratified_sample, synthetic_questions


def test_every_question_lands_in_exactly_one_split():
    qs = synthetic_questions(300, seed=0)
    assignment = make_splits(qs, (0.5, 0.2, 0.3), seed=1)
    assert set(assignment) == {q.question_id for q in qs}
    assert set(assignment.values()) <= {"train", "val", "test"}


def test_split_fractions_are_approximately_respected():
    qs = synthetic_questions(1000, seed=0)
    counts = pd.Series(make_splits(qs, (0.5, 0.2, 0.3), seed=1)).value_counts(normalize=True)
    assert abs(counts["train"] - 0.5) < 0.03
    assert abs(counts["val"] - 0.2) < 0.03
    assert abs(counts["test"] - 0.3) < 0.03


def test_stratified_by_subject():
    qs = synthetic_questions(500, seed=0)
    assignment = make_splits(qs, (0.5, 0.2, 0.3), seed=1)
    frame = pd.DataFrame([{"subject": q.subject, "split": assignment[q.question_id]} for q in qs])
    share = frame.groupby("subject")["split"].apply(lambda s: (s == "test").mean())
    assert share.max() - share.min() < 0.12


def _rows_for(qs):
    return pd.DataFrame(
        [
            {"question_id": q.question_id, "agent_id": a, "phase": p, "condition": c}
            for q in qs
            for a in range(3)
            for p in ("pre", "post")
            for c in ("debate", "self_reconsider")
        ]
    )


def test_derived_rows_inherit_their_question_split():
    qs = synthetic_questions(60, seed=0)
    assignment = make_splits(qs, (0.5, 0.2, 0.3), seed=1)
    rows = _rows_for(qs)
    assert_no_leakage(rows, assignment)  # 12 rows per question, all one split
    assert len(rows) == 12 * len(qs)


def test_rows_from_another_question_set_are_rejected():
    qs = synthetic_questions(20, seed=0)
    assignment = make_splits(qs, (0.5, 0.2, 0.3), seed=1)
    # synthetic ids are positional, so take a disjoint slice of a larger pool
    rows = _rows_for(synthetic_questions(60, seed=0)[40:])
    with pytest.raises(ValueError, match="absent from the split map"):
        assert_no_leakage(rows, assignment)


def test_a_per_row_split_is_caught():
    # The failure that actually bites: splitting rows instead of questions, so the same
    # question's three agent rows straddle the train/test boundary.
    qs = synthetic_questions(20, seed=0)
    rows = _rows_for(qs)
    rows["split"] = ["train" if i % 2 else "test" for i in range(len(rows))]
    with pytest.raises(AssertionError, match="span multiple splits"):
        assert_no_leakage(rows)


def test_grouped_split_passes_the_same_check():
    qs = synthetic_questions(20, seed=0)
    assignment = make_splits(qs, (0.5, 0.2, 0.3), seed=1)
    rows = _rows_for(qs)
    rows["split"] = rows["question_id"].map(assignment)
    assert_no_leakage(rows)


def test_stratified_sample_balances_subjects():
    qs = synthetic_questions(500, seed=0)
    sample = stratified_sample(qs, 100, seed=3)
    counts = pd.Series([q.subject for q in sample]).value_counts()
    assert len(sample) == 100
    assert counts.max() - counts.min() <= 1
