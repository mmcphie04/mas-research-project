import pytest

from mad.parsing import parse_answer, parse_answer_field, parse_confidence

ROUND1 = """Here is your previous response:

Step 1: ...
Answer: (C)
Confidence: 70

Here are the responses from the other members of your panel:
Answer: (A)

Reconsidering, I now prefer a different option.
Answer: (F)
Confidence: 55"""


def test_takes_last_answer_not_the_quoted_one():
    # A first-match parser reads (C) here and reports that nothing changed -- a silent zero
    # revision rate rather than a crash.
    assert parse_answer(ROUND1, 10) == (5, True)
    assert parse_confidence(ROUND1) == 55.0


@pytest.mark.parametrize("text", [
    "Answer: cannot determine.",       # was read as C
    "Answer: Because the force doubles, B.",  # was read as B
    "Answer: Analysis of each option follows",  # was read as A
    "Answer: A careful look shows the answer is D",
    "Answer: b",                         # lowercase is never a letter
    "The required format is Answer: (B) on its own line.",  # quoted mid-sentence
    "I am not sure.",
    "",
    "Answer:",
    "Answer: (AB)",
])
def test_prose_and_malformed_fields_are_not_answers(text):
    p = parse_answer_field(text, 10)
    assert p.index == -1 and not p.ok, f"{text!r} parsed as {p}"


@pytest.mark.parametrize("text,idx,status", [
    ("Answer: (D)", 3, "strict"),
    ("Answer:(D)", 3, "strict"),
    ("**Answer: (D)**", 3, "strict"),
    ("**Answer:** (D)", 3, "strict"),
    ("- Answer: (D).", 3, "strict"),
    ("Answer: D", 3, "permissive"),
    ("answer: D", 3, "permissive"),
    ("Answer: (D) 42 newtons", 3, "permissive"),
    ("Answer: D) 42 newtons", 3, "permissive"),
])
def test_documented_and_unambiguous_forms(text, idx, status):
    p = parse_answer_field("Reasoning first.\n" + text + "\nConfidence: 80", 10)
    assert (p.index, p.status) == (idx, status)


def test_out_of_range_last_field_is_invalid_and_earlier_fields_are_not_used():
    p = parse_answer_field("Answer: (B)\nOn reflection:\nAnswer: (J)", 4)
    assert p.index == -1 and p.status == "out_of_range" and p.n_fields == 2
    assert parse_answer("Answer: (C)", 4) == (2, True)


def test_letter_offset_points_at_the_letter():
    text = "Some reasoning.\n**Answer: (G)**\nConfidence: 10"
    p = parse_answer_field(text, 10)
    assert text[p.letter_start] == "G"


def test_multiple_fields_last_valid_wins():
    p = parse_answer_field("Answer: (A)\nwait\nAnswer: B\n", 10)
    assert (p.index, p.status, p.n_fields) == (1, "permissive", 2)


def test_confidence_is_line_anchored():
    assert parse_confidence("My confidence: 90 is not a field\nConfidence: 40") == 40.0
    assert parse_confidence("no confidence line") != parse_confidence("no confidence line")  # nan
