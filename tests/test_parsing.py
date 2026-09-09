from mad.parsing import parse_answer, parse_confidence

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


def test_parse_failure_is_flagged_not_guessed():
    assert parse_answer("I am not sure.", 10) == (-1, False)
    assert parse_answer("", 10) == (-1, False)


def test_letter_beyond_option_count_is_a_failure():
    assert parse_answer("Answer: (J)", 4) == (-1, False)
    assert parse_answer("Answer: (C)", 4) == (2, True)


def test_format_tolerance():
    assert parse_answer("answer: b", 10) == (1, True)
    assert parse_answer("Answer:(D)", 10) == (3, True)
