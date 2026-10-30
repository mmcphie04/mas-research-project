"""Cross-model judge of revisions (revision gate, section 5; exploratory).

The judge sees the question with its options (exactly the block the agent saw, so the option
letters match both responses) and two responses labelled only "Response 1" and "Response 2" --
the agent's original and its revision, confidence lines removed. It is asked which response's
final answer is correct; the readout is the judge's probability on "1" against "2" after the
assistant prefill ``JUDGE_PREFILL`` (fp32 head, one sequence per pass). Both orders are run and
averaged: P(revision correct) = (P("2" | original first) + P("1" | revision first)) / 2.

The own-answer baseline asks the judge model the question itself under the main collection's
protocol (250-word guidance, 1,280-token cap, the same system prompt and persona as Gemma).

Templates are hashed into ``TEMPLATE_HASHES`` and recorded with every output.
"""

from __future__ import annotations

import hashlib

from mad import prompts as P

JUDGE_SYSTEM = "You are a careful, impartial grader of answers to multiple-choice questions."

JUDGE_USER = """{question_block}

Two responses to this question follow. Each ends with its final answer in the form "Answer: (X)".

=== Response 1 ===
{response_1}

=== Response 2 ===
{response_2}

Which response's final answer is correct? If neither seems fully right, choose the one more likely to be correct. Reply with exactly one line:
Better response: (1)
or
Better response: (2)"""

JUDGE_PREFILL = "Better response: ("
CHOICES = ("1", "2")
THINK_CLOSE = "\n</think>\n\n"  # appended when a thinking trace hits its cap without closing
OWN_PERSONA = "You are a careful, methodical reasoner."
OWN_BREVITY_WORDS = 250

TEMPLATE_HASHES = {name: hashlib.sha256(t.encode()).hexdigest()[:16] for name, t in {
    "JUDGE_SYSTEM": JUDGE_SYSTEM, "JUDGE_USER": JUDGE_USER, "JUDGE_PREFILL": JUDGE_PREFILL,
    "THINK_CLOSE": THINK_CLOSE, "OWN_SYSTEM": P.SYSTEM.format(persona=OWN_PERSONA, brevity=P.brevity_text(OWN_BREVITY_WORDS)),
}.items()}


def judge_messages(question_block: str, response_1: str, response_2: str) -> list[dict]:
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(question_block=question_block, response_1=response_1,
                                                          response_2=response_2)}]


def both_orders(event: dict) -> dict[str, list[dict]]:
    """'original_first': Response 1 = original; 'revised_first': Response 1 = revision."""
    q, o, r = event["question_block"], event["original"], event["revised"]
    return {"original_first": judge_messages(q, o, r), "revised_first": judge_messages(q, r, o)}


def p_revision(p2_original_first: float, p1_revised_first: float) -> float:
    """Order-averaged probability that the revision's final answer is the correct one."""
    return (p2_original_first + p1_revised_first) / 2


def position_bias(p1_original_first: float, p1_revised_first: float) -> float:
    """Mean probability on "Response 1" across the two orders, minus 1/2 (0 = no position bias)."""
    return (p1_original_first + p1_revised_first) / 2 - 0.5


def own_messages(question_block: str) -> list[dict]:
    system = P.SYSTEM.format(persona=OWN_PERSONA, brevity=P.brevity_text(OWN_BREVITY_WORDS))
    return [{"role": "system", "content": system}, {"role": "user", "content": question_block}]


def after_thinking(text: str) -> str:
    """The visible answer of a thinking-mode response: the text after the last </think>."""
    return text.rsplit("</think>", 1)[-1].strip() if "</think>" in text else text.strip()
