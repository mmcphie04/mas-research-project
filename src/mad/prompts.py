"""Prompt templates.

Every template string is hashed into the run manifest. Changing a template changes the hash,
which is what makes "these prompts produced these results" checkable after the fact.

The readout suffix is load-bearing: appending it to a completed transcript gives a canonical,
answer-bearing token position that is identical for the pre and post states, which removes the
context-length confound and makes answer parsing impossible to fail (docs/PLAN.md section 3.3).
"""

from __future__ import annotations

from mad.permute import LETTERS, Permutation

Message = dict[str, str]

SYSTEM = """{persona} You are answering a multiple-choice question that may require several steps of reasoning.

Think through the problem, then end your response with exactly two lines:
Answer: (X)
Confidence: NN

where X is the letter of your chosen option and NN is an integer from 0 to 100 giving your confidence that the answer is correct."""

SYSTEM_NO_CONF = """{persona} You are answering a multiple-choice question that may require several steps of reasoning.

Think through the problem, then end your response with exactly one line:
Answer: (X)

where X is the letter of your chosen option."""

QUESTION = """Subject: {subject}

Question: {question}

Options:
{options}"""

PEER_BLOCK = """Here is your previous response:

{own}

Here are the responses from the other members of your panel:

{peers}

Reconsider the question in light of these responses. You may keep your answer or change it. Give your reasoning, then end with the same two lines (Answer / Confidence)."""

SELF_BLOCK = """Here is your previous response:

{own}

Reconsider the question carefully on your own. You may keep your answer or change it. Give your reasoning, then end with the same two lines (Answer / Confidence)."""

PEER_ENTRY = """--- Panel member {idx} ---
{text}"""

READOUT_SUFFIX = "\nFinal answer: ("

TEMPLATES = {
    "system": SYSTEM,
    "system_no_conf": SYSTEM_NO_CONF,
    "question": QUESTION,
    "peer_block": PEER_BLOCK,
    "self_block": SELF_BLOCK,
    "peer_entry": PEER_ENTRY,
    "readout_suffix": READOUT_SUFFIX,
}


def format_options(options: list[str], perm: Permutation) -> str:
    shown = perm.display_options(options)
    return "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(shown))


def round0_messages(
    *,
    persona: str,
    subject: str,
    question: str,
    options: list[str],
    perm: Permutation,
    ask_confidence: bool = True,
) -> list[Message]:
    system = (SYSTEM if ask_confidence else SYSTEM_NO_CONF).format(persona=persona)
    user = QUESTION.format(
        subject=subject, question=question, options=format_options(options, perm)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def round1_messages(
    *,
    base: list[Message],
    own_response: str,
    peer_responses: list[str] | None,
) -> list[Message]:
    """Round-1 turn. ``peer_responses=None`` produces the self-reconsideration control."""
    if peer_responses is None:
        block = SELF_BLOCK.format(own=own_response.strip())
    else:
        peers = "\n\n".join(
            PEER_ENTRY.format(idx=i + 1, text=r.strip()) for i, r in enumerate(peer_responses)
        )
        block = PEER_BLOCK.format(own=own_response.strip(), peers=peers)
    return [*base, {"role": "user", "content": block}]


def letter_tokens(n_options: int) -> list[str]:
    return list(LETTERS[:n_options])
