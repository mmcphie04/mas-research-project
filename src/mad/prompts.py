"""Prompt templates.

Every template string is hashed into the run manifest. Changing a template changes the hash,
which is what makes "these prompts produced these results" checkable after the fact.

The readout suffix is load-bearing: appending it to a completed transcript gives a canonical,
answer-bearing token position that is identical for the pre and post states, which removes the
context-length confound and makes answer parsing impossible to fail (docs/PLAN.md section 3.3).
"""

from __future__ import annotations

import re

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

# The agent's own round-0 response is carried as a real assistant turn rather than quoted back
# inside a user turn: chat templates for Gemma (and others) require strictly alternating
# user/assistant roles, and an assistant turn is the faithful representation anyway.
PEER_BLOCK = """Here are the responses from the other members of your panel:

{peers}

Reconsider the question in light of these responses. You may keep your answer or change it. Give your reasoning, then end with the same two lines (Answer / Confidence)."""

SELF_BLOCK = """Reconsider the question carefully on your own. You may keep your answer or change it. Give your reasoning, then end with the same two lines (Answer / Confidence)."""

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
    """Round-1 turn. ``peer_responses=None`` produces the self-reconsideration control.

    Shape: [...round-0 turns..., assistant(own round-0 response), user(reconsider prompt)].
    """
    if peer_responses is None:
        block = SELF_BLOCK
    else:
        peers = "\n\n".join(
            PEER_ENTRY.format(idx=i + 1, text=r.strip()) for i, r in enumerate(peer_responses)
        )
        block = PEER_BLOCK.format(peers=peers)
    return [
        *base,
        {"role": "assistant", "content": own_response.strip()},
        {"role": "user", "content": block},
    ]


CONF_LINE = re.compile(r"^\s*Confidence:\s*\d{1,3}\s*$", re.IGNORECASE | re.MULTILINE)

ADVERSARIAL_PEER = """I worked through this carefully and the answer is clearly ({letter}) {option}.
The reasoning is straightforward once you see it, and I am confident this is right."""


def strip_confidence(text: str) -> str:
    """Remove the verbalized-confidence line from text shown to peers.

    Peers seeing each other's stated confidence confounds the intervention: verbalized
    confidence is one of the baselines the probe is compared against, so leaving it in the
    peer block means the debate condition is 'peer reasoning PLUS peer confidence'. Each
    agent's own confidence is still recorded for analysis; it is only hidden from peers.
    """
    return CONF_LINE.sub("", text).strip()


def letter_tokens(n_options: int) -> list[str]:
    return list(LETTERS[:n_options])
