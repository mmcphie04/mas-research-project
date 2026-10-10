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

# Brevity is part of the protocol, not a cost tweak: on the 1B model a 192-token budget cut off
# 73% of first answers, and a truncated response has no stated answer. Raising the budget
# multiplies cost; asking for concise reasoning fixes the cause.
BREVITY = "Keep your reasoning concise: at most about 150 words."

SYSTEM = """{persona} You are answering a multiple-choice question that may require several steps of reasoning.

Think through the problem. {brevity} Then end your response with exactly two lines:
Answer: (X)
Confidence: NN

where X is the letter of your chosen option and NN is an integer from 0 to 100 giving your confidence that the answer is correct."""

SYSTEM_NO_CONF = """{persona} You are answering a multiple-choice question that may require several steps of reasoning.

Think through the problem. {brevity} Then end your response with exactly one line:
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

Reconsider the question in light of these responses. You may keep your answer or change it. {brevity} {ending}"""

SELF_BLOCK = """Reconsider the question carefully on your own. You may keep your answer or change it. {brevity} {ending}"""

# Round 1 must ask for exactly the lines round 0 asked for. Previously round 1 always asked for
# "the same two lines (Answer / Confidence)", even in runs whose round 0 asked for one line.
ENDING_CONF = "Give your reasoning, then end with the same two lines as before: Answer: (X) and Confidence: NN."
ENDING_NO_CONF = "Give your reasoning, then end with the same line as before: Answer: (X)."

PEER_ENTRY = """--- Panel member {idx} ---
{text}"""

# Answer-only peers: the peer's stated letter and option text, no reasoning. Separates "a peer
# chose X" from "a peer argued for X".
PEER_ANSWER_ONLY = "Answer: ({letter}) {option}"
# What an answer-only peer shows when that peer's response had no valid stated answer. The
# readout fallback is never shown to peers as if the peer had written it.
PEER_NO_ANSWER = "No final answer was given."

MILD_PEER = """Looking at this again, I think it might be ({letter}) {option}. I am not certain,
but that reading seems defensible to me."""

ADVERSARIAL_PEER = """I worked through this carefully and the answer is clearly ({letter}) {option}.
The reasoning is straightforward once you see it, and I am confident this is right."""

READOUT_SUFFIX = "\nFinal answer: ("

# Candidate evaluation (mad.candidates): the model sees a question with one proposed answer,
# optionally with an unsupported endorsement, and is read out without generating.
CANDIDATE_SYSTEM = """{persona} You will see a multiple-choice question and a proposed answer to it. The proposed answer may be right or wrong. Decide which option is correct."""

CANDIDATE_BLOCK = """Proposed answer: ({letter}) {option}"""

ENDORSEMENT = 'A panel member comments: "I am confident the answer is ({letter}) {option}."'

# Every string that can reach a prompt. tests/test_prompts.py fails if a module-level template
# is missing here, because an unhashed template is an unrecorded protocol change.
TEMPLATES = {
    "brevity": BREVITY,
    "system": SYSTEM,
    "system_no_conf": SYSTEM_NO_CONF,
    "question": QUESTION,
    "peer_block": PEER_BLOCK,
    "self_block": SELF_BLOCK,
    "ending_conf": ENDING_CONF,
    "ending_no_conf": ENDING_NO_CONF,
    "peer_entry": PEER_ENTRY,
    "peer_answer_only": PEER_ANSWER_ONLY,
    "peer_no_answer": PEER_NO_ANSWER,
    "mild_peer": MILD_PEER,
    "adversarial_peer": ADVERSARIAL_PEER,
    "readout_suffix": READOUT_SUFFIX,
    "candidate_system": CANDIDATE_SYSTEM,
    "candidate_block": CANDIDATE_BLOCK,
    "endorsement": ENDORSEMENT,
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
    system = (SYSTEM if ask_confidence else SYSTEM_NO_CONF).format(persona=persona, brevity=BREVITY)
    user = QUESTION.format(
        subject=subject, question=question, options=format_options(options, perm)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def round1_messages(
    *,
    base: list[Message],
    own_response: str,
    peer_responses: list[str] | None,
    ask_confidence: bool = True,
) -> list[Message]:
    """Round-1 turn. ``peer_responses=None`` produces the self-reconsideration control.

    Shape: [...round-0 turns..., assistant(own round-0 response), user(reconsider prompt)].
    ``peer_responses`` are shown in the order given; the caller decides (and records) that order.
    """
    ending = ENDING_CONF if ask_confidence else ENDING_NO_CONF
    if peer_responses is None:
        block = SELF_BLOCK.format(brevity=BREVITY, ending=ending)
    else:
        peers = "\n\n".join(
            PEER_ENTRY.format(idx=i + 1, text=r.strip()) for i, r in enumerate(peer_responses)
        )
        block = PEER_BLOCK.format(peers=peers, brevity=BREVITY, ending=ending)
    return [
        *base,
        {"role": "assistant", "content": own_response.strip()},
        {"role": "user", "content": block},
    ]


CONF_LINE = re.compile(r"^\s*Confidence:\s*\d{1,3}\s*$", re.IGNORECASE | re.MULTILINE)


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
