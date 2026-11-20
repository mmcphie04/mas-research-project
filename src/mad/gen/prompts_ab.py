"""Prompt templates for Experiments A (chooser) and B (debate arms). Hashed into TEMPLATE_HASHES,
recorded with every job; a change is a protocol change."""

from __future__ import annotations

import hashlib

from mad import prompts as P

# ---------------------------------------------------------------- Experiment A: GenSelect-style chooser
CHOOSER_SYSTEM = "You are an expert grader of solutions to multiple-choice questions."

CHOOSER_USER = """{question_block}

Below are {m} candidate solutions to this question, in no particular order. Each ends with its final answer in the form "Answer: (X)". At most one of the final answers is correct.

{candidates}

Compare the candidates' reasoning, check the decisive steps yourself, and decide which candidate's final answer is most likely correct. End your response with exactly one line:
Best candidate: N
where N is the number of the candidate (1 to {m})."""

CANDIDATE_ENTRY = """=== Candidate {i} ===
{text}"""

# ---------------------------------------------------------------- thinking mode: optional visible rationale
# Qwen3 in thinking mode writes ALL its reasoning inside the thinking block and shows only the two
# final lines (judge run: 44 of 44 visible answers were 12 tokens). Peers and choosers see only the
# visible text, so without this note thinking-mode agents are shown bare letters. OFF unless the
# run sets --visible-rationale (a protocol choice recorded in every part's info).
VISIBLE_RATIONALE = ("When you have finished thinking, write out your reasoning for the reader (at most about 250 words), "
                     "then end with the two lines.")

# ---------------------------------------------------------------- Experiment B: debate arms
PEER_ENTRY_NAMED = """--- Panel member {num} ---
{text}"""

# the anonymized arm: the standard peer block, peers labelled only by shown position (A, B) and
# shown in a seeded random order -- it differs from "debate" in labels and order, nothing else
PEER_ENTRY_ANON = """--- Response {label} ---
{text}"""

DISSENT = ("State your honest answer even if it contradicts the others; change only if you can identify a "
           "specific error.")

DISSENT_BLOCK = """Here are the responses from the other members of your panel:

{peers}

Reconsider the question in light of these responses. You may keep your answer or change it. """ + DISSENT + """ {brevity} {ending}"""

TEMPLATES = {"CHOOSER_SYSTEM": CHOOSER_SYSTEM, "CHOOSER_USER": CHOOSER_USER, "CANDIDATE_ENTRY": CANDIDATE_ENTRY,
             "PEER_ENTRY_NAMED": PEER_ENTRY_NAMED, "PEER_ENTRY_ANON": PEER_ENTRY_ANON, "DISSENT_BLOCK": DISSENT_BLOCK,
             "PEER_BLOCK": P.PEER_BLOCK, "SELF_BLOCK": P.SELF_BLOCK,
             "SYSTEM": P.SYSTEM, "QUESTION": P.QUESTION, "BREVITY_250": P.BREVITY_250, "ENDING_CONF": P.ENDING_CONF,
             "READOUT_SUFFIX": P.READOUT_SUFFIX, "VISIBLE_RATIONALE": VISIBLE_RATIONALE}
TEMPLATE_HASHES = {k: hashlib.sha256(v.encode()).hexdigest()[:16] for k, v in TEMPLATES.items()}
