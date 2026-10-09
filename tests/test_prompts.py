"""Chat-structure invariants.

Gemma's chat template forbids a system role AND requires strictly alternating user/assistant
turns. Quoting the agent's own round-0 response inside a second user turn violates both and
fails only at render time, on a real tokenizer -- i.e. after the model has been downloaded.
These checks are pure and run in milliseconds.
"""

from mad.backends.hf import HFBackend
from mad.permute import identity_permutation
from mad.prompts import round0_messages, round1_messages


def _base():
    return round0_messages(
        persona="You are careful.",
        subject="physics",
        question="What is the acceleration?",
        options=["1", "2", "3", "4"],
        perm=identity_permutation(4),
    )


def _roles(messages):
    return [m["role"] for m in messages]


def test_round0_is_system_then_user():
    assert _roles(_base()) == ["system", "user"]


def test_round1_carries_the_agent_response_as_an_assistant_turn():
    msgs = round1_messages(base=_base(), own_response="Answer: (B)", peer_responses=["Answer: (C)"])
    assert _roles(msgs) == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "Answer: (B)"


def test_self_reconsideration_has_the_same_shape_and_no_peer_content():
    msgs = round1_messages(base=_base(), own_response="Answer: (B)", peer_responses=None)
    assert _roles(msgs) == ["system", "user", "assistant", "user"]
    assert "panel" not in msgs[3]["content"].lower()


def test_peer_block_contains_every_peer_response():
    msgs = round1_messages(
        base=_base(), own_response="Answer: (B)", peer_responses=["first peer", "second peer"]
    )
    assert "first peer" in msgs[3]["content"] and "second peer" in msgs[3]["content"]


def test_roles_alternate_after_the_system_turn_is_merged_away():
    # what a Gemma-style template actually sees
    for peers in ([], ["Answer: (C)"], None):
        msgs = round1_messages(
            base=_base(), own_response="Answer: (B)", peer_responses=peers if peers != [] else []
        )
        merged = HFBackend._merge_system(msgs)
        roles = _roles(merged)
        assert roles[0] == "user", roles
        assert all(a != b for a, b in zip(roles, roles[1:], strict=False)), roles


def test_merge_system_folds_content_into_the_first_user_turn():
    merged = HFBackend._merge_system(_base())
    assert _roles(merged) == ["user"]
    assert "You are careful." in merged[0]["content"]
    assert "acceleration" in merged[0]["content"]


def test_every_module_level_template_is_hashed():
    """An unhashed template is a protocol change the manifest cannot see."""
    import mad.prompts as P

    strings = {k: v for k, v in vars(P).items() if k.isupper() and isinstance(v, str)}
    hashed = set(P.TEMPLATES.values())
    missing = [k for k, v in strings.items() if v not in hashed and k != "LETTERS"]
    assert not missing, f"templates missing from TEMPLATES: {missing}"


def test_round1_asks_for_the_same_lines_as_round0():
    for ask in (True, False):
        r0 = round0_messages(persona="p", subject="s", question="q", options=["1", "2"],
                             perm=identity_permutation(2), ask_confidence=ask)
        for peers in (None, ["peer"]):
            r1 = round1_messages(base=r0, own_response="x", peer_responses=peers, ask_confidence=ask)
            assert ("Confidence" in r1[-1]["content"]) == ask
            assert ("Confidence" in r0[0]["content"]) == ask
            assert "150 words" in r0[0]["content"] and "150 words" in r1[-1]["content"]
