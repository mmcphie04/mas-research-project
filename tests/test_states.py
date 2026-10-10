"""The activation loader must return exactly the requested phase and position, never another."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mad.analysis.states import MissingStates, StateBank
from mad.store import ActivationWriter

PHASES = ("pre", "peer_exposure", "post")
POSITIONS = ("readout_last", "generation_last", "stated_answer", "before_stated_answer")


def _code(q, a, phase, pos):
    """A value that identifies one state uniquely (small enough to be exact in float16)."""
    return ((q * 3 + a) * 3 + PHASES.index(phase)) * 4 + POSITIONS.index(pos)


@pytest.fixture
def bank(tmp_path):
    L, d = 3, 4
    w = ActivationWriter(tmp_path, L, d, shard_size=7)  # small shards: keys span several files
    for q in range(5):
        for a in range(3):
            for phase in PHASES:
                for pos in POSITIONS:
                    if pos in ("stated_answer", "before_stated_answer") and (phase == "peer_exposure" or q == 2):
                        continue  # no stated answer: exposure has no text; question 2 failed to parse
                    w.add({"run_id": "r", "condition": "debate", "question_id": f"q{q}", "agent_id": a,
                           "phase": phase, "position": pos},
                          np.full((L + 1, d), _code(q, a, phase, pos), dtype=np.float16))
    w.close()
    return StateBank(tmp_path)


def _keys(qs=range(5)):
    return pd.DataFrame([{"condition": "debate", "question_id": f"q{q}", "agent_id": a}
                         for q in qs for a in range(3)])


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("pos", ["readout_last", "generation_last"])
def test_every_phase_and_position_is_returned_exactly(bank, phase, pos):
    keys = _keys()
    X, present = bank.matrix(keys, phase, pos, layer=2)
    expect = [_code(int(q[1:]), a, phase, pos) for q, a in zip(keys.question_id, keys.agent_id, strict=True)]
    assert present.all() and np.array_equal(X[:, 0], np.array(expect, dtype=np.float32))


def test_missing_position_is_an_error_by_default(bank):
    with pytest.raises(MissingStates, match="stated_answer"):
        bank.matrix(_keys(), "post", "stated_answer", layer=1)


def test_mask_policy_drops_only_the_missing_rows_and_never_substitutes(bank):
    keys = _keys()
    X, present = bank.matrix(keys, "post", "stated_answer", layer=1, missing="mask")
    assert (~present).sum() == 3 and not present[keys.question_id == "q2"].any()
    got = X[:, 0] % 4  # the position part of the code
    assert (got == POSITIONS.index("stated_answer")).all(), "another position was substituted"


def test_duplicate_keys_are_refused(tmp_path):
    w = ActivationWriter(tmp_path, 1, 2)
    k = {"run_id": "r", "condition": "c", "question_id": "q", "agent_id": 0, "phase": "pre",
         "position": "readout_last"}
    w.add(k, np.zeros((2, 2)))
    w.add(k, np.ones((2, 2)))
    w.close()
    with pytest.raises(ValueError, match="duplicate"):
        StateBank(tmp_path)
