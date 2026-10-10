"""The one activation loader: explicit keys, no silent substitution.

Every stored state is identified by (condition, question, agent, phase, position). Earlier
loaders keyed on (condition, question, agent, phase) only, so once several token positions were
saved, later positions silently overwrote earlier ones in the lookup -- and a missing
``stated_answer`` state would have been filled by whichever position happened to be left.

``StateBank.matrix`` returns states for a requested list of keys at ONE position, and applies a
declared missingness policy: ``"error"`` (default: any missing state is a failure), or
``"mask"`` (return a boolean mask of which requested states exist; the caller must restrict its
analysis to those rows and report how many were dropped). It never falls back to another
position or phase.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mad.store import ActivationStore

KEY = ("condition", "question_id", "agent_id", "phase", "position")


class MissingStates(KeyError):
    pass


class StateBank:
    def __init__(self, run_dir: str | Path):
        self.store = ActivationStore(run_dir)
        ix = self.store.index.reset_index(drop=True)
        if "position" not in ix.columns:
            ix["position"] = "readout_last"
        dup = ix.duplicated(list(KEY))
        if dup.any():
            raise ValueError(f"{int(dup.sum())} duplicate state keys in {run_dir}")
        self._row = {tuple(k): i for i, k in enumerate(ix[list(KEY)].itertuples(index=False))}
        self.index = ix
        self.n_layers = self.store.n_layers
        self.d_model = self.store.d_model
        self._cache: dict[int, np.ndarray] = {}

    @property
    def positions(self) -> list[str]:
        return sorted(self.index.position.unique())

    def availability(self) -> pd.DataFrame:
        """How many states exist for each (condition, phase, position)."""
        return self.index.groupby(["condition", "phase", "position"]).size().rename("states").reset_index()

    def rows(self, keys: pd.DataFrame, phase: str, position: str, missing: str = "error"):
        """Row numbers (into ``load_layer``) for each key at one phase and position.

        ``keys`` needs columns condition, question_id, agent_id. Returns (rows, present_mask);
        rows are -1 where absent. With ``missing="error"`` any absence raises MissingStates.
        """
        if missing not in ("error", "mask"):
            raise ValueError("missing must be 'error' or 'mask'")
        rows = np.array([self._row.get((c, q, int(a), phase, position), -1)
                         for c, q, a in zip(keys.condition, keys.question_id, keys.agent_id, strict=True)])
        present = rows >= 0
        if missing == "error" and not present.all():
            k = keys.iloc[int(np.argmin(present))]
            raise MissingStates(f"{int((~present).sum())} of {len(rows)} requested states absent at "
                                f"phase={phase!r} position={position!r}, e.g. {tuple(k)}")
        return rows, present

    def layer(self, layer: int) -> np.ndarray:
        if layer not in self._cache:
            self._cache = {layer: self.store.load_layer(layer)}  # keep one layer in memory
        return self._cache[layer]

    def matrix(self, keys: pd.DataFrame, phase: str, position: str, layer: int,
               missing: str = "error") -> tuple[np.ndarray, np.ndarray]:
        """(states, present_mask): states for the present keys only, in key order."""
        rows, present = self.rows(keys, phase, position, missing)
        return self.layer(layer)[rows[present]], present
