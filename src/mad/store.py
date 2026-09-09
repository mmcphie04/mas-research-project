"""Activation storage.

Layout is LAYER-MAJOR -- each shard is (n_layers + 1, n_states, d_model) -- so reading one layer
across every state is a contiguous read. That is what keeps a layer sweep affordable on a 13 GB
machine: one layer of 18,000 states at d=3840 is ~138 MB, while the full tensor is ~6.8 GB.

The rule that keeps this from exploding is FIXED TOKEN POSITIONS ONLY (docs/RISKS.md risk 16):
all layers at a handful of canonical positions is fine, one layer at all token positions is 1.8 TB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FILE = "activation_index.parquet"
SHARD_FMT = "acts_{shard:04d}.npy"

KEY_COLUMNS = ["run_id", "condition", "question_id", "agent_id", "phase", "position"]


def storage_bytes(
    n_states: int, n_layers: int, d_model: int, n_positions: int = 1, itemsize: int = 2
) -> int:
    """Exact activation footprint. Use this before committing to a collection run."""
    return n_states * n_positions * (n_layers + 1) * d_model * itemsize


class ActivationWriter:
    def __init__(self, run_dir: str | Path, n_layers: int, d_model: int, shard_size: int = 500):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n_layers = n_layers
        self.d_model = d_model
        self.shard_size = shard_size
        self._buf: list[np.ndarray] = []
        self._keys: list[dict] = []
        self._rows: list[dict] = []
        self._shard = 0
        self.shard_paths: list[Path] = []

    def add(self, key: dict, hidden: np.ndarray) -> None:
        expected = (self.n_layers + 1, self.d_model)
        if hidden.shape != expected:
            raise ValueError(f"hidden shape {hidden.shape} != expected {expected}")
        self._buf.append(hidden.astype(np.float16))
        self._keys.append(dict(key))
        if len(self._buf) >= self.shard_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        block = np.stack(self._buf, axis=1)  # (L+1, n, d) -- layer-major
        path = self.run_dir / SHARD_FMT.format(shard=self._shard)
        np.save(path, block)
        self.shard_paths.append(path)
        for offset, key in enumerate(self._keys):
            self._rows.append({**key, "shard": self._shard, "offset": offset})
        self._shard += 1
        self._buf.clear()
        self._keys.clear()

    def close(self) -> pd.DataFrame:
        self._flush()
        index = pd.DataFrame(self._rows)
        index.attrs["n_layers"] = self.n_layers
        index.attrs["d_model"] = self.d_model
        meta = pd.DataFrame(
            [{"n_layers": self.n_layers, "d_model": self.d_model, "n_states": len(index)}]
        )
        index.to_parquet(self.run_dir / INDEX_FILE, index=False)
        meta.to_parquet(self.run_dir / "activation_meta.parquet", index=False)
        return index


class ActivationStore:
    """Read-side view. Loads one layer at a time; never materialises the full tensor."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.index = pd.read_parquet(self.run_dir / INDEX_FILE)
        meta = pd.read_parquet(self.run_dir / "activation_meta.parquet").iloc[0]
        self.n_layers = int(meta["n_layers"])
        self.d_model = int(meta["d_model"])
        self.index = self.index.sort_values(["shard", "offset"]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.index)

    def load_layer(self, layer: int, rows: np.ndarray | pd.Series | None = None) -> np.ndarray:
        """Return (n, d_model) float32 for one layer, in index order.

        ``rows`` is an optional boolean mask or positional index over ``self.index``.
        """
        if not 0 <= layer <= self.n_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.n_layers}")
        blocks = []
        for shard in sorted(self.index["shard"].unique()):
            path = self.run_dir / SHARD_FMT.format(shard=int(shard))
            arr = np.load(path, mmap_mode="r")
            blocks.append(np.asarray(arr[layer], dtype=np.float32))
        full = np.concatenate(blocks, axis=0)
        if rows is None:
            return full
        rows = np.asarray(rows)
        return full[rows]

    def layer_range(self) -> range:
        return range(self.n_layers + 1)
