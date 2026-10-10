"""Activation storage.

Values are bfloat16 (see ``to_bf16_bits`` for why not float16), 2 bytes each.

Layout is LAYER-MAJOR -- each shard is (n_layers + 1, n_states, d_model) -- so reading one layer
across every state is a contiguous read. That is what keeps a layer sweep affordable on a 13 GB
machine: one layer of 18,000 states at d=3840 is ~138 MB, while the full tensor is ~6.8 GB.

The rule that keeps this from exploding is FIXED TOKEN POSITIONS ONLY (docs/RISKS.md risk 16):
all layers at a handful of canonical positions is fine, one layer at all token positions is 1.8 TB.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FILE = "activation_index.parquet"
SHARD_FMT = "acts_{shard:04d}.npy"

# Storage format: bfloat16, kept as its raw 16-bit pattern in uint16 arrays (numpy has no native
# bfloat16). Not float16: Gemma's residual stream has a few "massive activations" -- on
# gemma-3-12b one dimension reaches 163,840 at the readout position in layers 40-47 (and over
# 800,000 at other positions) -- far beyond float16's 65,504, so a float16 cast turned them into
# infinity. bfloat16 has float32's range, the same 2 bytes, and is the model's own compute format
# on the GPU, so storing it is lossless there. Readers decode to float32; old float16 shards
# still load.
BF16_NAN = 0x7FC0


def to_bf16_bits(x) -> np.ndarray:
    """float32 values -> bfloat16 bit patterns (uint16), round-to-nearest-even; NaN kept."""
    f = np.ascontiguousarray(x, dtype=np.float32)
    u = f.view(np.uint32)
    out = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
    out[np.isnan(f)] = BF16_NAN
    return out


def from_bf16_bits(b) -> np.ndarray:
    """bfloat16 bit patterns (uint16) -> float32."""
    return (np.asarray(b, dtype=np.uint32) << np.uint32(16)).view(np.float32)


def decode(arr) -> np.ndarray:
    """Stored activations of either format -> float32."""
    a = np.asarray(arr)
    return from_bf16_bits(a) if a.dtype == np.uint16 else a.astype(np.float32)

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
        self._buf.append(to_bf16_bits(hidden))
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
            [{"n_layers": self.n_layers, "d_model": self.d_model, "n_states": len(index),
              "storage_dtype": "bfloat16"}]
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

        Only the offsets the index lists are read from each shard, so a shard may hold rows this
        index does not use (a saved round-0 file with a missing position, for instance). Two
        on-disk layouts: ``layer_major`` (L+1, n, d) -- this writer's shards -- and ``row_major``
        (n, L+1, d) -- the round-0 cache, referenced by condition directories instead of copied.
        ``rows`` is an optional boolean mask or positional index over ``self.index``.
        """
        if not 0 <= layer <= self.n_layers:
            raise IndexError(f"layer {layer} out of range 0..{self.n_layers}")
        blocks = []
        paths = self._shard_paths()
        layouts = (dict(zip(self.index["shard"], self.index["layout"], strict=True))
                   if "layout" in self.index.columns else {})
        for shard, grp in self.index.groupby("shard", sort=True):
            arr = np.load(paths[int(shard)], mmap_mode="r")
            off = grp["offset"].to_numpy()
            if layouts.get(shard, "layer_major") == "row_major":
                blocks.append(decode(arr[off, layer]))
            else:
                blocks.append(decode(arr[layer][off]))
        full = np.concatenate(blocks, axis=0)
        if rows is None:
            return full
        rows = np.asarray(rows)
        return full[rows]

    def _shard_paths(self) -> dict[int, Path]:
        """Shard number -> file. An index may record each shard's path relative to its own
        directory (a merged view, or a condition referencing the shared round-0 cache) instead
        of the default file name in the same directory."""
        if "shard_path" in self.index.columns:
            pairs = self.index[["shard", "shard_path"]].drop_duplicates()
            return {int(s): (self.run_dir / p).resolve() for s, p in zip(pairs.shard, pairs.shard_path,
                                                                          strict=True)}
        return {int(s): self.run_dir / SHARD_FMT.format(shard=int(s)) for s in self.index["shard"].unique()}

    def layer_range(self) -> range:
        return range(self.n_layers + 1)


def merge_runs(run_dirs: list[str | Path], out_dir: str | Path, rows_file: str) -> pd.DataFrame:
    """Concatenate several chunk runs into one logical run directory.

    Chunked collection is what makes an overnight job survivable: each chunk writes its own
    completion record, so a crash costs one chunk rather than the night. Shard files are copied
    and renumbered rather than re-read, so merging is IO-bound and quick.
    """
    import shutil

    run_dirs = [Path(d) for d in run_dirs]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = [pd.read_parquet(d / "activation_meta.parquet").iloc[0] for d in run_dirs]
    n_layers = {int(m["n_layers"]) for m in metas}
    d_model = {int(m["d_model"]) for m in metas}
    if len(n_layers) != 1 or len(d_model) != 1:
        raise ValueError(f"cannot merge runs with differing shapes: layers={n_layers} d={d_model}")

    indices, frames, next_shard = [], [], 0
    for d in run_dirs:
        idx = pd.read_parquet(d / INDEX_FILE).sort_values(["shard", "offset"])
        remap = {}
        for old in sorted(idx["shard"].unique()):
            shutil.copy2(d / SHARD_FMT.format(shard=int(old)), out_dir / SHARD_FMT.format(shard=next_shard))
            remap[int(old)] = next_shard
            next_shard += 1
        idx["shard"] = idx["shard"].map(remap)
        indices.append(idx)
        frames.append(pd.read_parquet(d / rows_file))

    index = pd.concat(indices, ignore_index=True)
    rows = pd.concat(frames, ignore_index=True)
    index.to_parquet(out_dir / INDEX_FILE, index=False)
    pd.DataFrame(
        [{"n_layers": n_layers.pop(), "d_model": d_model.pop(), "n_states": len(index)}]
    ).to_parquet(out_dir / "activation_meta.parquet", index=False)
    rows.to_parquet(out_dir / rows_file, index=False)
    shutil.copy2(run_dirs[0] / "manifest.json", out_dir / "manifest.json")
    tr = [pd.read_json(d / "transcripts.jsonl", lines=True, dtype={"question_id": str})
          for d in run_dirs if (d / "transcripts.jsonl").exists()]
    if tr:  # transcripts are what make a readout re-run possible without regenerating
        pd.concat(tr, ignore_index=True).to_json(
            out_dir / "transcripts.jsonl", orient="records", lines=True
        )
    (out_dir / "completion.json").write_text(
        json.dumps(
            {
                "timestamp_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "row_counts": {"states": len(rows), "activations": len(index)},
                "merged_from": [str(d) for d in run_dirs],
                "shard_sha256": {},
                "errors": [],
            },
            indent=2,
        )
    )
    return rows
