from __future__ import annotations

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

EMBEDDING = "encoder.embeddings.tok_embeddings.weight"


def map_tensor(path: Path, name: str) -> np.memmap:
    """Map one FP32 tensor's bytes read-only, following the safetensors layout
    (little-endian u64 header length, JSON header, data offsets relative to the data section)."""
    with path.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    info = header[name]
    start, end = info["data_offsets"]
    shape = tuple(info["shape"])
    if not all(type(value) is int and value >= 0 for value in (start, end, *shape)):
        raise ValueError(f"{name} has invalid offsets or shape")
    if info["dtype"] != "F32" or end - start != 4 * int(np.prod(shape)):
        raise ValueError(f"{name} must be a contiguous FP32 tensor to be memory-mapped")
    if 8 + header_length + end > path.stat().st_size:
        raise ValueError(f"{name} extends past the end of {path}")
    return np.memmap(path, dtype=np.float32, mode="r", offset=8 + header_length + start, shape=shape)


def load_tensors(path: Path, skip: frozenset[str] = frozenset()) -> dict[str, mx.array]:
    with safe_open(str(path), framework="np") as stream:
        return {name: mx.array(stream.get_tensor(name)) for name in stream.keys() if name not in skip}  # noqa: SIM118
