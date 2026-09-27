from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal, cast

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from .engine import JuliaEngine
from .model import JuliaDecisionModel, mlx_name, torch_name
from .modernbert import EncoderConfig, MappedEmbedding
from .tokenizer import JuliaTokenizer
from .weights import EMBEDDING, load_tensors, map_tensor

DTYPES = {"float32": mx.float32, "float16": mx.float16}


def auto_cache_limit() -> int:
    working_set = mx.device_info().get("max_recommended_working_set_size")
    return 1 << 30 if working_set is None else min(4 << 30, int(working_set) // 8)


CHECKPOINT_FILES = ["julia_config.json", "encoder/config.json", "model.safetensors", "tokenizer/*"]


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    root = Path(checkpoint)
    return root if root.is_dir() else Path(snapshot_download(str(checkpoint), allow_patterns=CHECKPOINT_FILES))


def load_model(checkpoint: str | Path, *, max_length: int | None = None, head_length: int = 256, batch_size: int = 16,
               encoding_cache: int = 2048, token_cache: int = 8192, padding_ratio: float = 1.25, strict_encoding: bool = False,
               dtype: str = "float32", embedding: Literal["mapped", "resident"] = "mapped", memory_cache_limit: int | Literal["auto"] | None = "auto") -> JuliaEngine:
    root = resolve_checkpoint(checkpoint)
    julia_config = json.loads((root / "julia_config.json").read_text())
    if julia_config.get("format_version") != 1:
        raise ValueError("unsupported Julia checkpoint format")
    config = EncoderConfig.from_hf(json.loads((root / "encoder" / "config.json").read_text()))
    limit = config.max_position_embeddings
    max_length = limit if max_length is None else max_length
    if type(max_length) is not int or not 1 <= max_length <= limit:
        raise ValueError(f"max_length must be an integer between 1 and {limit}")
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {', '.join(DTYPES)}")
    if embedding not in ("mapped", "resident"):
        raise ValueError("embedding must be 'mapped' or 'resident'")
    weights_path = root / "model.safetensors"
    with weights_path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("checkpoint contains Git LFS pointers; fetch the real model weights first")
    model = JuliaDecisionModel(config, julia_config["head_layers"], julia_config["n_act"], julia_config["dropout"])
    mapped = embedding == "mapped"
    weights = load_tensors(weights_path, frozenset({EMBEDDING}) if mapped else frozenset())
    unsupported = sorted(name for name in weights if mlx_name(name) is None)
    if unsupported:
        raise ValueError(f"unsupported trained checkpoint weights: {', '.join(unsupported)}")
    if mapped:
        model.encoder.embeddings.tok_embeddings = MappedEmbedding(map_tensor(weights_path, EMBEDDING))
    model.load_weights([(cast(str, mlx_name(name)), value) for name, value in weights.items()], strict=True)
    del weights
    if dtype != "float32":
        model.encoder.set_projection_dtype(DTYPES[dtype])
    model.eval()
    mx.eval(model.parameters())
    engine = JuliaEngine(model, JuliaTokenizer(root / "tokenizer", token_cache), root, max_length, head_length, strict_encoding,
                         batch_size, encoding_cache, padding_ratio)
    if memory_cache_limit is not None:
        engine.set_memory_cache_limit(auto_cache_limit() if memory_cache_limit == "auto" else memory_cache_limit)
    return engine


def save_model(engine: JuliaEngine, directory: str | Path) -> Path:
    """Write a checkpoint the upstream PyTorch runtime loads with ``julia.inference.load_model``."""
    root = Path(directory)
    if root.resolve() == engine.source.resolve():
        raise ValueError("refusing to overwrite the loaded checkpoint in place")
    (root / "encoder").mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine.source / "encoder" / "config.json", root / "encoder" / "config.json")
    shutil.copytree(engine.source / "tokenizer", root / "tokenizer", dirs_exist_ok=True)
    julia_config = json.loads((engine.source / "julia_config.json").read_text())
    (root / "julia_config.json").write_text(json.dumps({**julia_config, "weight_dtype": "float32"}, indent=2) + "\n")
    parameters = cast(list[tuple[str, mx.array]], tree_flatten(engine.model.parameters()))
    tensors = {torch_name(name): value.astype(mx.float32) for name, value in parameters}
    embeddings = engine.model.encoder.embeddings.tok_embeddings
    if isinstance(embeddings, MappedEmbedding):
        tensors[EMBEDDING] = mx.array(np.asarray(embeddings.table))
    mx.save_safetensors(str(root / "model.safetensors"), tensors, metadata={"format": "pt", "family": "julia"})
    return root
