"""ModernBERT encoder for Julia inference, derived from pappitti/modernbert-mlx (MIT)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

BLOCKED_LOCAL_MIN_TOKENS = 512


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    layer_types: tuple[str, ...]
    local_attention: int
    global_rope_theta: float
    local_rope_theta: float
    norm_eps: float
    norm_bias: bool
    attention_bias: bool
    mlp_bias: bool
    max_position_embeddings: int

    @classmethod
    def from_hf(cls, config: dict[str, Any]) -> EncoderConfig:
        rope = config.get("rope_parameters", {})
        layers = config.get("layer_types") or [
            "full_attention" if index % config["global_attn_every_n_layers"] == 0 else "sliding_attention"
            for index in range(config["num_hidden_layers"])]
        if config.get("hidden_activation", "gelu") != "gelu":
            raise ValueError("Julia MLX supports the GELU ModernBERT encoder only")
        return cls(
            vocab_size=config["vocab_size"], hidden_size=config["hidden_size"], intermediate_size=config["intermediate_size"],
            num_attention_heads=config["num_attention_heads"], layer_types=tuple(layers), local_attention=config["local_attention"],
            global_rope_theta=rope.get("full_attention", {}).get("rope_theta", config.get("global_rope_theta", 160000.0)),
            local_rope_theta=rope.get("sliding_attention", {}).get("rope_theta", config.get("local_rope_theta", 10000.0)),
            norm_eps=config["norm_eps"], norm_bias=config["norm_bias"], attention_bias=config["attention_bias"],
            mlp_bias=config["mlp_bias"], max_position_embeddings=config["max_position_embeddings"])


@dataclass(frozen=True)
class AttentionMasks:
    """Key masks shared by every layer of one forward pass; True marks an attendable key."""
    valid: mx.array
    padded: bool
    global_mask: mx.array | None
    local_mask: mx.array | None
    blocked: bool


def attention_masks(attention_mask: mx.array | None, batch: int, tokens: int, half_window: int) -> AttentionMasks:
    padded = attention_mask is not None
    valid = attention_mask.astype(mx.bool_) if padded else mx.ones((batch, tokens), dtype=mx.bool_)
    global_mask = valid[:, None, None, :] if padded else None
    if tokens <= half_window + 1:
        return AttentionMasks(valid, padded, global_mask, global_mask, False)
    if tokens > BLOCKED_LOCAL_MIN_TOKENS:
        return AttentionMasks(valid, padded, global_mask, None, True)
    positions = mx.arange(tokens)
    local = (mx.abs(positions[None, :] - positions[:, None]) <= half_window)[None, None]
    if padded:
        # Padded query rows keep their own diagonal so no softmax row is empty; valid
        # queries never see padded keys, so this cannot change a real token.
        local = (local & valid[:, None, None, :]) | ((~valid)[:, None, :, None] & mx.eye(tokens, dtype=mx.bool_)[None, None])
    return AttentionMasks(valid, padded, global_mask, local, False)


def blocked_local_attention(query: mx.array, key: mx.array, value: mx.array, scale: float, valid: mx.array, half_window: int) -> mx.array:
    """Exact sliding-window attention in O(tokens * window) time and memory.

    Queries are grouped in blocks of ``half_window``; every key a block's queries may
    reach lies in the previous, current, or next block.
    """
    batch, heads, tokens, width = query.shape
    block = half_window
    blocks = -(-tokens // block)
    tail = blocks * block - tokens

    def pad(values: mx.array, before: int, after: int) -> mx.array:
        return mx.pad(values, [(0, 0), (0, 0), (before, after), (0, 0)])

    def windows(values: mx.array) -> mx.array:
        return mx.concatenate([values[:, :, offset * block:(offset + blocks) * block].reshape(values.shape[0], values.shape[1], blocks, block, *values.shape[3:])
                               for offset in range(3)], axis=3)

    query = pad(query, 0, tail).reshape(batch, heads, blocks, block, width)
    key, value = windows(pad(key, block, block + tail)), windows(pad(value, block, block + tail))
    key_valid = mx.pad(valid, [(0, 0), (block, block + tail)])
    key_valid = mx.concatenate([key_valid[:, offset * block:(offset + blocks) * block].reshape(batch, blocks, block) for offset in range(3)], axis=2)
    query_valid = mx.pad(valid, [(0, 0), (0, tail)]).reshape(batch, blocks, block)
    rows, columns = mx.arange(block)[:, None], mx.arange(3 * block)[None, :]
    mask = (mx.abs(rows - (columns - block)) <= half_window)[None, None] & key_valid[:, :, None, :]
    mask = mask | ((~query_valid)[:, :, :, None] & mx.equal(columns, rows + block)[None, None])

    def fold(values: mx.array) -> mx.array:
        return values.transpose(0, 2, 1, 3, 4).reshape(batch * blocks, heads, values.shape[3], width)

    output = mx.fast.scaled_dot_product_attention(fold(query), fold(key), fold(value), scale=scale, mask=mask.reshape(batch * blocks, 1, block, 3 * block))
    output = output.reshape(batch, blocks, heads, block, width).transpose(0, 2, 1, 3, 4).reshape(batch, heads, blocks * block, width)
    return output[:, :, :tokens]


class MappedEmbedding:
    """Vocabulary rows read from a memory-mapped FP32 table; not an MLX parameter."""

    def __init__(self, table: np.memmap):
        self.table = table

    def __call__(self, input_ids: mx.array) -> mx.array:
        ids = np.asarray(input_ids)
        unique, inverse = np.unique(ids, return_inverse=True)
        return mx.array(self.table[unique])[mx.array(inverse.reshape(ids.shape))]


class ModernBertEmbeddings(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.tok_embeddings: nn.Embedding | MappedEmbedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=config.mlp_bias)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)

    def __call__(self, hidden: mx.array) -> mx.array:
        values, gate = mx.split(self.Wi(hidden.astype(self.Wi.weight.dtype)), 2, axis=-1)
        return self.Wo(nn.gelu(values) * gate).astype(hidden.dtype)


class ModernBertAttention(nn.Module):
    def __init__(self, config: EncoderConfig, local: bool):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.heads, self.width = config.num_attention_heads, config.hidden_size // config.num_attention_heads
        self.local, self.half_window = local, config.local_attention // 2
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.attention_bias)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.rotary_emb = nn.RoPE(self.width, base=config.local_rope_theta if local else config.global_rope_theta)

    def __call__(self, hidden: mx.array, masks: AttentionMasks) -> mx.array:
        batch, tokens, _ = hidden.shape
        qkv = self.Wqkv(hidden.astype(self.Wqkv.weight.dtype)).reshape(batch, tokens, 3, self.heads, self.width).transpose(2, 0, 3, 1, 4)
        query, key, value = self.rotary_emb(qkv[0]), self.rotary_emb(qkv[1]), qkv[2]
        scale = 1 / math.sqrt(self.width)
        if self.local and masks.blocked:
            output = blocked_local_attention(query, key, value, scale, masks.valid, self.half_window)
        else:
            mask = masks.local_mask if self.local else masks.global_mask
            output = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)
        return self.Wo(output.transpose(0, 2, 1, 3).reshape(batch, tokens, -1)).astype(hidden.dtype)


class ModernBertEncoderLayer(nn.Module):
    def __init__(self, config: EncoderConfig, index: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias) if index else nn.Identity()
        self.attn = ModernBertAttention(config, config.layer_types[index] == "sliding_attention")
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)
        self.mlp = ModernBertMLP(config)

    def __call__(self, hidden: mx.array, masks: AttentionMasks) -> mx.array:
        hidden = hidden + self.attn(self.attn_norm(hidden), masks)
        return hidden + self.mlp(self.mlp_norm(hidden))


class ModernBertModel(nn.Module):
    """The residual stream, norms, and embeddings stay FP32: from the middle layers on,
    Julia-1's residual carries outlier channels near 4,000, where FP16 spacing is 4."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.embeddings = ModernBertEmbeddings(config)
        self.layers = [ModernBertEncoderLayer(config, index) for index in range(len(config.layer_types))]
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)

    def set_projection_dtype(self, dtype: mx.Dtype) -> None:
        for layer in self.layers:
            for projection in (layer.attn.Wqkv, layer.attn.Wo, layer.mlp.Wi, layer.mlp.Wo):
                projection.set_dtype(dtype)

    def __call__(self, input_ids: mx.array, masks: AttentionMasks) -> mx.array:
        hidden = self.embeddings(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, masks)
        return self.final_norm(hidden)
