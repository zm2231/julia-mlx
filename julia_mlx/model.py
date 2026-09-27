from __future__ import annotations

import math
from typing import Literal, overload

import mlx.core as mx
from mlx import nn

from .modernbert import EncoderConfig, ModernBertModel, attention_masks

TORCH_TO_MLX = (
    ("scorer.0.", "scorer.norm."), ("scorer.1.", "scorer.linear."), ("scorer.3.", "scorer.out."),
    ("act_head.0.", "act_head.first."), ("act_head.2.", "act_head.out."),
    (".self_attn.in_proj_", ".attn.Wqkv."), (".self_attn.out_proj.", ".attn.Wo."),
)


def _shared(name: str) -> bool:
    return name.startswith("encoder.") or name in {"type_emb.weight", "temperature"}


def mlx_name(name: str) -> str | None:
    if _shared(name):
        return name
    for source, target in TORCH_TO_MLX:
        if source in name:
            return name.replace(source, target, 1)
    return name if name.startswith("head.layers.") else None


def torch_name(name: str) -> str:
    if _shared(name):
        return name
    for source, target in TORCH_TO_MLX:
        if target in name:
            return name.replace(target, source, 1)
    return name


class JuliaHeadAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.width = heads, width
        self.Wqkv, self.Wo = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.weights_dropout = nn.Dropout(dropout) if dropout else None

    def split(self, values: mx.array) -> mx.array:
        return values.reshape(*values.shape[:2], self.heads, self.width // self.heads).transpose(0, 2, 1, 3)

    def attend(self, query: mx.array, key: mx.array, value: mx.array, mask: mx.array | None) -> mx.array:
        scale = (self.width // self.heads) ** -0.5
        if self.training and self.weights_dropout is not None:
            scores = (query * scale) @ key.transpose(0, 1, 3, 2)
            if mask is not None:
                scores = mx.where(mask, scores, -math.inf)
            weights = self.weights_dropout(mx.softmax(scores, axis=-1, precise=True))
            output = weights @ value
        else:
            output = mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)
        return self.Wo(output.transpose(0, 2, 1, 3).reshape(query.shape[0], query.shape[2], self.width))

    def __call__(self, values: mx.array, mask: mx.array | None) -> mx.array:
        query, key, value = mx.split(self.Wqkv(values), 3, axis=-1)
        return self.attend(self.split(query), self.split(key), self.split(value), mask)


class JuliaHeadLayer(nn.Module):
    """PyTorch TransformerEncoderLayer(norm_first=True, activation=relu)."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(width, eps=1e-5), nn.LayerNorm(width, eps=1e-5)
        self.attn = JuliaHeadAttention(width, heads, dropout)
        self.linear1, self.linear2 = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)
        self.dropout1, self.dropout, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def feed_forward(self, values: mx.array) -> mx.array:
        return values + self.dropout2(self.linear2(self.dropout(nn.relu(self.linear1(self.norm2(values))))))

    def __call__(self, values: mx.array, mask: mx.array | None) -> mx.array:
        return self.feed_forward(values + self.dropout1(self.attn(self.norm1(values), mask)))

    def selected(self, values: mx.array, positions: mx.array, mask: mx.array | None) -> mx.array:
        rows = mx.arange(values.shape[0])[:, None]
        normalized = self.norm1(values)
        width = self.attn.width
        query = normalized[rows, positions] @ self.attn.Wqkv.weight[:width].T + self.attn.Wqkv.bias[:width]
        key, value = mx.split(normalized @ self.attn.Wqkv.weight[width:].T + self.attn.Wqkv.bias[width:], 2, axis=-1)
        attention = self.attn.attend(self.attn.split(query), self.attn.split(key), self.attn.split(value), mask)
        return self.feed_forward(values[rows, positions] + attention)


class JuliaHead(nn.Module):
    def __init__(self, width: int, layers: int, dropout: float):
        super().__init__()
        self.layers = [JuliaHeadLayer(width, max(1, width // 64), dropout) for _ in range(layers)]


class JuliaScorer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm, self.linear, self.out = nn.LayerNorm(width, eps=1e-5), nn.Linear(width, width), nn.Linear(width, 1)

    def __call__(self, values: mx.array) -> mx.array:
        return self.out(nn.gelu(self.linear(self.norm(values)))).squeeze(-1)


class JuliaActionHead(nn.Module):
    def __init__(self, width: int, n_actions: int):
        super().__init__()
        self.first, self.out = nn.Linear(width + 4, 256), nn.Linear(256, n_actions)

    def __call__(self, values: mx.array) -> mx.array:
        return self.out(nn.gelu(self.first(values)))


class JuliaDecisionModel(nn.Module):
    def __init__(self, config: EncoderConfig, head_layers: int, n_act: int, dropout: float):
        super().__init__()
        width = config.hidden_size
        self.encoder, self.type_emb = ModernBertModel(config), nn.Embedding(3, width)
        self.head = JuliaHead(width, head_layers, dropout)
        self.scorer, self.act_head = JuliaScorer(width), JuliaActionHead(width, n_act)
        self.temperature = mx.ones((3,), dtype=mx.float32)
        self.freeze(keys=["temperature"], recurse=False)

    @overload
    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None, marker_pos: mx.array, marker_mask: mx.array, qtype: mx.array, return_actions: Literal[False] = False) -> mx.array: ...

    @overload
    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None, marker_pos: mx.array, marker_mask: mx.array, qtype: mx.array, return_actions: Literal[True]) -> tuple[mx.array, mx.array]: ...

    def __call__(self, input_ids: mx.array, attention_mask: mx.array | None, marker_pos: mx.array, marker_mask: mx.array, qtype: mx.array, return_actions: bool = False) -> mx.array | tuple[mx.array, mx.array]:
        batch, tokens = input_ids.shape
        masks = attention_masks(attention_mask, batch, tokens, self.encoder.config.local_attention // 2)
        hidden = self.encoder(input_ids, masks) + self.type_emb(qtype)[:, None, :]
        selected = mx.concatenate([mx.zeros_like(marker_pos[:, :1]), marker_pos], axis=-1) if return_actions else marker_pos
        rows = mx.arange(batch)[:, None]
        layers = self.head.layers
        if not layers:
            hidden = hidden[rows, selected]
        elif self.training:
            for layer in layers:
                hidden = layer(hidden, masks.global_mask)
            hidden = hidden[rows, selected]
        else:
            for layer in layers[:-1]:
                hidden = layer(hidden, masks.global_mask)
            hidden = layers[-1].selected(hidden, selected, masks.global_mask)
        scores = self.scorer(hidden[:, 1:] if return_actions else hidden).astype(mx.float32)
        scores = mx.where(marker_mask, scores, -1e4)
        if not return_actions:
            return scores
        probabilities = mx.softmax(mx.stop_gradient(scores), axis=-1)
        top = mx.flip(mx.topk(probabilities, k=min(2, probabilities.shape[-1]), axis=-1), axis=-1)
        if top.shape[-1] == 1:
            top = mx.concatenate([top, mx.zeros_like(top)], axis=-1)
        count = mx.maximum(mx.sum(marker_mask, axis=-1).astype(mx.float32), 2)
        entropy = -mx.sum(probabilities * mx.log(mx.maximum(probabilities, 1e-9)), axis=-1) / mx.log(count)
        features = mx.stack([top[:, 0], top[:, 0] - top[:, 1], entropy, count / 255], axis=-1)
        return scores, self.act_head(mx.concatenate([hidden[:, 0].astype(mx.float32), features], axis=-1).astype(self.act_head.first.weight.dtype))
