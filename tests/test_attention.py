import mlx.core as mx
import pytest

from julia_mlx.modernbert import (
    BLOCKED_LOCAL_MIN_TOKENS,
    attention_masks,
    blocked_local_attention,
)


def dense_local(query, key, value, scale, valid, half_window):
    tokens = query.shape[2]
    positions = mx.arange(tokens)
    window = (mx.abs(positions[None, :] - positions[:, None]) <= half_window)[None, None]
    mask = (window & valid[:, None, None, :]) | ((~valid)[:, None, :, None] & mx.eye(tokens, dtype=mx.bool_)[None, None])
    return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)


@pytest.mark.parametrize("tokens", [BLOCKED_LOCAL_MIN_TOKENS + 1, 640, 1000, 1537])
def test_blocked_local_attention_matches_dense(tokens):
    mx.random.seed(tokens)
    batch, heads, width, half = 3, 6, 64, 64
    query, key, value = (mx.random.normal((batch, heads, tokens, width)) for _ in range(3))
    lengths = mx.array([tokens, tokens - 97, 70])
    valid = mx.arange(tokens)[None, :] < lengths[:, None]
    expected = dense_local(query, key, value, width ** -0.5, valid, half)
    actual = blocked_local_attention(query, key, value, width ** -0.5, valid, half)
    real = valid[:, None, :, None]
    assert float(mx.max(mx.abs(mx.where(real, expected - actual, 0)))) < 1e-5
    assert mx.all(mx.isfinite(actual)).item()


def test_masks_choose_strategy_by_length():
    assert attention_masks(None, 1, 60, 64).local_mask is None
    assert not attention_masks(None, 1, 300, 64).blocked
    assert attention_masks(None, 1, BLOCKED_LOCAL_MIN_TOKENS + 1, 64).blocked
    padded = attention_masks(mx.array([[1, 1, 0]] * 2), 2, 3, 64)
    assert padded.padded and padded.global_mask is not None and padded.global_mask.shape == (2, 1, 1, 3)
