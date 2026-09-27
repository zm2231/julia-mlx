import os

import pytest

CHECKPOINT = os.environ.get("JULIA_CHECKPOINT", "")
pytestmark = pytest.mark.skipif(not CHECKPOINT, reason="set JULIA_CHECKPOINT (and PYTHONPATH to the upstream Julia source) for parity")

CHOICES = ["Billing and payment disputes", "Shipping and delivery", "Account access and login"]


def max_delta(expected, actual):
    return max(abs(left - right) for lhs, rhs in zip(expected, actual) for left, right in zip(lhs, rhs))


def argmaxes(rows):
    return [max(range(len(values)), key=values.__getitem__) for values in rows]


@pytest.fixture(scope="module")
def torch_engine():
    pytest.importorskip("julia")
    from julia.inference import load_model  # pyright: ignore[reportMissingImports]

    return load_model(CHECKPOINT, device="cpu", max_length=8192, batch_size=16)


@pytest.fixture(scope="module")
def mlx_engine():
    from julia_mlx import load_model

    return load_model(CHECKPOINT, max_length=8192, batch_size=16)


def test_mixed_types_and_padded_batch_match_upstream(torch_engine, mlx_engine):
    rows = [
        {"state": "I was charged twice for the same order.", "question": "Which team should handle this request?", "options": CHOICES, "type": "choice"},
        {"state": "A customer cannot reset their password.", "question": "Should this be escalated?", "options": ["false", "true"], "type": "noul"},
        {"state": "A longer customer message with enough words to require padding in a mixed batch. " * 12, "question": "Priority?", "options": ["Low", "Medium", "High"], "type": "score"},
        *({"state": "Customer needs help with a disputed payment. " * repetitions, "question": "Which team?", "options": ["Billing", "Shipping", "Access"], "type": "choice"} for repetitions in (12, 13, 14)),
    ]
    expected, actual = torch_engine.logits(rows), mlx_engine.logits(rows)
    assert max_delta(expected, actual) < 2e-4
    assert argmaxes(expected) == argmaxes(actual)


def test_blocked_local_attention_padded_long_batch_matches_upstream(torch_engine, mlx_engine):
    rows = [{"state": f"Case {index}: the customer describes a disputed payment and a delayed parcel. " * repetitions,
             "question": "Which team should handle this request?", "options": CHOICES, "type": "choice"}
            for index, repetitions in enumerate((45, 52, 60))]
    assert min(len(item["ids"]) for item in mlx_engine._encode(rows)) > 512
    assert max_delta(torch_engine.logits(rows), mlx_engine.logits(rows)) < 2e-4


def test_near_limit_context_matches_upstream(torch_engine, mlx_engine):
    row = {"state": "context " * 7800, "question": "Which team should handle this request?", "options": ["Billing", "Shipping", "Access", "Product"], "type": "choice"}
    assert 7800 < len(mlx_engine._encode([row])[0]["ids"]) <= 8192
    assert max_delta(torch_engine.logits([row]), mlx_engine.logits([row])) < 2e-4


def test_actions_match_upstream(torch_engine, mlx_engine):
    import mlx.core as mx

    from julia_mlx import collate

    rows = [
        {"state": "I was charged twice for the same order.", "question": "Which team?", "options": ["Billing", "Shipping", "Access"], "type": "choice"},
        {"state": "A customer cannot reset their password.", "question": "Should this be escalated?", "options": ["false", "true"], "type": "noul"},
        {"state": "context " * 180, "question": "Priority?", "options": ["Low", "Medium", "High"], "type": "score"},
    ]
    expected_scores, expected_actions = torch_engine.model(**torch_engine._pack(torch_engine._encode(rows)), return_actions=True)
    actual_scores, actual_actions = mlx_engine.model(**collate(mlx_engine.tokenizer, rows, 8192, 256, strict=False), return_actions=True)
    mx.eval(actual_scores, actual_actions)
    assert max_delta(expected_scores.tolist(), actual_scores.tolist()) < 2e-4
    assert max_delta(expected_actions.tolist(), actual_actions.tolist()) < 2e-4


def test_gradients_match_upstream(torch_engine, mlx_engine):
    import mlx.core as mx
    import numpy as np
    import torch
    from mlx.nn.losses import cross_entropy
    from mlx.nn.utils import value_and_grad
    from mlx.utils import tree_flatten

    from julia_mlx import collate
    from julia_mlx.model import mlx_name

    rows = [
        {"state": "I was charged twice for the same order.", "question": "Which team?", "options": ["Billing", "Shipping", "Access"], "type": "choice"},
        {"state": "Não consigo redefinir minha senha.", "question": "Escalate?", "options": ["false", "true"], "type": "noul"},
    ]
    targets = [0, 1]
    model = torch_engine.model
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad()
    with torch.enable_grad():
        loss = torch.nn.functional.cross_entropy(model(**torch_engine._pack(torch_engine._encode(rows))), torch.tensor(targets))
        loss.backward()
    expected = {mlx_name(name): parameter.grad.numpy() for name, parameter in model.named_parameters() if parameter.grad is not None}
    model.zero_grad()

    from julia_mlx import load_model

    resident = load_model(CHECKPOINT, max_length=8192, embedding="resident")
    batch = collate(resident.tokenizer, rows, 8192, 256, strict=False)
    _, gradients = value_and_grad(resident.model, lambda: cross_entropy(resident.model(**batch), mx.array(targets), reduction="mean"))()
    actual = dict(tree_flatten(gradients))
    assert "temperature" not in actual
    for name in ("encoder.embeddings.tok_embeddings.weight", "encoder.layers.1.attn.Wqkv.weight", "head.layers.0.attn.Wqkv.weight", "scorer.linear.weight"):
        scale = np.abs(expected[name]).max()
        assert np.abs(np.array(actual[name]) - expected[name]).max() < 1e-3 * scale


def test_saved_checkpoint_loads_in_upstream(tmp_path, torch_engine, mlx_engine):
    from julia.inference import (  # pyright: ignore[reportMissingImports]
        load_model as load_torch,
    )

    from julia_mlx import save_model

    rows = [{"state": "I was charged twice.", "question": "Which team?", "options": ["Billing", "Shipping", "Access"], "type": "choice"}]
    exported = load_torch(str(save_model(mlx_engine, tmp_path / "export")), device="cpu", max_length=1024)
    assert max_delta(torch_engine.logits(rows), exported.logits(rows)) < 1e-6


def test_router_matches_upstream_tournament(monkeypatch, torch_engine, mlx_engine):
    import julia.router.router as upstream_router  # pyright: ignore[reportMissingImports]

    from julia_mlx import Router

    class Reducer:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def argmax(values):
            return max(range(len(values)), key=values.__getitem__)

    monkeypatch.setattr(upstream_router, "BendReducer", Reducer)
    rows = [{"state": f"Customer case {index}: determine the most appropriate resolution.", "question": "Which candidate is best?",
             "options": [f"Candidate {option}" for option in range(count)], "type": "choice"} for index, count in enumerate((21, 37, 77))]
    for left, right in zip(upstream_router.Router(torch_engine, batch_size=8).route_many(rows), Router(mlx_engine, batch_size=8).route_many(rows)):
        assert (left.index, left.candidates, left.rounds, left.model_rows, left.cache_hits, left.hierarchical, left.probability_scope) == (
            right.index, right.candidates, right.rounds, right.model_rows, right.cache_hits, right.hierarchical, right.probability_scope)
        assert max(abs(a - b) for a, b in zip(left.probabilities, right.probabilities)) < 2e-4


def test_typed_api_encoding_audit_and_caches(mlx_engine):
    from julia_mlx import Router, load_model

    engine = load_model(CHECKPOINT, max_length=1024, strict_encoding=True, encoding_cache=8, token_cache=8)
    request = {"state": "I was charged twice.", "question": "Which team should handle this request?", "options": ["Billing", "Shipping", "Account access"], "type": "choice"}
    assert engine.predict([request], probabilities=False)[0]["index"] in range(3)
    assert engine.encoding_info([request])[0]["stateTruncated"] is False
    typed = engine.predict(state=request["state"], questions={"team": {"type": "choice", "instructions": request["question"], "criteria": {"billing": "Billing", "shipping": "Shipping", "access": "Account access"}}})
    assert typed["answers"]["team"]["choice"] in {"billing", "shipping", "access"}
    router = Router(engine, cache_size=16)
    large = dict(request, options=[f"Option {index}" for index in range(21)])
    first, second = router.route(large), router.route(large)
    assert first.index == second.index and second.cache_hits > 0
    assert mlx_engine.max_length == 8192


def test_mapped_embedding_matches_resident_exactly(mlx_engine):
    from mlx.utils import tree_flatten

    from julia_mlx import load_model

    rows = [{"state": "Não consigo redefinir minha senha. パスワード 🙂 " * 30, "question": "Escalate?", "options": ["false", "true"], "type": "noul"},
            {"state": "I was charged twice.", "question": "Which team?", "options": CHOICES, "type": "choice"}]
    resident = load_model(CHECKPOINT, max_length=8192, embedding="resident")
    assert mlx_engine.logits(rows) == resident.logits(rows)
    assert "encoder.embeddings.tok_embeddings.weight" not in dict(tree_flatten(mlx_engine.model.parameters()))
