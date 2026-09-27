import os

import pytest

from julia_mlx import sequence, validate_row


def test_rejects_too_many_options():
    with pytest.raises(ValueError, match="2–20"):
        validate_row({"state": "x", "question": "q", "options": ["x"] * 21})


def test_rejects_wrong_noul_shape():
    with pytest.raises(ValueError, match="noul"):
        validate_row({"state": "x", "question": "q", "options": ["false", "true", "maybe"], "type": "noul"})


@pytest.mark.parametrize("row", [None, {"state": "x", "question": "q", "options": ["a", "b"], "target": 2}, {"state": "x", "question": "q", "options": ["a", "b"], "teacher_logits": [0.0, float("nan")]}, {"state": "x", "question": "q", "options": ["a", "b"], "teacher_logits": [True, 0.0]}])
def test_rejects_invalid_optional_training_fields(row):
    with pytest.raises(ValueError):
        validate_row(row)


@pytest.mark.skipif(not os.environ.get("JULIA_CHECKPOINT"), reason="set JULIA_CHECKPOINT to load the tokenizer")
def test_tokenizer_special_ids_and_truncation_status():
    from julia_mlx import JuliaTokenizer

    tokenizer = JuliaTokenizer(os.environ["JULIA_CHECKPOINT"] + "/tokenizer")
    assert (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id) == (2, 1, 4, 0)
    encoded = sequence(tokenizer, {"state": "context " * 100, "question": "q", "options": ["a", "b"]}, 1024, 256, strict=False)
    assert encoded["truncated"] is False
    assert sequence(tokenizer, {"state": "context " * 2000, "question": "q", "options": ["a", "b"]}, 1024, 256, strict=False)["truncated"] is True


@pytest.mark.skipif(not os.environ.get("JULIA_CHECKPOINT"), reason="set JULIA_CHECKPOINT to compare tokenizers")
def test_serialization_matches_upstream():
    pytest.importorskip("julia")
    from julia.data import (  # pyright: ignore[reportMissingImports]
        sequence as upstream_sequence,
    )
    from transformers import AutoTokenizer

    from julia_mlx import JuliaTokenizer

    root = os.environ["JULIA_CHECKPOINT"]
    ours, theirs = JuliaTokenizer(root + "/tokenizer"), AutoTokenizer.from_pretrained(root + "/tokenizer")
    rows = [
        {"state": {"ticket": "Não consigo redefinir minha senha. パスワード 🙂"}, "question": "Escalate?", "options": ["false", "true"], "type": "noul"},
        {"state": "tab\tnew\nline <mask> literal " * 40, "question": "Which team?", "options": ["Billing", "Shipping", "Access"], "type": "choice"},
        {"state": "x " * 3000, "question": "Priority?", "options": ["Low " * 60, "High"], "type": "score"},
    ]
    for row in rows:
        expected = upstream_sequence(theirs, row, 1024, 256, strict=False)
        assert sequence(ours, row, 1024, 256, strict=False) == expected


@pytest.mark.parametrize(("lengths", "max_length", "expected"), [([1025], 1025, 1025), ([100, 1000], 1025, 1024), ([1000], 1010, 1010), ([33], 8192, 64), ([7], None, 32)])
def test_collate_padding_respects_max_length(lengths, max_length, expected):
    from types import SimpleNamespace

    from julia_mlx import collate_encoded

    encoded = [{"ids": [5] * length, "markers": [1, 2], "qtype": 0} for length in lengths]
    batch = collate_encoded(SimpleNamespace(pad_token_id=0), encoded, 32, max_length)  # pyright: ignore[reportArgumentType]
    assert batch["input_ids"].shape[1] == expected
    assert batch["marker_pos"].tolist() == [[1, 2]] * len(lengths)
    mask = batch["attention_mask"]
    assert (mask is None) == (expected == max(lengths) and len(set(lengths)) == 1)
    if mask is not None:
        assert mask.sum(axis=1).tolist() == lengths


@pytest.mark.parametrize(("info", "expected"), [({}, 1 << 30), ({"max_recommended_working_set_size": 48 << 30}, 4 << 30),
                                                 ({"max_recommended_working_set_size": 8 << 30}, 1 << 30), ({"max_recommended_working_set_size": 4}, 0)])
def test_auto_cache_limit_tolerates_device_metadata(monkeypatch, info, expected):
    import mlx.core as mx

    from julia_mlx.checkpoint import auto_cache_limit

    monkeypatch.setattr(mx, "device_info", lambda: info)
    assert auto_cache_limit() == expected


def _safetensors(path, entry, data=b"\0" * 16):
    import json
    import struct

    header = json.dumps({"t": entry}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return path


def test_map_tensor_reads_fp32_bytes(tmp_path):
    import numpy as np

    from julia_mlx.weights import map_tensor

    values = np.arange(4, dtype=np.float32)
    path = _safetensors(tmp_path / "ok.safetensors", {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}, values.tobytes())
    assert map_tensor(path, "t").tolist() == [[0, 1], [2, 3]]


@pytest.mark.parametrize("entry", [{"dtype": "F32", "shape": [1], "data_offsets": [-4, 0]}, {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]},
                                   {"dtype": "F32", "shape": [1], "data_offsets": [0.0, 4]}, {"dtype": "F32", "shape": [1], "data_offsets": [False, 4]},
                                   {"dtype": "F32", "shape": [True], "data_offsets": [0, 4]}, {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
                                   {"dtype": "F32", "shape": [8], "data_offsets": [0, 32]}])
def test_map_tensor_rejects_malformed_entries(tmp_path, entry):
    from julia_mlx.weights import map_tensor

    with pytest.raises(ValueError):
        map_tensor(_safetensors(tmp_path / "bad.safetensors", entry), "t")
