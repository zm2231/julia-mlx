from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

import mlx.core as mx

from .tokenizer import JuliaTokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}


class Batch(TypedDict):
    input_ids: mx.array
    attention_mask: mx.array | None
    marker_pos: mx.array
    marker_mask: mx.array
    qtype: mx.array


def display_probabilities(probabilities: Sequence[float]) -> list[float]:
    values = list(probabilities)
    winner = max(range(len(values)), key=values.__getitem__)
    if values[winner] > 0.95 and all(value < 0.045 for index, value in enumerate(values) if index != winner):
        return [1.0 if index == winner else 0.0 for index in range(len(values))]
    visible = [value if value >= 0.01 else 0.0 for value in values]
    total = sum(visible)
    return [value / total for value in visible]


def validate_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("request must be a JSON object")
    if not isinstance(row.get("state"), (str, dict, list)) or not isinstance(row.get("question"), str):
        raise ValueError("state must be text/JSON and question must be text")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 20:
        raise ValueError("options must contain 2–20 values")
    if not all(isinstance(option, str) and option for option in options):
        raise ValueError("options must contain nonempty strings")
    if row.get("type", "choice") not in QTYPES:
        raise ValueError("type must be choice, score, or noul")
    if row.get("type") == "noul" and len(options) != 2:
        raise ValueError("noul options must be ordered [false, true]")
    if "target" in row and (type(row["target"]) is not int or not 0 <= row["target"] < len(options)):
        raise ValueError("target must index the supplied option list")
    teacher = row.get("teacher_logits")
    if teacher is not None and (
        not isinstance(teacher, list)
        or len(teacher) != len(options)
        or not all(type(value) in (int, float) and math.isfinite(value) for value in teacher)
    ):
        raise ValueError("teacher logits must be finite and match option count/order")


def sequence(tokenizer: JuliaTokenizer, row: Mapping[str, Any], max_length: int, head_length: int, *, strict: bool) -> dict[str, Any]:
    validate_row(row)
    if head_length + 4 >= max_length:
        raise ValueError("max_length must leave room beyond the question head")
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    qtype, reserved = row.get("type", "choice"), tokenizer.mask_token
    if strict and any(reserved in text for text in [state, row["question"], *row["options"]]):
        raise ValueError("reserved model marker in request")

    def clean(text: str) -> str:
        return text.replace(reserved, " ")

    head = tokenizer.encode(f"{qtype} question: {clean(row['question'])}")
    option_ids = [tokenizer.encode(" " + clean(option)) for option in row["options"]]
    if strict and any(len(option) > 48 for option in option_ids):
        raise ValueError("option exceeds 48-token model contract")
    options = [[tokenizer.mask_token_id] + option[:48] for option in option_ids]
    budget = head_length - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_length - 16) // len(options))
        options = [option[:per_option] for option in options]
        budget = head_length - sum(map(len, options))
    if strict and (len(head) > budget or any(len(option) != len(original) + 1 for option, original in zip(options, option_ids))):
        raise ValueError("question/options exceed lossless head budget")
    ids, markers = [tokenizer.cls_token_id] + head[:max(8, budget)] + [tokenizer.sep_token_id], []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    state_ids, room = tokenizer.encode(clean(state)), max_length - len(ids) - 1
    if room < 1:
        raise ValueError("question/options exceed sequence budget; shorten descriptions")
    if strict and len(state_ids) > room:
        raise ValueError("state exceeds lossless context budget")
    result = {"ids": ids + state_ids[:room] + [tokenizer.sep_token_id], "markers": markers, "qtype": QTYPES[qtype], "truncated": len(state_ids) > room}
    if strict:
        result["option_tokens"] = list(map(len, option_ids))
    return result


def collate_encoded(tokenizer: JuliaTokenizer, encoded: Sequence[Mapping[str, Any]], pad_multiple: int = 1, max_length: int | None = None) -> Batch:
    lengths = [len(item["ids"]) for item in encoded]
    length = -(-max(lengths) // pad_multiple) * pad_multiple
    if max_length is not None:
        length = max(max(lengths), min(length, max_length))
    options = max(len(item["markers"]) for item in encoded)
    ids = [item["ids"] + [tokenizer.pad_token_id] * (length - len(item["ids"])) for item in encoded]
    markers = [item["markers"] + [0] * (options - len(item["markers"])) for item in encoded]
    marker_mask = [[True] * len(item["markers"]) + [False] * (options - len(item["markers"])) for item in encoded]
    attention = None
    if min(lengths) != length:
        attention = mx.array([[1] * size + [0] * (length - size) for size in lengths], dtype=mx.int32)
    return {"input_ids": mx.array(ids, dtype=mx.int32), "attention_mask": attention,
            "marker_pos": mx.array(markers, dtype=mx.int32), "marker_mask": mx.array(marker_mask),
            "qtype": mx.array([item["qtype"] for item in encoded], dtype=mx.int32)}


def collate(tokenizer: JuliaTokenizer, rows: Sequence[Mapping[str, Any]], max_length: int, head_length: int, *, strict: bool) -> Batch:
    return collate_encoded(tokenizer, [sequence(tokenizer, row, max_length, head_length, strict=strict) for row in rows], max_length=max_length)
