from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, overload

import mlx.core as mx

from .encoding import collate_encoded, display_probabilities, sequence, validate_row
from .model import JuliaDecisionModel
from .tokenizer import JuliaTokenizer

PAD_MULTIPLE = 32

State = str | dict[str, Any] | list[Any]


@dataclass
class JuliaEngine:
    model: JuliaDecisionModel
    tokenizer: JuliaTokenizer
    source: Path
    max_length: int
    head_length: int
    strict_encoding: bool
    batch_size: int = 16
    encoding_cache: int = 2048
    padding_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.encoding_cache < 0:
            raise ValueError("invalid batch/cache size")
        if not 1 <= self.padding_ratio <= 16:
            raise ValueError("padding_ratio must be between 1 and 16")
        self._encoded: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def clear_cache(self) -> None:
        with self._lock:
            self._encoded.clear()
            self.tokenizer.cache.clear()

    @staticmethod
    def memory_stats() -> dict[str, int]:
        return {"active_bytes": int(mx.get_active_memory()), "cache_bytes": int(mx.get_cache_memory()), "peak_bytes": int(mx.get_peak_memory())}

    @staticmethod
    def set_memory_cache_limit(limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError("memory cache limit must be a nonnegative integer of bytes")
        mx.set_cache_limit(limit_bytes)

    @staticmethod
    def trim_memory() -> None:
        mx.clear_cache()

    def _encode(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        encoded = []
        for row in rows:
            validate_row(row)
            key = json.dumps([self.max_length, self.head_length, self.strict_encoding, row["state"], row["question"], row["options"], row.get("type", "choice")], ensure_ascii=False, allow_nan=False)
            item = self._encoded.get(key)
            if item is None:
                item = sequence(self.tokenizer, row, self.max_length, self.head_length, strict=self.strict_encoding)
                if self.encoding_cache:
                    self._encoded[key] = item
                    while len(self._encoded) > self.encoding_cache:
                        self._encoded.popitem(last=False)
            else:
                self._encoded.move_to_end(key)
            encoded.append(item)
        return encoded

    def encoding_info(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not self.strict_encoding:
            raise ValueError("lossless encoding audit requires strict_encoding=True")
        with self._lock:
            return [{"tokens": len(item["ids"]), "optionTokens": list(item["option_tokens"]), "headLength": self.head_length, "stateTruncated": False, "optionsTruncated": False} for item in self._encode(rows)]

    def _batch_indices(self, encoded: Sequence[Mapping[str, Any]]) -> Iterator[list[int]]:
        order = sorted(range(len(encoded)), key=lambda index: len(encoded[index]["ids"]))
        group: list[int] = []
        tokens = 0
        for index in order:
            length = len(encoded[index]["ids"])
            if group and (len(group) == self.batch_size or (len(group) + 1) * length > self.padding_ratio * (tokens + length)):
                yield group
                group, tokens = [], 0
            group.append(index)
            tokens += length
        if group:
            yield group

    def logits(self, rows: Sequence[Mapping[str, Any]]) -> list[list[float]]:
        if not rows:
            return []
        with self._lock:
            encoded = self._encode(rows)
            result: list[list[float]] = [[] for _ in rows]
            pending: tuple[list[int], mx.array] | None = None
            for indices in [*self._batch_indices(encoded), None]:
                launched = None
                if indices is not None:
                    scores = self.model(**collate_encoded(self.tokenizer, [encoded[index] for index in indices], PAD_MULTIPLE, self.max_length))
                    mx.async_eval(scores)
                    launched = (indices, scores)
                if pending is not None:
                    for index, values in zip(pending[0], cast(list[list[float]], pending[1].tolist())):
                        values = values[:len(encoded[index]["markers"])]
                        if not all(math.isfinite(value) for value in values):
                            raise FloatingPointError("inference returned nonfinite logits")
                        result[index] = values
                pending = launched
            return result

    @overload
    def predict(self, rows: Sequence[Mapping[str, Any]], questions: None = None, *, state: None = None, probabilities: bool = True) -> list[dict[str, Any]]: ...

    @overload
    def predict(self, rows: State | None = None, *, questions: Mapping[str, Any], state: State | None = None, probabilities: bool = True) -> dict[str, Any]: ...

    def predict(self, rows: Sequence[Mapping[str, Any]] | State | None = None, questions: Mapping[str, Any] | None = None, *, state: State | None = None, probabilities: bool = True) -> list[dict[str, Any]] | dict[str, Any]:
        if questions is not None:
            if rows is not None and state is not None:
                raise ValueError("pass state either positionally or by keyword, not both")
            return self.predict_typed(cast(State, rows) if rows is not None else state, questions)
        if rows is None or state is not None:
            raise ValueError("provide legacy rows or state with questions")
        result = []
        for scores in self.logits(cast(Sequence[Mapping[str, Any]], rows)):
            item: dict[str, Any] = {"index": max(range(len(scores)), key=scores.__getitem__)}
            if probabilities:
                item["probabilities"] = display_probabilities(_softmax(scores))
            result.append(item)
        return result

    def predict_typed(self, state: State | None, questions: Mapping[str, Any]) -> dict[str, Any]:
        if state is None or not isinstance(questions, Mapping) or not questions:
            raise ValueError("state and a nonempty question mapping are required")
        rows, metadata = [], []
        for identifier, question in questions.items():
            if not isinstance(identifier, str) or not identifier or not isinstance(question, Mapping):
                raise ValueError("questions require nonempty string IDs and question objects")
            kind, criteria = question.get("type"), question.get("criteria")
            if kind == "choice":
                if not isinstance(criteria, Mapping) or not criteria or any(not isinstance(key, str) or not key for key in criteria):
                    raise ValueError("choice criteria must map nonempty IDs to descriptions")
                keys, labels = list(criteria), list(criteria.values())
            elif kind == "score":
                if not isinstance(criteria, list):
                    raise ValueError("score requires an ordered rubric")
                keys, labels = [str(index) for index in range(len(criteria))], criteria
            elif kind == "noul":
                keys, labels = ["false", "true"], ["false", "true"]
            else:
                raise ValueError("unsupported question type")
            rows.append({"state": state, "question": question.get("instructions"), "type": kind, "options": labels})
            metadata.append((identifier, kind, keys))
        answers = {}
        for (identifier, kind, keys), scores in zip(metadata, self.logits(rows)):
            probabilities = _softmax(scores)
            answer: dict[str, Any] = {"type": kind, "probabilities": dict(zip(keys, probabilities))}
            if kind == "choice":
                answer["choice"] = keys[max(range(len(probabilities)), key=probabilities.__getitem__)]
            elif kind == "score":
                answer["score"] = sum(index * value for index, value in enumerate(probabilities))
            else:
                answer["noul"] = probabilities[1]
            if kind != "noul":
                answer["max_probability"] = max(probabilities)
            answers[identifier] = answer
        return {"answers": answers}


def _softmax(scores: Sequence[float]) -> list[float]:
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]
