from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .encoding import QTYPES, display_probabilities
from .engine import JuliaEngine


@dataclass(frozen=True)
class RouteResult:
    index: int
    candidates: tuple[int, ...]
    probabilities: tuple[float, ...]
    rounds: int
    model_rows: int
    cache_hits: int
    hierarchical: bool
    probability_scope: str


class Router:
    def __init__(self, engine: JuliaEngine, *, width: int = 20, survivors: int = 2, batch_size: int = 16, cache_size: int = 0, max_options: int = 4096):
        if not 2 <= width <= 20 or not 1 <= survivors < width:
            raise ValueError("require 2 <= width <= 20 and 1 <= survivors < width")
        if batch_size < 1 or cache_size < 0 or max_options < width:
            raise ValueError("invalid batch/cache/capacity limit")
        self.engine, self.width, self.survivors = engine, width, survivors
        self.batch_size, self.cache_size, self.max_options = batch_size, cache_size, max_options
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = threading.RLock()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def _validate(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(row, Mapping) or not isinstance(row.get("state"), (str, dict, list)) or not isinstance(row.get("question"), str):
            raise ValueError("state must be text/JSON and question must be text")
        options = row.get("options")
        if not isinstance(options, list) or not 2 <= len(options) <= self.max_options or not all(isinstance(option, str) and option for option in options):
            raise ValueError(f"options must contain 2–{self.max_options} nonempty strings")
        kind = row.get("type", "choice")
        if kind not in QTYPES or (kind == "noul" and len(options) != 2):
            raise ValueError("invalid Julia decision type")
        if kind != "choice" and len(options) > self.width:
            raise ValueError("hierarchical routing supports choice decisions only")
        return json.loads(json.dumps({"state": row["state"], "question": row["question"], "options": options, "type": kind}, ensure_ascii=False, allow_nan=False))

    @staticmethod
    def _confident_winner(scores: Sequence[float], best: int) -> bool:
        runner_up = max(score for index, score in enumerate(scores) if index != best)
        total = sum(math.exp(score - scores[best]) for score in scores)
        return 1 / total > 0.95 and math.exp(runner_up - scores[best]) / total < 0.045

    def _score(self, rows: Sequence[dict[str, Any]]) -> tuple[list[tuple[float, ...]], int, int]:
        values: list[tuple[float, ...] | None] = [None] * len(rows)
        missing: OrderedDict[str, tuple[dict[str, Any], list[int]]] = OrderedDict()
        hits = 0
        for index, row in enumerate(rows):
            key = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            if key in self._cache:
                values[index] = self._cache[key]
                self._cache.move_to_end(key)
                hits += 1
            elif key in missing:
                missing[key][1].append(index)
                hits += 1
            else:
                missing[key] = (row, [index])
        pending = list(missing.items())
        for offset in range(0, len(pending), self.batch_size):
            chunk = pending[offset:offset + self.batch_size]
            scores = self.engine.logits([entry[1][0] for entry in chunk])
            if len(scores) != len(chunk):
                raise ValueError("engine returned the wrong number of rows")
            for (key, (row, indices)), values_for_row in zip(chunk, scores):
                result = tuple(float(value) for value in values_for_row)
                if len(result) != len(row["options"]) or not all(math.isfinite(value) for value in result):
                    raise ValueError("engine logits must be finite and match option count")
                for index in indices:
                    values[index] = result
                if self.cache_size:
                    self._cache[key] = result
                    self._cache.move_to_end(key)
                    while len(self._cache) > self.cache_size:
                        self._cache.popitem(last=False)
        return [value for value in values if value is not None], len(pending), hits

    def route(self, row: Mapping[str, Any]) -> RouteResult:
        return self.route_many([row])[0]

    def route_many(self, rows: Sequence[Mapping[str, Any]]) -> list[RouteResult]:
        with self._lock:
            requests = [self._validate(row) for row in rows]
            candidates = [list(range(len(row["options"]))) for row in requests]
            results: list[RouteResult | None] = [None] * len(requests)
            rounds, model_rows, cache_hits = [0] * len(requests), 0, 0
            while any(result is None for result in results):
                jobs: list[dict[str, Any]] = []
                layout: list[tuple[int, list[int], bool]] = []
                for index, row in enumerate(requests):
                    if results[index] is not None:
                        continue
                    rounds[index] += 1
                    current, final = candidates[index], len(candidates[index]) <= self.width
                    candidates[index] = []
                    for start in range(0, len(current), self.width):
                        group = current[start:start + self.width]
                        if len(group) == 1:
                            candidates[index].extend(group)
                        else:
                            jobs.append(dict(row, options=[row["options"][option] for option in group]))
                            layout.append((index, group, final))
                scored, used, hits = self._score(jobs)
                model_rows += used
                cache_hits += hits
                for (index, group, final), scores in zip(layout, scored):
                    best = max(range(len(scores)), key=scores.__getitem__)
                    if final:
                        maximum = max(scores)
                        weights = [math.exp(score - maximum) for score in scores]
                        total = sum(weights)
                        results[index] = RouteResult(group[best], tuple(group), tuple(display_probabilities([weight / total for weight in weights])), rounds[index], 0, 0, len(requests[index]["options"]) > self.width, "final_candidates" if len(requests[index]["options"]) > self.width else "all_options")
                    elif self._confident_winner(scores, best):
                        candidates[index].append(group[best])
                    else:
                        survivors = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:min(self.survivors, len(scores))]
                        candidates[index].extend(sorted(group[position] for position in survivors))
            return [RouteResult(result.index, result.candidates, result.probabilities, result.rounds, model_rows, cache_hits, result.hierarchical, result.probability_scope) for result in results if result is not None]
