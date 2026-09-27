"""One Julia-1 benchmark measurement in a fresh process; run through benchmarks/suite.py."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))
from typed_decisions import load_questions


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def long_rows(questions: list[dict], count: int) -> list[dict]:
    states = [question["row"]["state"] for question in questions[::5]]
    rows = []
    for index in range(count):
        text, cursor = "", index * 7
        while len(text) < 40000:
            text += states[cursor % len(states)] + "\n"
            cursor += 1
        rows.append({"state": text, "question": "Which team should handle this request?", "options": ["Billing", "Shipping", "Access", "Product"], "type": "choice"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("workload", choices=["single", "batch", "long"])
    parser.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--embedding", choices=["mapped", "resident"], default="mapped")
    args = parser.parse_args()

    questions = load_questions()
    long = args.workload == "long"
    settings = {"max_length": 8192 if long else 1024, "head_length": 512, "strict_encoding": not long, "batch_size": 16, "encoding_cache": 0, "token_cache": 0}
    started = time.perf_counter()
    if args.backend == "torch":
        from julia.inference import load_model  # pyright: ignore[reportMissingImports]
        engine = load_model(args.checkpoint, device="cpu", **settings)
    else:
        from julia_mlx import load_model
        engine = load_model(args.checkpoint, dtype=args.dtype, embedding=args.embedding, **settings)
    report = {"backend": args.backend, "dtype": args.dtype, "embedding": args.embedding, "workload": args.workload,
              "load_s": time.perf_counter() - started}
    rows = [question["row"] for question in questions]

    if args.workload == "single":
        for row in rows[1600:1650]:
            engine.logits([row])
        elapsed = []
        for row in rows[:400]:
            started = time.perf_counter()
            engine.logits([row])
            elapsed.append(time.perf_counter() - started)
        report.update(calls=len(elapsed), median_ms=statistics.median(elapsed) * 1e3, p95_ms=percentile(elapsed, .95) * 1e3, p99_ms=percentile(elapsed, .99) * 1e3)
    elif args.workload == "batch":
        engine.logits(rows)
        elapsed = []
        for _ in range(5):
            started = time.perf_counter()
            engine.logits(rows)
            elapsed.append(time.perf_counter() - started)
        report.update(decisions=len(rows), passes=len(elapsed), median_s=statistics.median(elapsed), min_s=min(elapsed), max_s=max(elapsed),
                      decisions_per_s=len(rows) / statistics.median(elapsed))
    else:
        batch = long_rows(questions, 16)
        engine.logits(batch[:1])
        started = time.perf_counter()
        engine.logits(batch)
        report.update(rows=len(batch), tokens_per_row=len(engine._encode(batch[:1])[0]["ids"]), seconds=time.perf_counter() - started)
    print(json.dumps({key: round(value, 2) if isinstance(value, float) else value for key, value in report.items()}))


if __name__ == "__main__":
    main()
