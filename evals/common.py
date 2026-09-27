import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ENGINE = {"max_length": 1024, "head_length": 512, "strict_encoding": True}
LOGIT_TOLERANCE = 2e-3


def parser(description: str, reference: Path) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("checkpoint")
    result.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    result.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    result.add_argument("--embedding", choices=["mapped", "resident"], default="mapped")
    result.add_argument("--reference", type=Path, default=reference, help="PyTorch reference outputs to compare against")
    result.add_argument("--write-reference", type=Path, help="write this run's outputs as the reference")
    return result


def parse(arguments: argparse.ArgumentParser) -> argparse.Namespace:
    args = arguments.parse_args()
    if args.write_reference and (args.write_reference.resolve() == args.reference.resolve()
                                 or (args.write_reference.exists() and os.path.samefile(args.write_reference, args.reference))):
        sys.exit("--write-reference must differ from the comparator --reference")
    if not args.reference.exists() and not args.write_reference:
        sys.exit(f"reference {args.reference} does not exist")
    return args


def load_engine(args: argparse.Namespace) -> Any:
    if args.backend == "torch":
        from julia.inference import load_model  # pyright: ignore[reportMissingImports]
        return load_model(args.checkpoint, device="cpu", **ENGINE)
    from julia_mlx import load_model
    return load_model(args.checkpoint, dtype=args.dtype, embedding=args.embedding, **ENGINE)


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def agreement(ids: Sequence[str], logits: Sequence[Sequence[float]], reference: dict[str, Any]) -> dict:
    if not reference:
        return {"top1_agreement": None, "max_abs_logit_delta": None}
    if len(ids) != len(logits) or not set(ids) <= reference.keys():
        raise ValueError("reference does not cover every evaluated example")
    same, delta = 0, 0.0
    for identifier, values in zip(ids, logits):
        expected = reference[identifier]
        if len(expected) != len(values):
            raise ValueError(f"reference for {identifier} has {len(expected)} logits, expected {len(values)}")
        same += argmax(values) == argmax(expected)
        delta = max(delta, *(abs(left - right) for left, right in zip(values, expected)))
    return {"top1_agreement": same, "max_abs_logit_delta": delta}


def read_reference(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def agreement_failures(report: dict, unit: str) -> list[str]:
    if report["top1_agreement"] is None:
        return []
    failures = []
    if report["top1_agreement"] != report["total"]:
        failures.append(f"winner differs from the PyTorch reference on {report['total'] - report['top1_agreement']} {unit}")
    if not report["max_abs_logit_delta"] < LOGIT_TOLERANCE:
        failures.append(f"max logit delta {report['max_abs_logit_delta']:.6f} exceeds {LOGIT_TOLERANCE}")
    return failures


def finish(args: argparse.Namespace, report: dict, outputs: dict[str, Any], failures: list[str], name: str) -> None:
    print(json.dumps(report, indent=2))
    if args.write_reference:
        args.write_reference.write_text(json.dumps(outputs) + "\n")
    if args.dtype == "float32" and failures:
        sys.exit(f"{name} gate failed: " + "; ".join(failures))
