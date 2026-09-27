"""Julia-1 typed-decisions release gate.

Reproduces the published CPU protocol (supersoniclabs.ia.br/data/julia-1-cpu-20260925.json):
strict encoding, 1,024-token limit, 512-token question/options budget, and Boolean
questions scored with their false/true descriptions as option text. The published
result is 1,451/2,000 (choice 426, noul 483, score 542).
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPOSITORY = "LocalLLaMA/typed-decisions"
REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
FILENAME = "all/test-00000-of-00001.parquet"
SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
PUBLISHED = {"choice": 426, "noul": 483, "score": 542}
ENGINE = {"max_length": 1024, "head_length": 512, "strict_encoding": True}
REFERENCE = Path(__file__).with_name("reference") / "typed-decisions-torch-cpu.json"
LOGIT_TOLERANCE = 2e-3


def load_questions() -> list[dict]:
    import pyarrow.parquet as pq

    path = hf_hub_download(REPOSITORY, FILENAME, repo_type="dataset", revision=REVISION)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != SHA256:
        raise ValueError("typed-decisions test split does not match the pinned sha256")
    questions = []
    for case in pq.read_table(path).to_pylist():
        gold = json.loads(case["gold"])
        for qid, question in json.loads(case["questions"]).items():
            kind, criteria = question["type"], question.get("criteria")
            if kind == "choice":
                keys, options = list(criteria), list(criteria.values())
            elif kind == "score":
                keys, options = [str(index) for index in range(len(criteria))], criteria
            else:
                keys, options = ["false", "true"], [(criteria or {}).get(key, key) for key in ("false", "true")]
            questions.append({
                "id": f"{case['id']}/{qid}",
                "row": {"state": case["state"], "question": question["instructions"], "options": options, "type": kind},
                "keys": keys,
                "gold": gold[qid]["label"],
            })
    return questions


def score(questions: list[dict], logits: list[list[float]]) -> dict:
    correct = {kind: 0 for kind in PUBLISHED}
    for question, values in zip(questions, logits):
        winner = question["keys"][max(range(len(values)), key=values.__getitem__)]
        correct[question["row"]["type"]] += winner == question["gold"]
    return {"correct": sum(correct.values()), "total": len(questions), "by_type": correct}


def agreement(questions: list[dict], logits: list[list[float]], reference: dict[str, list[float]]) -> dict:
    same, delta = 0, 0.0
    for question, values in zip(questions, logits):
        expected = reference[question["id"]]
        same += max(range(len(values)), key=values.__getitem__) == max(range(len(expected)), key=expected.__getitem__)
        delta = max(delta, *(abs(left - right) for left, right in zip(values, expected)))
    return {"top1_agreement": same, "max_abs_logit_delta": delta}


def gate_failures(report: dict) -> list[str]:
    failures = []
    if report["by_type"] != PUBLISHED:
        failures.append(f"per-type result {report['by_type']} differs from the published {PUBLISHED}")
    if report["top1_agreement"] != report["total"]:
        failures.append(f"winner differs from the PyTorch reference on {report['total'] - report['top1_agreement']} questions")
    if not report["max_abs_logit_delta"] < LOGIT_TOLERANCE:
        failures.append(f"max logit delta {report['max_abs_logit_delta']:.6f} exceeds {LOGIT_TOLERANCE}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Julia-1 typed-decisions release gate.")
    parser.add_argument("checkpoint")
    parser.add_argument("--backend", choices=["mlx", "torch"], default="mlx")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--embedding", choices=["mapped", "resident"], default="mapped")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="PyTorch reference logits to compare against")
    parser.add_argument("--write-reference", type=Path, help="write this run's logits as the reference")
    args = parser.parse_args()
    if args.write_reference and (args.write_reference.resolve() == args.reference.resolve()
                                 or (args.write_reference.exists() and os.path.samefile(args.write_reference, args.reference))):
        sys.exit("--write-reference must differ from the comparator --reference")

    questions = load_questions()
    if args.backend == "torch":
        from julia.inference import load_model  # pyright: ignore[reportMissingImports]
        engine = load_model(args.checkpoint, device="cpu", **ENGINE)
    else:
        from julia_mlx import load_model
        engine = load_model(args.checkpoint, dtype=args.dtype, embedding=args.embedding, **ENGINE)
    reference = json.loads(args.reference.read_text())
    logits = engine.logits([question["row"] for question in questions])
    report = {"backend": args.backend, "dtype": args.dtype, "embedding": args.embedding, **score(questions, logits), **agreement(questions, logits, reference)}
    print(json.dumps(report, indent=2))
    if args.write_reference:
        args.write_reference.write_text(json.dumps({question["id"]: values for question, values in zip(questions, logits)}) + "\n")
    failures = gate_failures(report)
    if args.dtype == "float32" and failures:
        sys.exit("typed-decisions gate failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
