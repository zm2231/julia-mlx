"""Julia-1 typed-decisions release gate.

Reproduces the published CPU protocol (supersoniclabs.ia.br/data/julia-1-cpu-20260925.json):
strict encoding, 1,024-token limit, 512-token question/options budget, and Boolean
questions scored with their false/true descriptions as option text. The published
result is 1,451/2,000 (choice 426, noul 483, score 542).
"""
import hashlib
import json
from pathlib import Path

from common import (
    agreement,
    agreement_failures,
    argmax,
    finish,
    load_engine,
    parse,
    parser,
    read_reference,
)
from huggingface_hub import hf_hub_download

REPOSITORY = "LocalLLaMA/typed-decisions"
REVISION = "c76749ec58bd8c3d2ea706b31c333a9059c38f90"
FILENAME = "all/test-00000-of-00001.parquet"
SHA256 = "4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c"
PUBLISHED = {"choice": 426, "noul": 483, "score": 542}
REFERENCE = Path(__file__).with_name("reference") / "typed-decisions-torch-cpu.json"


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
        correct[question["row"]["type"]] += question["keys"][argmax(values)] == question["gold"]
    return {"correct": sum(correct.values()), "total": len(questions), "by_type": correct}


def gate_failures(report: dict) -> list[str]:
    failures = [] if report["by_type"] == PUBLISHED else [f"per-type result {report['by_type']} differs from the published {PUBLISHED}"]
    return failures + agreement_failures(report, "questions")


def main() -> None:
    args = parse(parser("Julia-1 typed-decisions release gate.", REFERENCE))
    questions = load_questions()
    engine = load_engine(args)
    reference = read_reference(args.reference)
    logits = engine.logits([question["row"] for question in questions])
    ids = [question["id"] for question in questions]
    if reference and reference.keys() != set(ids):
        raise ValueError("reference does not match the typed-decisions questions")
    report = {"backend": args.backend, "dtype": args.dtype, "embedding": args.embedding, **score(questions, logits), **agreement(ids, logits, reference)}
    finish(args, report, dict(zip(ids, logits)), gate_failures(report), "typed-decisions")


if __name__ == "__main__":
    main()
