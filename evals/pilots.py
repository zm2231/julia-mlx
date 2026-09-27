"""Julia-1 classification pilots (BTZSC AG News, DAIR Emotion, Banking77).

Rebuilds the 100-example samples of the pinned Jev protocol
(github.com/AbdelStark/jev-benchmarks at 0d610cc, configs/pilot-v1.yaml) and checks them
against its published manifest hash. Each example is one choice question over the BTZSC
label descriptions with `state={"text": text}`, as the Jev adapter sends it. The published
CPU run (supersoniclabs.ia.br/data/julia-1-cpu-20260925.json) records AG News 94/100 and
DAIR Emotion 86/100. Its Banking77 figure (60/100, three abstentions) comes from an
unpublished ranking shortlist, so Banking77 here runs the package Router over all 72 labels
and is gated on agreement with PyTorch rather than on the published count.
"""
import hashlib
import json
import random
from collections import defaultdict
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

REPOSITORY = "btzsc/btzsc"
REVISION = "fef2a2ac62b69c58670047dddf045c53d7c3cb5e"
SEED = 20260917
SAMPLES = 100
DATASETS = {
    "agnews": ("topic", "b51a89341546018960b5724596ad3192f12d0649c5e45e91a06c0cee4253510f"),
    "emotiondair": ("emotion", "779970bb63817275a9d4004e4114ed00f2d86dc7670c36e13b4d018f85dc18ae"),
    "banking77": ("intent", "f42b14bddca9455891438e5e03ad2626e9de908eac0b65dbee4d5fa9eff8a776"),
}
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
QUESTION = "Which single label best describes the input text?"
PUBLISHED = {"agnews": 94, "emotiondair": 86}
ROUTED = "banking77"
REFERENCE = Path(__file__).with_name("reference") / "pilots-torch-cpu.json"


def balanced_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)
    chosen: list[int] = []
    while len(chosen) < min(limit, len(targets)):
        progress = False
        for target in sorted(by_class):
            if by_class[target] and len(chosen) < limit:
                chosen.append(by_class[target].pop())
                progress = True
        if not progress:
            break
    return sorted(chosen)


def load_examples() -> list[dict]:
    import pyarrow.parquet as pq

    examples = []
    for offset, (name, (task, sha256)) in enumerate(DATASETS.items()):
        path = hf_hub_download(REPOSITORY, f"{name}/test-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha256:
            raise ValueError(f"BTZSC {name} does not match the pinned sha256")
        table = pq.read_table(path).to_pydict()
        texts, binary = [str(text) for text in table["text"]], [int(value) for value in table["labels"]]
        classes = next(index for index in range(1, len(texts)) if texts[index] != texts[0])
        labels = [str(table["hypothesis"][index]) for index in range(classes)]
        valid, targets = [], []
        for sample in range(len(texts) // classes):
            values = binary[sample * classes:(sample + 1) * classes]
            if sum(values) == 1:
                valid.append(sample)
                targets.append(values.index(1))
        for position in balanced_indices(targets, SAMPLES, SEED + offset):
            sample, text = valid[position], texts[valid[position] * classes]
            examples.append({"dataset": name, "task": task, "example_id": f"{name}:{sample}", "text": text,
                             "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "labels": labels, "target_index": targets[position]})
    manifest = "".join(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n" for example in examples)
    if hashlib.sha256(manifest.encode()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("pilot sample does not match the Jev protocol manifest")
    return examples


def request(example: dict) -> dict:
    return {"state": {"text": example["text"]}, "question": QUESTION, "options": example["labels"], "type": "choice"}


def gate_failures(report: dict) -> list[str]:
    failures = [f"{name} scored {report['correct'][name]}, published {count}" for name, count in PUBLISHED.items() if report["correct"][name] != count]
    return failures + agreement_failures(report, "examples")


def evaluate(engine, reference: dict) -> tuple[dict, dict]:
    from julia_mlx import Router

    examples = load_examples()
    direct = [example for example in examples if example["dataset"] != ROUTED]
    routed = [example for example in examples if example["dataset"] == ROUTED]
    logits = engine.logits([request(example) for example in direct])
    routes = [route.index for route in Router(engine).route_many([request(example) for example in routed])]
    ids = [example["example_id"] for example in direct]
    predictions = dict(zip(ids, map(argmax, logits))) | {example["example_id"]: route for example, route in zip(routed, routes)}
    correct = {name: sum(predictions[example["example_id"]] == example["target_index"] for example in examples if example["dataset"] == name) for name in DATASETS}
    same = agreement(ids, logits, reference)
    if reference:
        if reference.keys() != {example["example_id"] for example in examples}:
            raise ValueError("reference does not match the pilot examples")
        same["top1_agreement"] += sum(route == reference[example["example_id"]] for example, route in zip(routed, routes))
    outputs = dict(zip(ids, logits)) | {example["example_id"]: route for example, route in zip(routed, routes)}
    return {"correct": correct, "total": len(examples), **same}, outputs


def main() -> None:
    args = parse(parser("Julia-1 classification pilots.", REFERENCE))
    engine = load_engine(args)
    report, outputs = evaluate(engine, read_reference(args.reference))
    finish(args, {"backend": args.backend, "dtype": args.dtype, "embedding": args.embedding, **report}, outputs, gate_failures(report), "pilots")


if __name__ == "__main__":
    main()
