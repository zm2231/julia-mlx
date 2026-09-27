# Julia-1 MLX

**Julia-1 for Apple silicon.** Runs [Supersonic Labs](https://supersoniclabs.ia.br/)' [Julia-1](https://supersoniclabs.ia.br/julia-1/) decision model on the Mac's GPU with [MLX](https://github.com/ml-explore/mlx): the same answers as the official PyTorch runtime on its published evaluations, 5–14× faster on the same Mac.

Julia-1 is a 144M-parameter decision model: given context, a question, and 2–20 candidate answers, it picks one and scores every option. It handles classification, routing, ordered scales, and yes/no decisions. This package loads the original [checkpoint](https://huggingface.co/SupersonicLabs/Julia-1) unchanged: there is no conversion step and no second copy of the weights.

- **Parity.** In FP32, the scores on typed-decisions and on the AG News and DAIR Emotion pilots match Supersonic's published CPU run exactly. Every test picks the same winner as PyTorch, with logits within 0.00115.
- **Speed.** On an M4 Pro, compared with the reference runtime on the same machine: 6.62 ms per call instead of 34.15 ms, 143 decisions/s instead of 24.5 in batches, and 5.91 s instead of 83.41 s for 16 requests of 8,192 tokens.
- **Full interface.** The typed-question API, the list API, the hierarchical `Router`, the encoding checks, gradients, and export to the upstream checkpoint layout are all included.

Julia-1 MLX is an independent project. It is not affiliated with or endorsed by Supersonic Labs.

## Install

Requires an Apple silicon Mac and Python 3.11 or newer.

```sh
pip install julia-mlx
```

## Quick start

```python
from julia_mlx import load_model

engine = load_model("SupersonicLabs/Julia-1", max_length=1024, head_length=512, strict_encoding=True)
result = engine.predict(
    state="I was charged twice for the same order.",
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"},
        },
        "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Low", "Medium", "High"]},
        "refund": {"type": "noul", "instructions": "Should this customer receive a refund?", "criteria": {"false": "No refund", "true": "Refund the duplicate charge"}},
    },
)
print(result["answers"]["team"]["choice"], result["answers"]["urgency"]["score"], result["answers"]["refund"]["noul"])
```

`load_model` accepts a Hugging Face repo ID or a local checkpoint directory. For a repo ID it downloads only the files it needs (weights, configs, tokenizer) into the Hugging Face cache. `max_length=1024, head_length=512, strict_encoding=True` are the settings Supersonic evaluated. With `strict_encoding=True`, the engine raises an error rather than silently truncating an input.

The CLI scores one JSON request:

```sh
julia-mlx SupersonicLabs/Julia-1 request.json --max-length 1024
```

## API

- `engine.predict(state=..., questions=...)` is the named-question API: `choice` returns the winning key, `score` the expected level, and `noul` the probability of true, each with per-option probabilities. As in the upstream API, a `noul` question takes optional `criteria` that describe `false` and `true`, and falls back to the literal words without them. Supply descriptions when you have them: on typed-decisions, the literal words score 391/600 Boolean questions instead of 483.
- `engine.predict(rows)` and `engine.logits(rows)` take upstream-style rows (`state`, `question`, `options`, `type`). `probabilities=False` returns indices only, and `engine.encoding_info(rows)` reports how each row was encoded and whether anything was truncated.
- `Router(engine).route(row)` handles `choice` questions with up to 4,096 options. It narrows the options in groups of up to 20 and reranks the survivors, exactly as the upstream router does.

## Accuracy

The release gates rerun Supersonic's published CPU evaluation ([record](https://supersoniclabs.ia.br/data/julia-1-cpu-20260925.json)) on the same pinned data and settings. FP32 must reproduce the published counts and pick the same winner as the committed PyTorch reference on every example.

| Evaluation | Published (PyTorch CPU) | MLX FP32 | MLX FP16 | Same winner as PyTorch (FP32 / FP16) | Max logit difference (FP32 / FP16) |
| --- | ---: | ---: | ---: | ---: | ---: |
| [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (choice / noul / score) | 1,451 / 2,000 (426 / 483 / 542) | 1,451 (426 / 483 / 542) | 1,451 | 2,000 / 1,998 of 2,000 | 0.00115 / 0.46 |
| AG News pilot, 4 labels | 94 / 100 | 94 | 94 | 100 / 100 | 0.000195 / 0.132 (both pilots) |
| DAIR Emotion pilot, 6 labels | 86 / 100 | 86 | 86 | 100 / 100 | 0.000195 / 0.132 (both pilots) |
| Banking77 pilot, 72 labels | 60 / 100 via an unpublished shortlist | 62 via `Router` | 62 via `Router` | 100 / 100 | n/a (routed) |

The pilots rebuild the 100-example [BTZSC](https://huggingface.co/datasets/btzsc/btzsc) samples of the pinned [Jev benchmark protocol](https://github.com/AbdelStark/jev-benchmarks/tree/0d610cc53e79bcbec691312b0c4adb4a0e371642) and check them against its manifest hash. Supersonic's Banking77 run narrowed the 72 labels with a ranking shortlist whose procedure is not published, and abstained three times. Here, Banking77 runs the package `Router` over all 72 labels, where PyTorch also scores 62, and is gated on agreement with PyTorch. The [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) results on the model card come from a CUDA BF16 run with no published protocol, and are not reproduced.

## Performance

Apple M4 Pro (64 GB), measured with `benchmarks/suite.py`. Each configuration runs in a fresh process with the default `load_model` settings, on unique typed-decisions inputs, with encoding and token caches disabled for both runtimes. The reference is the upstream PyTorch runtime on the same machine's CPU, at its default of 4 threads (10 threads measured no faster); upstream has no Apple GPU path. Peak footprint is macOS's "peak memory footprint" from `/usr/bin/time -l`, which includes MLX's allocator cache.

| Workload | PyTorch CPU FP32 | MLX FP32, resident embedding | MLX FP32 (default) | MLX FP16 |
| --- | ---: | ---: | ---: | ---: |
| Single call, median / p95 | 34.15 / 38.97 ms | 6.65 / 6.83 ms | 6.62 / 6.80 ms | 6.41 / 6.51 ms |
| 2,000 typed questions, batch 16 | 24.46 /s | 139.73 /s | 143.37 /s | 169.02 /s |
| 16 requests × 8,192 tokens | 83.41 s | 5.87 s | 5.91 s | 4.97 s |
| Peak footprint, single calls | 699 MiB | 2,353 MiB | 1,971 MiB | 1,415 MiB |
| Peak footprint, 2,000 questions | 704 MiB | 7,027 MiB | 6,861 MiB | 5,870 MiB |
| Peak footprint, 16 × 8,192 tokens | 50,497 MiB | 8,984 MiB | 8,605 MiB | 7,360 MiB |
| Load time | 2.55 s | 0.46 s | 0.42 s | 0.41 s |

Supersonic's own measurement of the reference runtime on an Apple M4 is 33.15 ms per call ([hardware record](https://supersoniclabs.ia.br/data/julia-1-hardware-20260926.json)), in line with the PyTorch column.

How the speed and memory are obtained:

- **Attention.** Julia-1 keeps mmBERT-small's attention layout: every third layer attends globally, and the other 14 attend within ±64 tokens (`local_attention=128`). Upstream evaluates those local layers densely against a full `tokens × tokens` mask. Beyond 512 tokens, this runtime computes the same attention exactly in 64-token blocks against their neighbors, so local-layer cost grows linearly with length. Below that, a dense boolean mask is faster.
- **Embedding.** The 256k-row vocabulary embedding is 98M of the 144M parameters, and a request reads only the rows for its own tokens. By default (`embedding="mapped"`), the table stays a read-only memory map of `model.safetensors`. Each forward gathers its unique rows on the CPU (about 0.1 ms) and sends only those to the GPU, so MLX holds 175 MiB of weights instead of 550 MiB. The rows are the checkpoint's own FP32 bytes, so results are identical to `embedding="resident"`, which loads the whole table into MLX.
- **Precision.** `dtype="float16"` runs the encoder projections in half precision. The residual stream, norms, embeddings, and decision head stay FP32: from the middle layers on, Julia-1's residual stream carries outlier channels near 4,000, where FP16 cannot represent small updates. FP32 is the parity mode.
- **Allocator cache.** MLX keeps freed buffers for reuse, and varied batch shapes would otherwise grow that cache without bound (26.0 GiB over 2,000 typed questions). `load_model(memory_cache_limit="auto")` caps it at the smaller of 4 GiB and one eighth of the GPU's recommended working set; pass a byte count to override it, or `None` for MLX's default. Batches pad to multiples of 32 tokens so buffers get reused. `engine.set_memory_cache_limit`, `engine.memory_stats`, and `engine.trim_memory` adjust the cache at runtime.

| Cache cap | Single call | 2,000 questions | 16 × 8k tokens | Cache held after batch run |
| --- | ---: | ---: | ---: | ---: |
| 256 MiB | 8.50 ms | 134.8 /s | 6.77 s | 269 MiB |
| 1 GiB | 6.77 ms | 139.6 /s | 6.56 s | 1.0 GiB |
| 4 GiB | 6.77 ms | 143.4 /s | 6.04 s | 4.0 GiB |
| none | 6.77 ms | 145.6 /s | 6.02 s | 26.0 GiB |

PyTorch's footprint on short inputs is lower because it keeps no allocator cache. Much of MLX's batch footprint is that cache: a 1 GiB cap keeps 3 GiB less memory reserved than the default 4 GiB, for about 3% lower batch throughput and 9% slower long inputs. For a server taking single requests, two threads that each tokenize and score their own requests on their own stream (`mx.stream(mx.new_stream(mx.gpu))`) raised throughput from 120.2 to 163.7 requests/s, with outputs identical to serial calls, while median latency rose from 6.73 ms to 9.31 ms. Four threads reached 172.4 requests/s at a 17.86 ms median, and eight added nothing further.

## Fine-tuning and export

The MLX model is differentiable, and its gradients match PyTorch's. `model.train()` enables the decision head's 0.1 dropout, including attention-weight dropout, exactly where PyTorch's `TransformerEncoderLayer` applies it. The unused `temperature` buffer is frozen. Load with `embedding="resident"` to train the vocabulary embedding. `save_model(engine, directory)` writes an FP32 checkpoint in the upstream layout, which `julia.inference.load_model` loads directly.

## Verification

```sh
git clone https://github.com/zm2231/julia-mlx && cd julia-mlx
pip install -e ".[eval,test]"
hf download SupersonicLabs/Julia-1 --local-dir Julia-1
JULIA_CHECKPOINT=Julia-1 PYTHONPATH=Julia-1 pytest -q tests
python evals/typed_decisions.py SupersonicLabs/Julia-1
python evals/pilots.py SupersonicLabs/Julia-1
PYTHONPATH=Julia-1 python benchmarks/suite.py Julia-1
```

`PYTHONPATH=Julia-1` makes the upstream PyTorch runtime importable. Without it, the PyTorch comparisons in the tests are skipped, and both gates still run against the committed PyTorch reference outputs in `evals/reference/` (regenerate them with `--backend torch --write-reference PATH`). The gates' logit tolerance is 2e-3: FP32 MLX differs from PyTorch CPU by at most 0.00115 on the 2,000 typed-decisions questions and by at most 0.000195 on the AG News and Emotion pilots. The synthetic-input parity tests keep a 2e-4 bound.

## Attribution

- **Julia-1**, including its weights, decision head, marker serialization, typed API, and router, is by [Supersonic Labs](https://supersoniclabs.ia.br/julia-1/), released under Apache-2.0 at [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1). This repository contains no Julia-1 weights; `load_model` fetches them from that repository.
- **mmBERT-small** ([jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), Johns Hopkins CLSP) is the multilingual encoder and tokenizer that Julia-1 builds on.
- **ModernBERT** ([Answer.AI and LightOn](https://github.com/AnswerDotAI/ModernBERT)) is the encoder architecture.
- The MLX ModernBERT encoder derives from the MIT-licensed [pappitti/modernbert-mlx](https://github.com/pappitti/modernbert-mlx), which in turn builds on Apple's MIT-licensed [mlx-examples](https://github.com/ml-explore/mlx-examples).
- **Evaluation data**: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions), [BTZSC](https://huggingface.co/datasets/btzsc/btzsc), and the [Jev benchmark protocol](https://github.com/AbdelStark/jev-benchmarks) by AbdelStark.

## License

The code in this repository is MIT-licensed (see [LICENCE](LICENCE)). The Julia-1 weights are Apache-2.0 and remain in the upstream model repository under their own license.
