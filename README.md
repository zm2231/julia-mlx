# Julia-1 on MLX

Apple-silicon inference for [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1). It keeps Julia's marker serialization and maps the original PyTorch safetensors checkpoint into an MLX ModernBERT encoder plus Julia decision head.

## Status

The supported path is finite-choice inference: `state`, `question`, 2–20 options, and a `choice`, `score`, or `noul` type, plus the hierarchical `Router` for larger choice sets. The release gates are the published Julia-1 CPU evaluation ([record](https://supersoniclabs.ia.br/data/julia-1-cpu-20260925.json)), run on the same pinned data and protocol. FP32 MLX must reproduce the published counts and pick the same winner as PyTorch on every example:

| Evaluation | Published (PyTorch CPU) | MLX FP32 | MLX FP16 | Top-1 agreement with PyTorch (FP32 / FP16) | Max logit difference (FP32 / FP16) |
| --- | ---: | ---: | ---: | ---: | ---: |
| typed-decisions (choice / noul / score) | 1,451 / 2,000 (426 / 483 / 542) | 1,451 (426 / 483 / 542) | 1,451 | 2,000 / 1,998 of 2,000 | 0.00115 / 0.46 |
| AG News pilot, 4 labels | 94 / 100 | 94 | 94 | 100 / 100 | 0.000195 / 0.132 (both pilots) |
| DAIR Emotion pilot, 6 labels | 86 / 100 | 86 | 86 | 100 / 100 | 0.000195 / 0.132 (both pilots) |
| Banking77 pilot, 72 labels | 60 / 100 via an unpublished shortlist | 62 via `Router` | 62 via `Router` | 100 / 100 | — |

The pilots rebuild the 100-example samples of the pinned [Jev protocol](https://github.com/AbdelStark/jev-benchmarks/tree/0d610cc53e79bcbec691312b0c4adb4a0e371642) and check them against its manifest hash. The published Banking77 run narrows the 72 labels with a ranking shortlist whose procedure is not published (and abstains three times), so here Banking77 runs the package `Router` over all 72 labels, where PyTorch also scores 62, and is gated on agreement with PyTorch. The MASSIVE scenario results on the model card came from a CUDA BF16 run with no published protocol and are not reproduced.

FP32 is the parity mode. FP16 runs the encoder projections in half precision while the residual stream, norms, embeddings, and decision head stay FP32: from the middle layers on, Julia-1's residual stream carries outlier channels near 4,000, where FP16 cannot represent small updates. The 256k-row vocabulary embedding is 98M of the 144M parameters, and a request reads only the rows of its own tokens. By default (`embedding="mapped"`) it stays a read-only memory map of `model.safetensors`: each forward gathers its unique token rows on the CPU (about 0.1 ms for a typical request) and sends only those to the GPU, so MLX holds 175 MiB of weights instead of 550 MiB. The rows are the checkpoint's own FP32 bytes, so results are identical to `embedding="resident"`, which loads the table into MLX and is required to fine-tune the embedding.

## Install

```sh
python -m pip install -e .
python -c "from julia_mlx import download_model; download_model('Julia-1')"
```

## Use

```python
from julia_mlx import load_model

engine = load_model("Julia-1", max_length=1024, head_length=512, strict_encoding=True)
result = engine.predict(
    state="I was charged twice for the same order.",
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"},
        },
    },
)
print(result["answers"]["team"])
```

The legacy list API (`engine.predict(rows)`, `engine.logits(rows)`), `probabilities=False`, `encoding_info`, and `Router` follow the upstream runtime. As upstream, a `noul` question takes optional `criteria` mapping `false` and `true` to descriptions and falls back to the literal options `false` and `true` without them. Supply the descriptions when you have them: on typed-decisions, literal options score 391/600 Boolean questions instead of 483.

`load_model` bounds MLX's process-wide allocator cache (`memory_cache_limit="auto"`: the smaller of 4 GiB and one eighth of the GPU's recommended working set; pass bytes to override or `None` to keep MLX's default). MLX keeps freed buffers for reuse, keyed by size, and varied batch shapes otherwise grow the cache without bound: 24.6 GiB on 2,000 typed questions. Measured on an M4 Pro (64 GB):

| Cache cap | Single call | 2,000 questions | 16 × 8k tokens | Cache held after batch run |
| --- | ---: | ---: | ---: | ---: |
| 256 MiB | 8.26 ms | 128.1 /s | 6.92 s | 256 MiB |
| 1 GiB | 6.90 ms | 133.8 /s | 6.71 s | 1.0 GiB |
| 4 GiB | 6.89 ms | 140.0 /s | 6.14 s | 4.0 GiB |
| none | 6.75 ms | 140.9 /s | 6.13 s | 24.6 GiB |

`engine.set_memory_cache_limit(bytes)`, `engine.memory_stats()`, and `engine.trim_memory()` adjust it at runtime.

The CLI accepts a JSON file containing one request:

```sh
julia-mlx Julia-1 request.json --max-length 1024
```

## Fine-tuning and export

The MLX model is differentiable and matches PyTorch gradients. `model.train()` enables the decision head's 0.1 dropout (including attention-weight dropout) exactly where PyTorch's `TransformerEncoderLayer` applies it; the unused `temperature` buffer is frozen. `save_model(engine, directory)` writes an FP32 checkpoint in the upstream layout, which `julia.inference.load_model` loads directly.

## Performance

Julia-1 keeps mmBERT-small's attention layout: every third layer attends globally and the other 14 attend within ±64 tokens (`local_attention=128`). Upstream (Transformers ModernBERT with SDPA) evaluates those local layers densely against a full `tokens × tokens` mask. Beyond 512 tokens this runtime computes the same attention in 64-token query blocks against their three neighboring key blocks, so local-layer cost grows linearly with length; below that, a dense boolean mask is faster. Batches pad to a multiple of 32 tokens so MLX's allocator can reuse buffers across calls.

Apple M4 Pro (64 GB), `benchmarks/suite.py`: every configuration in a fresh process with the default `load_model` settings (auto cache cap, 4 GiB on this machine), unique typed-decisions inputs, encoding and token caches disabled for both runtimes, PyTorch at the upstream default of 4 CPU threads (10 threads measured no faster). Peak footprint is macOS `/usr/bin/time -l` "peak memory footprint", which includes the allocator cache.

| Workload | PyTorch CPU FP32 | MLX FP32, resident embedding | MLX FP32 (default) | MLX FP16 |
| --- | ---: | ---: | ---: | ---: |
| Single call, median / p95 | 34.3 / 40.6 ms | 6.84 / 6.98 ms | 6.89 / 7.04 ms | 6.56 / 6.65 ms |
| 2,000 typed questions, batch 16 | 24.0 /s | 129.9 /s | 136.6 /s | 163.8 /s |
| 16 requests × 8,192 tokens | 81.1 s | 5.98 s | 6.07 s | 5.11 s |
| Peak footprint, single calls | 699 MiB | 2,333 MiB | 1,972 MiB | 1,416 MiB |
| Peak footprint, 2,000 questions | 712 MiB | 7,045 MiB | 6,862 MiB | 5,871 MiB |
| Peak footprint, 16 × 8,192 tokens | 50,500 MiB | 9,133 MiB | 8,569 MiB | 7,369 MiB |
| Load time | 2.75 s | 0.50 s | 0.46 s | 0.44 s |

PyTorch's footprint on short inputs is lower because it keeps no allocator cache and memory-maps the checkpoint. Much of MLX's batch footprint is the cache: with a 1 GiB cap, the resident-embedding FP32 run of 2,000 questions peaked at 3,971 MiB instead of 7,045 MiB, at about 5% lower throughput (see the cache table above).

For a server taking single requests, two threads that each tokenize and score their own requests under their own `mx.stream(mx.new_stream(mx.gpu))` raised throughput from 117 to 149 requests/s on this machine, with outputs identical to serial calls; median latency rose from 6.95 ms to 9.89 ms, and four or more threads added no throughput.

Reproduce on a Mac with the upstream Julia source importable:

```sh
PYTHONPATH=Julia-1 python benchmarks/suite.py Julia-1
```

## Verification

```sh
python -m pip install -e ".[eval,test]"
JULIA_CHECKPOINT=Julia-1 PYTHONPATH=Julia-1 pytest -q tests
python evals/typed_decisions.py Julia-1
python evals/pilots.py Julia-1
```

Without `PYTHONPATH` pointing at the upstream source, the PyTorch comparisons skip and both gates still run against the committed PyTorch reference outputs in `evals/reference/` (regenerate with `--backend torch --write-reference PATH`). The gates' logit tolerance is 2e-3: FP32 MLX differs from PyTorch CPU by at most 0.00115 on the 2,000 typed-decisions questions and 0.000195 on the AG News and Emotion pilots, while the synthetic-input parity tests keep their 2e-4 bound.

## Attribution

The ModernBERT MLX encoder derives from the MIT-licensed [pappitti/modernbert-mlx](https://github.com/pappitti/modernbert-mlx). Julia-1 weights and their contract are Apache-2.0 in the upstream model repository.
