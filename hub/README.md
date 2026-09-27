---
license: apache-2.0
library_name: mlx
base_model: SupersonicLabs/Julia-1
base_model_relation: quantized
pipeline_tag: text-classification
tags:
  - mlx
  - apple-silicon
  - decision-model
  - text-classification
  - multilingual
  - routing
---

# Julia-1 MLX

**Julia-1 for Apple silicon.** Runs [Supersonic Labs](https://supersoniclabs.ia.br/)' [Julia-1](https://supersoniclabs.ia.br/julia-1/) decision model on the Mac's GPU with [MLX](https://github.com/ml-explore/mlx) through the [julia-mlx](https://github.com/zm2231/julia-mlx) runtime: the same answers as the official PyTorch runtime on its published evaluations, 5–14× faster on the same Mac. The upstream model repository is [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1).

The files in this repository are Supersonic Labs' Julia-1 checkpoint, unchanged (`model.safetensors` SHA-256 `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`). The runtime maps it into MLX directly, so no converted copy is needed. Precision (`dtype="float16"`) and embedding placement are load-time options rather than separate files.

This is an independent project, not affiliated with or endorsed by Supersonic Labs.

## Usage

```sh
pip install julia-mlx
```

```python
from julia_mlx import load_model

engine = load_model("zainmerchan/Julia-1-MLX", max_length=1024, head_length=512, strict_encoding=True)
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
print(result["answers"]["team"]["choice"])
```

`load_model("SupersonicLabs/Julia-1")` loads the upstream repository the same way.

## Accuracy

Supersonic's published CPU evaluation, rerun on the same pinned data and settings. FP32 is the default.

| Evaluation | Published (PyTorch CPU) | MLX FP32 | Same winner as PyTorch |
| --- | ---: | ---: | ---: |
| typed-decisions (choice / noul / score) | 1,451 / 2,000 (426 / 483 / 542) | 1,451 (426 / 483 / 542) | 2,000 / 2,000 |
| AG News pilot, 4 labels | 94 / 100 | 94 | 100 / 100 |
| DAIR Emotion pilot, 6 labels | 86 / 100 | 86 | 100 / 100 |
| Banking77 pilot, 72 labels | 60 / 100 via an unpublished shortlist | 62 via `Router` (PyTorch: 62) | 100 / 100 |

The largest FP32 logit difference from PyTorch is 0.00115. See the [runtime README](https://github.com/zm2231/julia-mlx#accuracy) for FP16 results and the evaluation protocol.

## Performance

On an Apple M4 Pro, against the reference PyTorch runtime on the same machine's CPU:

| Workload | PyTorch CPU | MLX FP32 | MLX FP16 |
| --- | ---: | ---: | ---: |
| Single call, median | 34.15 ms | 6.62 ms | 6.41 ms |
| 2,000 typed questions, batch 16 | 24.46 /s | 143.37 /s | 169.02 /s |
| 16 requests × 8,192 tokens | 83.41 s | 5.91 s | 4.97 s |

## Attribution

- **Julia-1** (weights, decision head, typed API) by [Supersonic Labs](https://supersoniclabs.ia.br/julia-1/), Apache-2.0: [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1).
- **mmBERT-small** by Johns Hopkins CLSP ([jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small)), the encoder and tokenizer Julia-1 builds on.
- **ModernBERT** by [Answer.AI and LightOn](https://github.com/AnswerDotAI/ModernBERT), the encoder architecture.
- The MLX encoder derives from [pappitti/modernbert-mlx](https://github.com/pappitti/modernbert-mlx) (MIT).

The weights keep their Apache-2.0 license. The julia-mlx runtime is MIT-licensed.
