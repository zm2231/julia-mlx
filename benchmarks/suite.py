"""Run every Julia-1 benchmark sequentially, one fresh process each, and print a Markdown report.

Peak memory is macOS `/usr/bin/time -l` "peak memory footprint", which includes
Metal allocations and excludes clean memory-mapped file pages.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).with_name("bench.py")
CONFIGS = [
    ("PyTorch CPU fp32", ["--backend", "torch"]),
    ("MLX fp32, resident embedding", ["--embedding", "resident"]),
    ("MLX fp32, mapped embedding", []),
    ("MLX fp16, mapped embedding", ["--dtype", "float16"]),
]
COLUMNS = {
    "single": ("Single call, 400 unique typed questions", ["median_ms", "p95_ms", "p99_ms", "peak_footprint_mib", "load_s"]),
    "batch": ("2,000 typed questions, batch 16, no caches", ["decisions_per_s", "median_s", "max_s", "peak_footprint_mib"]),
    "long": ("16 requests near 8,192 tokens", ["seconds", "tokens_per_row", "peak_footprint_mib"]),
}


def measure(checkpoint: str, workload: str, flags: list[str]) -> dict:
    process = subprocess.run(["/usr/bin/time", "-l", sys.executable, str(BENCH), checkpoint, workload, *flags],
                             capture_output=True, text=True, check=True)
    footprint = re.search(r"(\d+)\s+peak memory footprint", process.stderr)
    if footprint is None:
        raise RuntimeError("/usr/bin/time -l did not report a peak memory footprint")
    return {**json.loads(process.stdout.strip().splitlines()[-1]), "peak_footprint_mib": int(footprint.group(1)) // 2**20}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--workloads", nargs="+", choices=list(COLUMNS), default=list(COLUMNS))
    args = parser.parse_args()
    for workload in args.workloads:
        title, keys = COLUMNS[workload]
        print(f"\n### {title}\n\n| Backend | " + " | ".join(keys) + " |\n| --- |" + " ---: |" * len(keys), flush=True)
        for name, flags in CONFIGS:
            row = measure(args.checkpoint, workload, flags)
            print(f"| {name} | " + " | ".join(str(row[key]) for key in keys) + " |", flush=True)


if __name__ == "__main__":
    main()
