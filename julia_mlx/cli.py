import argparse
import json
from pathlib import Path

from .checkpoint import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Julia-1 through MLX.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    args = parser.parse_args()
    engine = load_model(args.checkpoint, max_length=args.max_length, dtype=args.dtype)
    print(json.dumps(engine.predict([json.loads(args.request.read_text())])[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
