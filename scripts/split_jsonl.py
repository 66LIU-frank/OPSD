#!/usr/bin/env python
"""Deterministically split a JSONL file into train/eval JSONL files."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--eval-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = [line for line in args.input_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    indices = list(range(len(lines)))
    random.Random(args.seed).shuffle(indices)

    eval_count = min(max(args.eval_size, 0), len(lines))
    eval_indices = set(indices[:eval_count])

    args.train_file.parent.mkdir(parents=True, exist_ok=True)
    args.eval_file.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    with args.train_file.open("w", encoding="utf-8") as train_f, args.eval_file.open(
        "w", encoding="utf-8"
    ) as eval_f:
        for idx, line in enumerate(lines):
            if idx in eval_indices:
                eval_f.write(line + "\n")
            else:
                train_f.write(line + "\n")
                train_count += 1

    print(
        f"Split {len(lines)} rows -> train={train_count}, eval={eval_count}, "
        f"seed={args.seed}"
    )


if __name__ == "__main__":
    main()
