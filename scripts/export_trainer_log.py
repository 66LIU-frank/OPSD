#!/usr/bin/env python
"""Export Hugging Face trainer_state.json log_history to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    rows = state.get("log_history", [])
    if not rows:
        raise ValueError(f"No log_history found in {args.trainer_state}")

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} trainer log rows to {args.output_csv}")


if __name__ == "__main__":
    main()
