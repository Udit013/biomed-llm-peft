#!/usr/bin/env python
"""Render the headline data-scaling results table from lm-eval outputs.

Missing runs render as PENDING RUN. Usage:
    python scripts/results_table.py            # print markdown
    python scripts/results_table.py --out results/headline_table.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.eval.results import render_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    table = render_table()
    print(table)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table)
        print(f"[results] wrote {args.out}")


if __name__ == "__main__":
    main()
