#!/usr/bin/env python
"""Per-subject MedMCQA error analysis: base vs fine-tuned.

Reads lm-eval `--log_samples` output from two run dirs and writes a markdown
report (+ JSON) classifying each subject as improved / neutral / worsened.

Usage:
    python scripts/error_analysis.py \
        --base results/base_0shot --finetuned results/qlora_5k \
        --out results/error_analysis_5k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.eval.error_analysis import compare, render_markdown


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="Base-model lm-eval output dir.")
    ap.add_argument("--finetuned", required=True, help="Fine-tuned lm-eval output dir.")
    ap.add_argument("--out", required=True, help="Output prefix (writes .md and .json).")
    args = ap.parse_args()

    result = compare(args.base, args.finetuned)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(result, indent=2))
    md = render_markdown(result)
    out.with_suffix(".md").write_text(md)
    print(md)
    print(f"[error-analysis] wrote {out.with_suffix('.md')} and "
          f"{out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
