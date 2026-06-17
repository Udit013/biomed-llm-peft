#!/usr/bin/env python
"""Run lm-evaluation-harness on MedMCQA (in-domain) + PubMedQA (OOD).

Evaluate the base model (0-shot and 5-shot) and/or a fine-tuned adapter.

Examples:
    # base zero-shot
    python scripts/run_eval.py --mode base0 --output results/base_0shot
    # base 5-shot
    python scripts/run_eval.py --mode base5 --output results/base_5shot
    # fine-tuned adapter
    python scripts/run_eval.py --adapter outputs/qlora_5k --output results/qlora_5k

    # structural check on Mac (prints the command, runs nothing):
    python scripts/run_eval.py --mode base0 --output /tmp/x --dry-run
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.eval.harness import build_command, run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None,
                    help="Path to a trained LoRA adapter dir (omit for base).")
    ap.add_argument("--mode", choices=["base0", "base5", "adapter"], default=None,
                    help="Convenience preset for few-shot count.")
    ap.add_argument("--num-fewshot", type=int, default=None)
    ap.add_argument("--tasks", default=None,
                    help="Comma-separated task names (default: both).")
    ap.add_argument("--output", required=True, help="lm-eval output_path dir.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap items per task (smoke tests only).")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    fewshot = {"base0": 0, "base5": 5, "adapter": 0}.get(args.mode, 0)
    if args.num_fewshot is not None:
        fewshot = args.num_fewshot

    cmd = build_command(
        base_model=args.base_model,
        output_path=args.output,
        adapter_dir=args.adapter,
        num_fewshot=fewshot,
        tasks=args.tasks.split(",") if args.tasks else None,
        load_in_4bit=not args.no_4bit,
        limit=args.limit,
    )
    rc = run(cmd, dry_run=args.dry_run)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
