#!/usr/bin/env python
"""Push a trained LoRA adapter + generated model card to the Hugging Face Hub.

Degrades gracefully: with no HF_TOKEN it writes the model card locally and tells
you what it WOULD push, without failing.

Usage:
    python scripts/push_to_hub.py --adapter outputs/qlora_5k \
        --repo-id your-username/qwen2.5-7b-medmcqa-qlora-5k
"""
from __future__ import annotations

import argparse
import os

import _bootstrap  # noqa: F401
from src.eval.error_analysis import compare, render_markdown
from src.eval.results import render_table
from src.serve.cost import render_cost_table
from src.utils.model_card import write_model_card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--repo-id", default=os.environ.get("HF_HUB_REPO_ID"))
    ap.add_argument("--base-eval", default="results/base_0shot",
                    help="Base lm-eval dir for the error-analysis comparison.")
    ap.add_argument("--ft-eval", default=None,
                    help="Fine-tuned lm-eval dir for the error-analysis comparison.")
    args = ap.parse_args()

    if not args.repo_id:
        raise SystemExit("Set --repo-id or HF_HUB_REPO_ID.")

    ea_md = render_markdown(compare(args.base_eval, args.ft_eval)) if args.ft_eval \
        else "**PENDING RUN** — run the fine-tuned eval first."

    card_path = write_model_card(
        adapter_dir=args.adapter,
        base_model=args.base_model,
        headline_table=render_table(),
        cost_table=render_cost_table(),
        error_analysis_md=ea_md,
        hub_repo_id=args.repo_id,
    )
    print(f"[hub] wrote model card -> {card_path}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hub] No HF_TOKEN set; skipping upload. Model card written locally.")
        print(f"[hub] Would push '{args.adapter}' -> '{args.repo_id}'.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=args.adapter, repo_id=args.repo_id, repo_type="model")
    print(f"[hub] pushed '{args.adapter}' -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
