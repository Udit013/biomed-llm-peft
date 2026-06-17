#!/usr/bin/env python
"""Train a QLoRA adapter on MedMCQA at a given N (data-scaling sweep point).

Usage:
    python scripts/train.py --config configs/qlora_5k.yaml

Resumable: re-run the same command after a disconnect to continue from the
latest checkpoint in the config's output_dir.
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (sys.path + .env)
from src.train.sft import train
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a YAML run config.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[train] config: {args.config}  run_name={cfg.run_name}  "
          f"N={cfg.data.train_size}  seed={cfg.seed}")
    train(cfg)


if __name__ == "__main__":
    main()
