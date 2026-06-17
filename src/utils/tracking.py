"""Weights & Biases integration that degrades gracefully.

If no WANDB_API_KEY is set (or wandb isn't installed, or WANDB_MODE=disabled),
training MUST continue with local logging and never block. This module decides
once, up front, whether W&B is usable and returns the right `report_to` value
for the HF Trainer plus a local JSONL logger as a fallback/supplement.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def wandb_enabled() -> bool:
    """True only if a key is present, wandb importable, and not disabled."""
    if os.environ.get("WANDB_MODE", "").lower() in {"disabled", "offline-skip"}:
        return False
    if not os.environ.get("WANDB_API_KEY"):
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    return True


def report_to() -> list[str]:
    """Value for TrainingArguments.report_to. Empty list => local logging only."""
    return ["wandb"] if wandb_enabled() else []


class LocalLogger:
    """Always-on JSONL logger so runs are recorded even without W&B."""

    def __init__(self, output_dir: str | Path):
        self.path = Path(output_dir) / "local_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"ts": time.time(), **record}
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")


def init_wandb(run_name: str, config: dict) -> None:
    """Init a W&B run if enabled; otherwise print why we're skipping. Never raises."""
    if not wandb_enabled():
        reason = (
            "no WANDB_API_KEY" if not os.environ.get("WANDB_API_KEY") else "wandb disabled"
        )
        print(f"[tracking] W&B off ({reason}); using local JSONL logging only.")
        return
    try:
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "biomed-llm-peft"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=run_name,
            config=config,
        )
        print(f"[tracking] W&B run started: {run_name}")
    except Exception as exc:  # never let tracking kill a run
        print(f"[tracking] W&B init failed ({exc}); continuing with local logging.")
