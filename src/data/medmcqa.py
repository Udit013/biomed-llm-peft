"""MedMCQA loading, leakage-safe splitting, subsampling, and SFT formatting.

Dataset: openlifescienceai/medmcqa (~194K 4-option medical MCQs).
Split provenance (stated explicitly to be unambiguous):
  * `train`        -> used for TRAINING only. We subsample the first N items
                      under a fixed seed; a disjoint `val_size` tail is held out
                      purely for eval-LOSS during training.
  * `validation`   -> 4183 labeled items, used for SCORING (in-domain accuracy).
                      MedMCQA's official `test` split has hidden labels (cop=-1),
                      so it cannot be scored locally; we do not use it.
There is no overlap between the trained-on subset and the scored validation set.
"""
from __future__ import annotations

from typing import Any

from .format import SYSTEM_PROMPT, render_question, render_target

DATASET_PATH = "openlifescienceai/medmcqa"
OPTION_KEYS = ["opa", "opb", "opc", "opd"]


def _options(row: dict[str, Any]) -> list[str]:
    return [row[k] for k in OPTION_KEYS]


def _is_valid_train(row: dict[str, Any]) -> bool:
    """Keep single-answer items with a valid correct-option index 0..3."""
    cop = row.get("cop", -1)
    return isinstance(cop, int) and 0 <= cop <= 3


def load_train_val(train_size: int, val_size: int, seed: int):
    """Return (train_ds, val_ds) HF Datasets, formatted as chat SFT examples.

    train_size == -1 uses the full filtered train split. The val slice is a
    disjoint tail of the shuffled train split (held-out for eval loss only).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_PATH, split="train")
    ds = ds.filter(_is_valid_train)
    ds = ds.shuffle(seed=seed)

    val = ds.select(range(val_size))
    rest = ds.select(range(val_size, len(ds)))
    if train_size == -1 or train_size >= len(rest):
        train = rest
    else:
        train = rest.select(range(train_size))

    train = train.map(_to_sft, remove_columns=train.column_names)
    val = val.map(_to_sft, remove_columns=val.column_names)
    return train, val


def _to_sft(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw MedMCQA row into a `messages` chat example for TRL SFT."""
    options = _options(row)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_question(row["question"], options)},
            {"role": "assistant", "content": render_target(row["cop"], options)},
        ],
        "subject_name": row.get("subject_name", "unknown"),
    }


def load_scoring_split():
    """Raw `validation` split for reference/inspection (lm-eval loads its own)."""
    from datasets import load_dataset

    return load_dataset(DATASET_PATH, split="validation")
