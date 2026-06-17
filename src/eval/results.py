"""Aggregate lm-eval results JSON into the headline data-scaling table.

Any row whose results file is absent renders as `PENDING RUN` — we never
fabricate a number. Fill the table only after the real eval has run.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

TASK_COLUMNS = [("medmcqa_val", "MedMCQA"), ("pubmedqa_ood", "PubMedQA")]

# (row label, lm-eval output dir). Edit/extend as runs complete.
DEFAULT_ROWS = [
    ("Base 0-shot", "results/base_0shot"),
    ("Base 5-shot", "results/base_5shot"),
    ("QLoRA 5K", "results/qlora_5k"),
    ("QLoRA 20K", "results/qlora_20k"),
    ("QLoRA 50K", "results/qlora_50k"),
]


def _find_results_file(run_dir: str | Path) -> Path | None:
    matches = glob.glob(str(Path(run_dir) / "**" / "results_*.json"), recursive=True)
    if not matches:
        # lm-eval sometimes writes results.json directly
        direct = list(Path(run_dir).glob("**/results.json")) if Path(run_dir).exists() else []
        matches = [str(p) for p in direct]
    return Path(max(matches, key=lambda p: Path(p).stat().st_mtime)) if matches else None


def read_accuracies(run_dir: str | Path) -> dict[str, float] | None:
    """Return {task: acc_percent} for a run dir, or None if not yet run."""
    f = _find_results_file(run_dir)
    if f is None:
        return None
    data = json.loads(f.read_text())
    res = data.get("results", {})
    out: dict[str, float] = {}
    for task, _label in TASK_COLUMNS:
        if task in res and "acc,none" in res[task]:
            out[task] = round(100 * res[task]["acc,none"], 2)
        elif task in res and "acc" in res[task]:
            out[task] = round(100 * res[task]["acc"], 2)
    return out


def render_table(rows=DEFAULT_ROWS) -> str:
    header = "| Model | " + " | ".join(label for _, label in TASK_COLUMNS) + " |"
    sep = "|---|" + "---|" * len(TASK_COLUMNS)
    lines = [header, sep]
    for label, run_dir in rows:
        accs = read_accuracies(run_dir)
        cells = []
        for task, _ in TASK_COLUMNS:
            if accs and task in accs:
                cells.append(f"{accs[task]:.2f}")
            else:
                cells.append("PENDING RUN")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
