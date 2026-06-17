"""Thin wrapper around the EleutherAI lm-evaluation-harness CLI.

All MedMCQA / PubMedQA scoring goes through `lm_eval` — we never hand-roll a
scoring loop. Base and fine-tuned models are evaluated under the SAME 4-bit
quantization and the SAME chat-templated prompt so the comparison is fair.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

# Mirror of src.data.format.SYSTEM_PROMPT (kept local so eval has no src import).
SYSTEM_PROMPT = (
    "You are a medical exam assistant. Choose the single best answer to the "
    "multiple-choice question. Respond with the letter of the correct option."
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "lm_eval_tasks"
DEFAULT_TASKS = ["medmcqa_val", "pubmedqa_ood"]


def build_command(
    base_model: str,
    output_path: str | Path,
    adapter_dir: str | None = None,
    num_fewshot: int = 0,
    tasks: list[str] | None = None,
    load_in_4bit: bool = True,
    batch_size: str = "auto",
    limit: int | None = None,
    apply_chat_template: bool = True,
) -> list[str]:
    """Construct the lm_eval argv. `adapter_dir` set => evaluate base+LoRA."""
    tasks = tasks or DEFAULT_TASKS
    model_args = [f"pretrained={base_model}"]
    if load_in_4bit:
        model_args.append("load_in_4bit=True")
    if adapter_dir:
        model_args.append(f"peft={adapter_dir}")

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", ",".join(model_args),
        "--tasks", ",".join(tasks),
        "--include_path", str(TASKS_DIR),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
        "--output_path", str(output_path),
        "--log_samples",
        "--seed", "42",
    ]
    if apply_chat_template:
        cmd += ["--apply_chat_template", "--system_instruction", SYSTEM_PROMPT]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def run(cmd: list[str], dry_run: bool = False) -> int:
    """Print and (unless dry_run) execute an lm_eval command."""
    print("[eval] " + " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode
