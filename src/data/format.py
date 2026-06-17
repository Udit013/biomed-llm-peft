"""Canonical prompt format for 4-option medical MCQs.

This is the single source of truth for how a MedMCQA item becomes text. Both
the SFT training pipeline (src/train) and the lm-eval task definitions
(lm_eval_tasks/) format items identically so that the base-vs-finetuned
comparison is apples-to-apples. lm_eval_tasks/utils.py intentionally mirrors
`render_question` — if you change the template here, change it there too.
"""
from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]

SYSTEM_PROMPT = (
    "You are a medical exam assistant. Choose the single best answer to the "
    "multiple-choice question. Respond with the letter of the correct option."
)


def render_question(question: str, options: list[str]) -> str:
    """Render the user-visible MCQ stem with lettered options."""
    lines = [f"Question: {question.strip()}"]
    for letter, opt in zip(LETTERS, options):
        lines.append(f"{letter}. {str(opt).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def render_target(correct_index: int, options: list[str]) -> str:
    """The assistant target: letter + option text (e.g. 'A. metformin')."""
    return f"{LETTERS[correct_index]}. {str(options[correct_index]).strip()}"
