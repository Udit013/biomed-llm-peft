"""Formatting helpers referenced by the lm-eval task YAMLs via `!function`.

The MedMCQA rendering here MUST stay in sync with src/data/format.py so that
training and evaluation see identical prompts. (lm-eval loads task include-paths
in isolation, so we mirror the few lines rather than import across trees.)
"""
from __future__ import annotations

LETTERS = ["A", "B", "C", "D"]


# ----------------------------- MedMCQA (in-domain) -----------------------------
def medmcqa_doc_to_text(doc) -> str:
    opts = [doc["opa"], doc["opb"], doc["opc"], doc["opd"]]
    lines = [f"Question: {doc['question'].strip()}"]
    for letter, opt in zip(LETTERS, opts):
        lines.append(f"{letter}. {str(opt).strip()}")
    lines.append("Answer:")
    return "\n".join(lines)


def medmcqa_doc_to_choice(doc) -> list[str]:
    # Score the full option texts (canonical MedMCQA multiple_choice setup).
    return [
        f"{LETTERS[i]}. {str(doc[k]).strip()}"
        for i, k in enumerate(["opa", "opb", "opc", "opd"])
    ]


# ----------------------------- PubMedQA (OOD) ----------------------------------
_PUBMEDQA_LABELS = ["yes", "no", "maybe"]


def pubmedqa_doc_to_text(doc) -> str:
    ctx = doc.get("context", {})
    contexts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
    abstract = " ".join(contexts).strip()
    return (
        f"Abstract: {abstract}\n"
        f"Question: {doc['question'].strip()}\n"
        f"Answer (yes/no/maybe):"
    )


def pubmedqa_doc_to_choice(doc) -> list[str]:
    return [" yes", " no", " maybe"]


def pubmedqa_doc_to_target(doc) -> int:
    return _PUBMEDQA_LABELS.index(doc["final_decision"].strip().lower())
