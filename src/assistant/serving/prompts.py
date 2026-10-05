"""Prompt templates for grounded biomedical answering.

The context passages are numbered [1], [2], ... and the model is instructed to
cite each claim with the matching marker — which is exactly what the
citation-verification agent checks against.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. Answer the question using "
    "ONLY the information in the provided numbered sources. Give every specific "
    "fact the sources DO support, even if they answer the question only partly, "
    "and then briefly note what they leave out. Cite each factual claim with a "
    "single marker per source, like [1] or [2][3] — never ranges like [1-3]. "
    "Only if no source is relevant at all, reply exactly: 'The provided sources do "
    "not contain enough evidence.' Do not invent citations or facts. This is "
    "research information, not medical advice."
)

SYSTEM_PROMPT_NO_RAG = (
    "You are a biomedical research assistant. Answer the question concisely and "
    "factually. This is research information, not medical advice."
)


def build_answer_messages(query: str, context: str | None = None) -> list[dict]:
    """Chat messages for the answer agent. `context=None` => parametric (no RAG)."""
    if context:
        user = (
            f"Sources:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the sources above, and cite each claim with [n]."
        )
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return [{"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
            {"role": "user", "content": f"Question: {query}"}]
