"""Citation-verification agent: flag answer claims unsupported by the evidence.

Uses the lexical grounding check (rag/citations.verify_claims) to score each
answer sentence against the retrieved passages, then marks each citation
supported/unsupported and sets an overall `all_supported` flag. For non-RAG
configs there is no evidence to verify against, so verification is skipped.
"""
from __future__ import annotations

from ..rag.citations import verify_claims
from .state import AgentState, Deps


def verify(state: AgentState, deps: Deps) -> AgentState:
    passages = state.get("passages", [])
    answer_text = state.get("answer", "")
    if not passages or not answer_text:
        return {"claims": [], "all_supported": None}

    claims, all_ok = verify_claims(answer_text, passages)

    # Mark each citation supported if any supported claim maps to its passage.
    supported_passages = {c["passage_index"] for c in claims
                          if c["supported"] and c["passage_index"] >= 0}
    citations = state.get("citations", [])
    for i, cit in enumerate(citations):
        cit.supported = i in supported_passages
        best = max((c["support_score"] for c in claims if c["passage_index"] == i),
                   default=0.0)
        cit.support_score = best

    return {"claims": claims, "all_supported": all_ok, "citations": citations}
