"""Retrieval agent: fetch + rerank evidence via the RAG pipeline.

Skipped for non-RAG configs (base / ft), which answer from parametric knowledge.
"""
from __future__ import annotations

from .state import AgentState, Deps


def retrieve(state: AgentState, deps: Deps) -> AgentState:
    if not deps.use_rag or not state.get("plan", {}).get("needs_retrieval", True):
        return {"passages": [], "citations": [], "latency_ms": dict(state.get("latency_ms", {}))}

    passages, citations, lat = deps.pipeline.retrieve_context(
        state["query"], metadata_filter=state.get("metadata_filter"))
    merged = {**state.get("latency_ms", {}), **lat}
    return {"passages": passages, "citations": citations, "latency_ms": merged}
