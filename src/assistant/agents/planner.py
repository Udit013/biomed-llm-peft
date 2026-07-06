"""Planner agent: decide the retrieval strategy from the query.

Deterministic heuristic (no LLM call, so it's fast, free, and testable): detect
source hints (CDC/WHO/NIH/"guideline") and recency hints, and produce a metadata
filter for the retriever. Whether retrieval runs at all is set by the config
(base/ft skip RAG; base_rag/ft_rag use it).
"""
from __future__ import annotations

from .state import AgentState, Deps

_SRC_KEYWORDS = {"cdc": "cdc", "who": "who", "nih": "nih"}
_RECENCY = ("recent", "latest", "current", "up to date", "up-to-date",
            "2021", "2022", "2023", "2024", "2025")


def plan(state: AgentState, deps: Deps) -> AgentState:
    query = state["query"]
    ql = query.lower()

    sources = [s for kw, s in _SRC_KEYWORDS.items() if kw in ql]
    if "guideline" in ql and not sources:
        sources = ["nih", "who", "cdc"]

    metadata_filter: dict = {}
    if sources:
        metadata_filter["source"] = sources
    if any(w in ql for w in _RECENCY):
        metadata_filter["year_min"] = 2018

    strategy = "filtered" if metadata_filter else "semantic"
    return {
        "plan": {"needs_retrieval": deps.use_rag, "strategy": strategy},
        "metadata_filter": metadata_filter or None,
    }
