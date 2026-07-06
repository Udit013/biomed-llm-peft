"""Shared agent state (LangGraph-compatible TypedDict) and node dependencies.

Each agent node is a pure function `(state, deps) -> partial_state`. Keeping them
framework-agnostic means they run identically whether orchestrated by LangGraph
(production) or a plain sequential loop (tests / no-langgraph environments), and
they're unit-testable without any graph machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from ..rag.pipeline import RAGPipeline
    from ..serving.providers import LLMProvider
    from ..schema import Citation, RetrievedPassage


class AgentState(TypedDict, total=False):
    query: str
    config_label: str                 # base | ft | base_rag | ft_rag
    plan: dict[str, Any]
    metadata_filter: dict[str, Any] | None
    passages: list["RetrievedPassage"]
    citations: list["Citation"]
    answer: str
    claims: list[dict[str, Any]]
    all_supported: bool | None
    latency_ms: dict[str, float]
    token_usage: dict[str, int]


@dataclass
class Deps:
    """Runtime dependencies bound into the graph nodes."""
    pipeline: "RAGPipeline"
    provider: "LLMProvider"
    use_rag: bool = True
    config_label: str = "ft_rag"
