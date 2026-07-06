"""Assemble the agents into a workflow and expose `AssistantService`.

The four agents are wired Planner → Retrieval → Answer → Verify. When LangGraph
is installed we build a real `StateGraph` (production); otherwise we fall back to
running the same node functions sequentially — identical behavior, zero-dependency,
so tests and CPU-only environments work. `AssistantService.answer()` returns a
fully-typed `GroundedAnswer`.
"""
from __future__ import annotations

from ..config import Settings, get_settings
from ..logging import get_logger
from ..schema import GroundedAnswer
from ..rag.pipeline import RAGPipeline
from ..serving.providers import LLMProvider
from . import answer as answer_agent
from . import planner as planner_agent
from . import retrieval as retrieval_agent
from . import verify as verify_agent
from .state import AgentState, Deps

log = get_logger(__name__)

# Config label -> (use_ft_adapter, use_rag). Serving one config per service.
CONFIGS = {
    "base": (False, False),
    "ft": (True, False),
    "base_rag": (False, True),
    "ft_rag": (True, True),
}


def _sequential(state: AgentState, deps: Deps) -> AgentState:
    for node in (planner_agent.plan, retrieval_agent.retrieve,
                 answer_agent.answer, verify_agent.verify):
        state = {**state, **node(state, deps)}
    return state


def build_langgraph(deps: Deps):
    """Compile a LangGraph StateGraph (lazy import). Raises ImportError if absent."""
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("planner", lambda s: planner_agent.plan(s, deps))
    g.add_node("retrieval", lambda s: retrieval_agent.retrieve(s, deps))
    g.add_node("answer", lambda s: answer_agent.answer(s, deps))
    g.add_node("verify", lambda s: verify_agent.verify(s, deps))
    g.set_entry_point("planner")
    g.add_edge("planner", "retrieval")
    g.add_edge("retrieval", "answer")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    return g.compile()


class AssistantService:
    """One serving configuration (e.g. FT+RAG for the live demo)."""

    def __init__(self, pipeline: RAGPipeline, provider: LLMProvider,
                 config_label: str = "ft_rag", settings: Settings | None = None,
                 prefer_langgraph: bool = True):
        self.cfg = settings or get_settings()
        _, use_rag = CONFIGS.get(config_label, (True, True))
        self.deps = Deps(pipeline=pipeline, provider=provider, use_rag=use_rag,
                         config_label=config_label)
        self.config_label = config_label
        self._graph = None
        if prefer_langgraph:
            try:
                self._graph = build_langgraph(self.deps)
                log.info("langgraph compiled", extra={"config": config_label})
            except Exception as exc:  # langgraph missing or incompatible
                log.info("langgraph unavailable; sequential fallback",
                         extra={"reason": str(exc)})

    def answer(self, query: str) -> GroundedAnswer:
        init: AgentState = {"query": query, "config_label": self.config_label,
                            "latency_ms": {}, "token_usage": {}}
        final = self._graph.invoke(init) if self._graph else _sequential(init, self.deps)
        return GroundedAnswer(
            query=query,
            answer=final.get("answer", ""),
            citations=final.get("citations", []),
            passages=final.get("passages", []),
            config=self.config_label,
            all_claims_supported=final.get("all_supported"),
            latency_ms=final.get("latency_ms", {}),
            token_usage=final.get("token_usage", {}),
        )
