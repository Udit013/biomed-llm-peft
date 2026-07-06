"""Answer agent: generate a grounded response from the retrieved context."""
from __future__ import annotations

import time

from ..serving.prompts import build_answer_messages
from .state import AgentState, Deps


def answer(state: AgentState, deps: Deps) -> AgentState:
    passages = state.get("passages", [])
    context = deps.pipeline.format_context(passages) if passages else None
    messages = build_answer_messages(state["query"], context)

    t0 = time.perf_counter()
    result = deps.provider.generate(messages)
    gen_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "answer": result.text,
        "latency_ms": {**state.get("latency_ms", {}), "generate_ms": gen_ms},
        "token_usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
