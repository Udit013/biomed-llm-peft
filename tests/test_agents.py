"""Phase-2 agent-workflow tests — CPU-only (fake embedder + fake LLM provider).

Exercises the full Planner → Retrieval → Answer → Verify pipeline via the
sequential fallback (LangGraph not required in CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _fakes import FakeEmbedder, FakeProvider
from src.assistant.agents.graph import AssistantService
from src.assistant.agents.planner import plan
from src.assistant.agents.state import Deps
from src.assistant.config import Settings
from src.assistant.rag.ingest import iter_sample_documents
from src.assistant.rag.pipeline import RAGPipeline
from src.assistant.rag.store import LocalVectorStore


def _pipeline(tmp_path):
    cfg = Settings(index_dir=tmp_path, use_reranker=False, retrieve_top_k=5, rerank_top_k=3)
    pipe = RAGPipeline(cfg, store=LocalVectorStore(tmp_path), embedder=FakeEmbedder(),
                       reranker=None)
    pipe.build_index(list(iter_sample_documents()))
    return pipe, cfg


def test_planner_source_filter():
    deps = Deps(pipeline=None, provider=None, use_rag=True)  # planner needs neither
    out = plan({"query": "What does the CDC recommend for adult vaccines?"}, deps)
    assert out["metadata_filter"]["source"] == ["cdc"]
    out2 = plan({"query": "latest guideline for hypertension"}, deps)
    assert set(out2["metadata_filter"]["source"]) == {"nih", "who", "cdc"}
    assert out2["metadata_filter"]["year_min"] == 2018


def test_ft_rag_answer_is_grounded(tmp_path):
    pipe, cfg = _pipeline(tmp_path)
    svc = AssistantService(pipe, FakeProvider(), config_label="ft_rag",
                           settings=cfg, prefer_langgraph=False)
    ans = svc.answer("What is first-line therapy for type 2 diabetes?")
    assert ans.config == "ft_rag"
    assert ans.answer and "[1]" in ans.answer
    assert ans.passages and ans.citations
    assert ans.all_claims_supported is True                 # fake echoes the source
    assert ans.citations[0].supported is True
    assert "retrieve_ms" in ans.latency_ms and "generate_ms" in ans.latency_ms
    assert ans.token_usage["total_tokens"] > 0


def test_base_config_skips_retrieval(tmp_path):
    pipe, cfg = _pipeline(tmp_path)
    svc = AssistantService(pipe, FakeProvider(), config_label="base",
                           settings=cfg, prefer_langgraph=False)
    ans = svc.answer("What is first-line therapy for type 2 diabetes?")
    assert ans.config == "base"
    assert ans.passages == [] and ans.citations == []
    assert ans.all_claims_supported is None                 # nothing to verify
    assert ans.answer


def test_semantic_grounding():
    from src.assistant.rag.citations import verify_claims_semantic
    from src.assistant.schema import Chunk, RetrievedPassage

    p = RetrievedPassage(chunk=Chunk(
        chunk_id="w::0", doc_id="who:x", source="who", title="t",
        text="Thiazide diuretics and ACE inhibitors are first-line for hypertension.",
        ordinal=0), score=0.9)
    emb = FakeEmbedder()
    claims, ok = verify_claims_semantic(
        "Thiazide diuretics are first-line for hypertension.", [p], emb, threshold=0.4)
    assert claims and claims[0]["supported"] and ok
    _, ok2 = verify_claims_semantic(
        "Aspirin cures the common cold entirely.", [p], emb, threshold=0.6)
    assert not ok2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
