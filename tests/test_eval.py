"""Phase-3 evaluation tests — metrics + 4-way benchmark on the offline sample."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _fakes import FakeEmbedder, FakeProvider
from src.assistant.agents.graph import AssistantService
from src.assistant.config import Settings
from src.assistant.eval.benchmark import run_benchmark
from src.assistant.eval.eval_set import sample_eval_set
from src.assistant.eval.generation_metrics import citation_coverage, groundedness
from src.assistant.eval.report import render_markdown, write_explorer_data
from src.assistant.eval.retrieval_metrics import recall_at_k, reciprocal_rank
from src.assistant.rag.ingest import iter_sample_documents
from src.assistant.rag.pipeline import RAGPipeline
from src.assistant.rag.store import LocalVectorStore
from src.assistant.schema import Chunk, RetrievedPassage


def _passages(doc_ids):
    return [RetrievedPassage(
        chunk=Chunk(chunk_id=f"{d}::0", doc_id=d, source="pubmed", title="t", text="x", ordinal=0),
        score=1.0, rank=i + 1) for i, d in enumerate(doc_ids)]


def test_retrieval_metrics():
    p = _passages(["pubmed:9", "pubmed:1", "pubmed:2"])
    gold = {"pubmed:1"}
    assert recall_at_k(p, gold, 1) == 0.0
    assert recall_at_k(p, gold, 3) == 1.0
    assert reciprocal_rank(p, gold) == pytest.approx(0.5)   # first relevant at rank 2


def test_generation_metrics():
    assert citation_coverage("Metformin is first-line [1]. It is safe [2].") == 1.0
    assert citation_coverage("Metformin is first-line. It is safe [1].") == 0.5
    claims = [{"supported": True}, {"supported": False}]
    assert groundedness(claims) == 0.5


def test_four_way_benchmark(tmp_path):
    cfg = Settings(index_dir=tmp_path, use_reranker=False, retrieve_top_k=5, rerank_top_k=3)
    pipe = RAGPipeline(cfg, store=LocalVectorStore(tmp_path), embedder=FakeEmbedder(), reranker=None)
    pipe.build_index(list(iter_sample_documents()))

    prov = FakeProvider()
    services = {c: AssistantService(pipe, prov, c, settings=cfg, prefer_langgraph=False)
                for c in ["base", "ft", "base_rag", "ft_rag"]}
    results = run_benchmark(services, sample_eval_set())

    assert results["n_questions"] == 3
    # RAG configs retrieve gold docs; non-RAG configs have no retrieval metrics
    assert results["configs"]["ft_rag"]["retrieval"]["recall@5"] > 0
    assert results["configs"]["base"]["retrieval"] == {}
    # RAG answers are grounded + cited; base is not
    assert results["configs"]["ft_rag"]["generation"]["groundedness"] > 0
    assert results["configs"]["base"]["generation"]["citation_coverage"] == 0.0
    assert results["configs"]["ft_rag"]["systems"]["total_tokens"] > 0

    md = render_markdown(results)
    assert "Fine-tuned + RAG" in md and "Recall@1" in md
    out = write_explorer_data(results, tmp_path / "explorer.json")
    assert out.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
