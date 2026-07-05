"""Phase-1 RAG core tests — CPU-only, no network, no model downloads.

A deterministic FakeEmbedder (hashed bag-of-words) stands in for the real
sentence-transformer so the pipeline runs end-to-end in CI in milliseconds.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant.rag.chunk import chunk_document, split_sentences
from src.assistant.rag.citations import build_citations, verify_claims
from src.assistant.rag.ingest import iter_sample_documents
from src.assistant.rag.pipeline import RAGPipeline
from src.assistant.rag.store import LocalVectorStore
from src.assistant.schema import Document, RetrievedPassage, Chunk
from src.assistant.config import Settings

_W = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector. Deterministic, no downloads."""
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _W.findall(text.lower()):
            v[hash(w) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts): return np.vstack([self._vec(t) for t in texts])
    def embed_query(self, query):    return self._vec(query)


def test_split_sentences():
    s = split_sentences("Metformin lowers glucose. It is first-line. Dose is 500 mg.")
    assert len(s) == 3 and s[0].startswith("Metformin")


def test_chunk_overlap_and_ids():
    doc = Document(doc_id="pubmed:1", source="pubmed", title="t",
                   text=" ".join(f"Sentence number {i} about diabetes." for i in range(40)))
    chunks = chunk_document(doc, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(c.chunk_id == f"pubmed:1::{c.ordinal}" for c in chunks)
    assert all(len(c.text) <= 200 for c in chunks)  # size + overlap slack


def test_local_store_search_and_filter(tmp_path):
    store = LocalVectorStore(tmp_path)
    emb = FakeEmbedder()
    from src.assistant.schema import EmbeddedChunk
    docs = list(iter_sample_documents())
    from src.assistant.rag.chunk import chunk_document
    chunks = [c for d in docs for c in chunk_document(d, 512, 64)]
    vecs = emb.embed_documents([c.text for c in chunks])
    store.add([EmbeddedChunk(chunk=c, embedding=v.tolist()) for c, v in zip(chunks, vecs)])
    assert store.count() == len(chunks)

    hits = store.search(emb.embed_query("first-line diabetes drug metformin"), k=3)
    assert hits and hits[0].chunk.source == "pubmed"        # most relevant is the metformin doc

    who_only = store.search(emb.embed_query("blood pressure"), k=5,
                            metadata_filter={"source": ["who"]})
    assert all(h.chunk.source == "who" for h in who_only)


def test_store_persistence(tmp_path):
    from src.assistant.schema import EmbeddedChunk
    emb = FakeEmbedder()
    c = Chunk(chunk_id="x::0", doc_id="x", source="cdc", title="t", text="influenza vaccine", ordinal=0)
    s1 = LocalVectorStore(tmp_path)
    s1.add([EmbeddedChunk(chunk=c, embedding=emb.embed_documents(["influenza vaccine"])[0].tolist())])
    s2 = LocalVectorStore(tmp_path)                          # reloads from disk
    assert s2.count() == 1 and s2.search(emb.embed_query("influenza"), k=1)


def test_pipeline_retrieve_context(tmp_path):
    cfg = Settings(index_dir=tmp_path, use_reranker=False, retrieve_top_k=5, rerank_top_k=3)
    pipe = RAGPipeline(cfg, store=LocalVectorStore(tmp_path), embedder=FakeEmbedder(),
                       reranker=None)
    pipe.build_index(list(iter_sample_documents()))
    passages, citations, lat = pipe.retrieve_context("recommended treatment for hypertension")
    assert passages and len(citations) == len(passages)
    assert citations[0].marker == "[1]"
    assert "retrieve_ms" in lat
    ctx = pipe.format_context(passages)
    assert "[1]" in ctx


def test_verify_claims_grounding():
    p = RetrievedPassage(
        chunk=Chunk(chunk_id="w::0", doc_id="who:x", source="who", title="t",
                    text="Thiazide diuretics and ACE inhibitors are first-line for hypertension.",
                    ordinal=0), score=0.9)
    claims, ok = verify_claims("Thiazide diuretics are first-line for hypertension.", [p],
                               threshold=0.3)
    assert claims[0]["supported"] and ok
    bad, ok2 = verify_claims("Aspirin cures the common cold entirely.", [p], threshold=0.5)
    assert not bad[0]["supported"] and not ok2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
