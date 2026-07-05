"""RAG pipeline: index-building and context retrieval.

Ties the stages together and is the single entry point the agents/API use:
  * `build_index(documents)` — chunk → embed → store.
  * `retrieve_context(query)` — retrieve → (optional) rerank → citations,
    returning ranked passages + numbered citations + per-stage latencies.

Answer *generation* lives in the serving/agent layer (Phase 2); this module is
generation-agnostic so retrieval can be evaluated on its own.
"""
from __future__ import annotations

from ..config import Settings, get_settings
from ..logging import get_logger, timed
from ..schema import Citation, Document, EmbeddedChunk, RetrievedPassage
from .chunk import chunk_document
from .citations import build_citations
from .embed import Embedder
from .rerank import Reranker
from .retrieve import Retriever
from .store import VectorStore, get_store

log = get_logger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings | None = None,
                 store: VectorStore | None = None,
                 embedder: Embedder | None = None,
                 reranker: Reranker | None = None):
        self.cfg = settings or get_settings()
        self.embedder = embedder or Embedder(self.cfg.embedding_model)
        self.store = store or get_store(
            self.cfg.vector_backend, self.cfg.index_dir,
            self.cfg.embedding_dim, self.cfg.database_url)
        self.reranker = reranker
        if self.cfg.use_reranker and reranker is None:
            self.reranker = Reranker(self.cfg.reranker_model)
        self.retriever = Retriever(self.store, self.embedder)

    # ---- indexing ----
    def build_index(self, documents: list[Document]) -> int:
        n_chunks = 0
        for doc in documents:
            chunks = chunk_document(doc, self.cfg.chunk_size, self.cfg.chunk_overlap)
            if not chunks:
                continue
            vecs = self.embedder.embed_documents([c.text for c in chunks])
            self.store.add([EmbeddedChunk(chunk=c, embedding=v.tolist())
                            for c, v in zip(chunks, vecs)])
            n_chunks += len(chunks)
        log.info("index built", extra={"documents": len(documents), "chunks": n_chunks,
                                       "total": self.store.count()})
        return n_chunks

    # ---- retrieval ----
    def retrieve_context(self, query: str, metadata_filter: dict | None = None
                         ) -> tuple[list[RetrievedPassage], list[Citation], dict]:
        latencies: dict[str, float] = {}
        with timed(log, "retrieve", query=query) as t:
            passages = self.retriever.retrieve(
                query, k=self.cfg.retrieve_top_k, metadata_filter=metadata_filter)
        latencies["retrieve_ms"] = t.ms

        if self.reranker and passages:
            with timed(log, "rerank", n=len(passages)) as t:
                passages = self.reranker.rerank(query, passages, top_k=self.cfg.rerank_top_k)
            latencies["rerank_ms"] = t.ms
        else:
            passages = passages[: self.cfg.rerank_top_k]

        citations = build_citations(passages)
        return passages, citations, latencies

    @staticmethod
    def format_context(passages: list[RetrievedPassage]) -> str:
        """Render numbered passages for the LLM prompt (matches [n] citation markers)."""
        return "\n\n".join(
            f"[{i}] ({p.chunk.source}: {p.chunk.title})\n{p.chunk.text}"
            for i, p in enumerate(passages, start=1)
        )
