"""Semantic retrieval: embed the query, search the store, apply metadata filters."""
from __future__ import annotations

from ..schema import RetrievedPassage
from .embed import Embedder
from .store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int = 20,
                 metadata_filter: dict | None = None) -> list[RetrievedPassage]:
        qvec = self.embedder.embed_query(query)
        return self.store.search(qvec, k=k, metadata_filter=metadata_filter)
