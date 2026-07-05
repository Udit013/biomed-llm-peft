"""Embedding model wrapper (sentence-transformers), lazily loaded.

Heavy import (sentence_transformers/torch) is deferred to first use so the module
imports on any machine for structural checks. Vectors are L2-normalized, so cosine
similarity == dot product (what the vector stores use).

For retrieval-quality parity, queries and documents use the same model; bge models
recommend a short instruction prefix on the QUERY side only, handled here.
"""
from __future__ import annotations

import numpy as np

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        model = self._load()
        vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._load()
        vec = model.encode([_QUERY_PREFIX + query], normalize_embeddings=True,
                           convert_to_numpy=True)
        return np.asarray(vec, dtype=np.float32)[0]
