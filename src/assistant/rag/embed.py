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


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class HFInferenceEmbedder:
    """Embed via HF Inference feature-extraction — no torch/sentence-transformers.

    Lets the FastAPI backend run on a tiny free-tier image (query-time embedding
    only). Uses the SAME model as the local embedder, so vectors match the index
    built by `scripts/rag_index.py`. Index building still uses the local Embedder
    (batch throughput); serving uses this.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", token: str | None = None):
        self.model_name = model_name
        self.token = token
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model_name, token=self.token)
        return self._client

    def _embed_one(self, text: str) -> np.ndarray:
        out = np.asarray(self._get_client().feature_extraction(text), dtype=np.float32)
        if out.ndim == 2:          # token-level -> mean-pool to a sentence vector
            out = out.mean(axis=0)
        return _l2(out)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts])

    def embed_query(self, query: str) -> np.ndarray:
        return self._embed_one(_QUERY_PREFIX + query)


def get_embedder(provider: str, model_name: str, token: str | None = None):
    if provider == "hf_inference":
        return HFInferenceEmbedder(model_name, token=token)
    return Embedder(model_name)
