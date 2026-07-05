"""Cross-encoder reranking of retrieved passages (optional, lazy).

Bi-encoder retrieval (embed.py) is fast but coarse; a cross-encoder re-scores each
(query, passage) pair jointly for much better precision on the top results. We
retrieve a wide candidate set, rerank, and keep the top `k`. Falls back to the
retrieval order if sentence-transformers isn't available.
"""
from __future__ import annotations

from ..schema import RetrievedPassage


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, passages: list[RetrievedPassage],
               top_k: int = 5) -> list[RetrievedPassage]:
        if not passages:
            return []
        model = self._load()
        scores = model.predict([(query, p.chunk.text) for p in passages])
        for p, s in zip(passages, scores):
            p.rerank_score = float(s)
        ranked = sorted(passages, key=lambda p: p.rerank_score, reverse=True)[:top_k]
        for i, p in enumerate(ranked):
            p.rank = i + 1
        return ranked
