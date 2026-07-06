"""Retrieval metrics: Recall@k and MRR against gold-relevant documents.

Gold relevance is at the *document* level (a set of `doc_id`s per query); a
retrieved passage counts as a hit if its parent doc is gold. This matches how the
curated eval set is annotated.
"""
from __future__ import annotations

from ..schema import RetrievedPassage


def _retrieved_doc_ids(passages: list[RetrievedPassage]) -> list[str]:
    seen, out = set(), []
    for p in passages:                       # dedup by doc, preserve rank order
        if p.chunk.doc_id not in seen:
            seen.add(p.chunk.doc_id)
            out.append(p.chunk.doc_id)
    return out


def recall_at_k(passages: list[RetrievedPassage], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    topk = set(_retrieved_doc_ids(passages)[:k])
    return len(topk & gold) / len(gold)


def reciprocal_rank(passages: list[RetrievedPassage], gold: set[str]) -> float:
    for rank, doc_id in enumerate(_retrieved_doc_ids(passages), start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval(per_query: list[dict]) -> dict[str, float]:
    """Mean of per-query {recall@k..., mrr}. Empty -> zeros."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}
