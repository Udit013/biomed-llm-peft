"""Typed data models shared across the RAG + agent pipeline (pydantic).

These are the contracts between stages: ingest → chunk → embed → store →
retrieve → rerank → answer → cite/verify. Keeping them typed makes the pipeline
self-documenting and lets the API serialize responses for free.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Source = Literal["pubmed", "pmc", "nih", "who", "cdc"]


class Document(BaseModel):
    """A normalized source document (one abstract or guideline)."""
    doc_id: str                      # e.g. "pubmed:12345678"
    source: Source
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # year, journal, authors, mesh...


class Chunk(BaseModel):
    """A retrievable slice of a Document."""
    chunk_id: str                    # f"{doc_id}::{ordinal}"
    doc_id: str
    source: Source
    title: str
    text: str
    ordinal: int
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddedChunk(BaseModel):
    chunk: Chunk
    embedding: list[float]


class RetrievedPassage(BaseModel):
    """A chunk returned by retrieval, with scores attached."""
    chunk: Chunk
    score: float                     # similarity from the vector store
    rerank_score: float | None = None
    rank: int | None = None


class Citation(BaseModel):
    """An inline citation linking an answer claim to supporting evidence."""
    marker: str                      # e.g. "[1]"
    doc_id: str
    source: Source
    title: str
    url: str | None = None
    quote: str                       # the supporting passage span
    supported: bool | None = None    # set by the citation-verification agent
    support_score: float | None = None


class GroundedAnswer(BaseModel):
    """Final answer with citations, evidence, and verification/latency metadata."""
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    passages: list[RetrievedPassage] = Field(default_factory=list)
    config: str = "ft_rag"           # base | ft | base_rag | ft_rag
    claims: list[dict[str, Any]] = Field(default_factory=list)  # per-sentence grounding
    all_claims_supported: bool | None = None
    latency_ms: dict[str, float] = Field(default_factory=dict)  # per-stage timings
    token_usage: dict[str, int] = Field(default_factory=dict)
