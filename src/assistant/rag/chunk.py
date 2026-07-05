"""Sentence-aware chunking with overlap.

Splits a Document into ~`chunk_size`-char chunks on sentence boundaries (so a
passage never cuts mid-sentence), with `overlap` chars of trailing context
carried into the next chunk to preserve continuity across boundaries. No NLTK
dependency — a lightweight regex sentence splitter is plenty for abstracts.
"""
from __future__ import annotations

import re

from ..schema import Chunk, Document

# Split on ., !, ? followed by whitespace + capital/number; keep it simple + fast.
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def chunk_document(doc: Document, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
    """Greedy sentence packing into <= chunk_size chars, with char overlap."""
    sentences = split_sentences(doc.text) or [doc.text.strip()]
    chunks: list[str] = []
    buf = ""
    for sent in sentences:
        if buf and len(buf) + 1 + len(sent) > chunk_size:
            chunks.append(buf)
            # carry the tail of the previous chunk as overlap context
            buf = (buf[-overlap:] + " " + sent).strip() if overlap else sent
        else:
            buf = f"{buf} {sent}".strip() if buf else sent
    if buf:
        chunks.append(buf)

    out: list[Chunk] = []
    for i, ctext in enumerate(chunks):
        out.append(Chunk(
            chunk_id=f"{doc.doc_id}::{i}",
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            text=ctext,
            ordinal=i,
            url=doc.url,
            metadata=doc.metadata,
        ))
    return out
