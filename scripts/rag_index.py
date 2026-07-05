#!/usr/bin/env python
"""Build the RAG index from the curated corpus (reproducible).

Ingests PubMed abstracts for the topic queries in configs/corpus.yaml plus any
local NIH/WHO/CDC guideline files, then chunks → embeds → stores. Backend and
models come from BIOMED_* env / defaults (local store unless BIOMED_VECTOR_BACKEND=pgvector).

Usage:
    python scripts/rag_index.py --config configs/corpus.yaml
    python scripts/rag_index.py --sample          # tiny offline corpus (no network)
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import yaml

from src.assistant.config import get_settings
from src.assistant.logging import get_logger
from src.assistant.rag.ingest import fetch_pubmed, iter_sample_documents, load_guidelines
from src.assistant.rag.pipeline import RAGPipeline

log = get_logger("rag_index")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/corpus.yaml")
    ap.add_argument("--sample", action="store_true", help="Tiny offline corpus (no network).")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.ensure_dirs()
    pipeline = RAGPipeline(cfg)

    if args.sample:
        docs = list(iter_sample_documents())
    else:
        spec = yaml.safe_load(open(args.config))
        docs = load_guidelines(cfg.corpus_dir)
        for q in spec.get("pubmed_queries", []):
            fetched = fetch_pubmed(q, retmax=spec.get("retmax_per_query", 200),
                                   email=cfg.ncbi_email, api_key=cfg.ncbi_api_key)
            log.info("fetched", extra={"query": q, "n": len(fetched)})
            docs.extend(fetched)

    # de-dup by doc_id (queries overlap)
    docs = list({d.doc_id: d for d in docs}.values())
    n = pipeline.build_index(docs)
    print(f"[rag_index] indexed {len(docs)} documents -> {n} chunks "
          f"({pipeline.store.count()} total in {cfg.vector_backend} store)")


if __name__ == "__main__":
    main()
