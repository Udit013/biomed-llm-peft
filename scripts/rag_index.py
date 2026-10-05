#!/usr/bin/env python
"""Build the RAG index from the curated corpus (reproducible).

Ingests PubMed abstracts for the topic queries in configs/corpus.yaml plus any
local NIH/WHO/CDC guideline files, then chunks → embeds → stores. Backend and
models come from BIOMED_* env / defaults (local store unless BIOMED_VECTOR_BACKEND=pgvector).

Usage:
    python scripts/rag_index.py --config configs/corpus.yaml            # upsert
    python scripts/rag_index.py --config configs/corpus.yaml --rebuild  # replace atomically
    python scripts/rag_index.py --config configs/corpus.yaml --dry-run  # fetch+embed, no writes
    python scripts/rag_index.py --sample          # tiny offline corpus (no network)
"""
from __future__ import annotations

import argparse
import re

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
    ap.add_argument("--rebuild", action="store_true",
                    help="Replace the whole index (staging table + atomic swap).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch, chunk and embed, report counts, write nothing.")
    args = ap.parse_args()

    cfg = get_settings()
    cfg.ensure_dirs()
    pipeline = RAGPipeline(cfg)

    if args.sample:
        docs = list(iter_sample_documents())
    else:
        spec = yaml.safe_load(open(args.config))
        docs = load_guidelines(cfg.corpus_dir)
        flt = spec.get("query_filter", "")
        for q in spec.get("pubmed_queries", []):
            fetched = fetch_pubmed(f"({q}){flt}", retmax=spec.get("retmax_per_query", 200),
                                   email=cfg.ncbi_email, api_key=cfg.ncbi_api_key)
            log.info("fetched", extra={"query": q, "n": len(fetched)})
            docs.extend(fetched)

    # de-dup by doc_id (queries overlap), then by abstract text: PubMed republishes
    # some abstracts under several PMIDs, which crowded the top-k with copies.
    docs = list({d.doc_id: d for d in docs}.values())
    by_text = {}
    for d in docs:
        by_text.setdefault(" ".join(re.findall(r"[a-z0-9]+", d.text.lower()))[:500], d)
    print(f"[rag_index] {len(docs)} unique PMIDs, {len(docs) - len(by_text)} "
          f"duplicate abstracts dropped")
    docs = list(by_text.values())

    if args.dry_run or args.rebuild:
        embedded = pipeline.embed_corpus(docs)
        print(f"[rag_index] {len(docs)} documents -> {len(embedded)} chunks embedded")
        if args.dry_run:
            print("[rag_index] dry run: nothing written")
            return
        pipeline.store.rebuild(embedded)
        print(f"[rag_index] rebuilt {cfg.vector_backend} index: {pipeline.store.count()} chunks")
        return
    n = pipeline.build_index(docs)
    print(f"[rag_index] indexed {len(docs)} documents -> {n} chunks "
          f"({pipeline.store.count()} total in {cfg.vector_backend} store)")


if __name__ == "__main__":
    main()
