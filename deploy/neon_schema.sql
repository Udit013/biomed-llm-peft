-- Neon Postgres + pgvector schema for the biomedical RAG index.
-- The app creates this automatically (src/assistant/rag/store.py:PgVectorStore),
-- but it's documented here for reference / manual setup.
-- Embedding dim 384 matches BAAI/bge-small-en-v1.5 (change if you swap models).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS biomed_chunks (
    chunk_id  TEXT PRIMARY KEY,
    doc_id    TEXT,
    source    TEXT,            -- pubmed | pmc | nih | who | cdc
    title     TEXT,
    text      TEXT,
    ordinal   INT,
    url       TEXT,
    year      INT,
    metadata  JSONB,
    embedding vector(384)
);

-- Approximate nearest-neighbor index (cosine) for fast retrieval.
CREATE INDEX IF NOT EXISTS biomed_chunks_emb_idx
    ON biomed_chunks USING hnsw (embedding vector_cosine_ops);

-- Optional metadata-filter helpers.
CREATE INDEX IF NOT EXISTS biomed_chunks_source_idx ON biomed_chunks (source);
CREATE INDEX IF NOT EXISTS biomed_chunks_year_idx   ON biomed_chunks (year);
