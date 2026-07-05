"""Vector store abstraction with two backends.

  * LocalVectorStore  — numpy cosine search, persisted to .npz + .jsonl. Zero
                        external services; used for dev, CI, tests, and offline
                        benchmarking.
  * PgVectorStore     — Neon Postgres + pgvector for production. Lazy-imports
                        psycopg/pgvector so the module loads without them.

Both implement the same `VectorStore` interface, so retrieval code is
backend-agnostic (`config.vector_backend` selects one). Vectors are assumed
L2-normalized (see embed.py), so cosine == inner product.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from ..schema import Chunk, EmbeddedChunk, RetrievedPassage


def _passes_filter(meta: dict, source: str, flt: dict[str, Any] | None) -> bool:
    """Metadata filtering: {'source': ['pubmed','who'], 'year_min': 2015}."""
    if not flt:
        return True
    if "source" in flt and source not in set(flt["source"]):
        return False
    if "year_min" in flt and int(meta.get("year", 0) or 0) < flt["year_min"]:
        return False
    if "year_max" in flt and int(meta.get("year", 9999) or 9999) > flt["year_max"]:
        return False
    return True


class VectorStore(ABC):
    @abstractmethod
    def add(self, embedded: list[EmbeddedChunk]) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int,
               metadata_filter: dict | None = None) -> list[RetrievedPassage]: ...

    @abstractmethod
    def count(self) -> int: ...


class LocalVectorStore(VectorStore):
    def __init__(self, index_dir: str | Path):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._vecs: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._load()

    def add(self, embedded: list[EmbeddedChunk]) -> None:
        if not embedded:
            return
        new = np.asarray([e.embedding for e in embedded], dtype=np.float32)
        self._vecs = new if self._vecs is None else np.vstack([self._vecs, new])
        self._chunks.extend(e.chunk for e in embedded)
        self._persist()

    def search(self, query_vec, k, metadata_filter=None) -> list[RetrievedPassage]:
        if self._vecs is None or len(self._chunks) == 0:
            return []
        sims = self._vecs @ np.asarray(query_vec, dtype=np.float32)  # cosine (normalized)
        order = np.argsort(-sims)
        out: list[RetrievedPassage] = []
        for idx in order:
            c = self._chunks[int(idx)]
            if not _passes_filter(c.metadata, c.source, metadata_filter):
                continue
            out.append(RetrievedPassage(chunk=c, score=float(sims[idx]), rank=len(out) + 1))
            if len(out) >= k:
                break
        return out

    def count(self) -> int:
        return len(self._chunks)

    # ---- persistence ----
    @property
    def _vec_path(self):  return self.dir / "vectors.npz"
    @property
    def _meta_path(self): return self.dir / "chunks.jsonl"

    def _persist(self) -> None:
        if self._vecs is not None:
            np.savez_compressed(self._vec_path, vecs=self._vecs)
        with self._meta_path.open("w") as fh:
            for c in self._chunks:
                fh.write(c.model_dump_json() + "\n")

    def _load(self) -> None:
        if self._vec_path.exists() and self._meta_path.exists():
            self._vecs = np.load(self._vec_path)["vecs"]
            with self._meta_path.open() as fh:
                self._chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]


class PgVectorStore(VectorStore):
    """Neon Postgres + pgvector backend (production)."""

    def __init__(self, dsn: str, dim: int, table: str = "biomed_chunks"):
        self.dsn, self.dim, self.table = dsn, dim, table
        self._ensure_schema()

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id TEXT PRIMARY KEY, doc_id TEXT, source TEXT, title TEXT,
                    text TEXT, ordinal INT, url TEXT, year INT,
                    metadata JSONB, embedding vector({self.dim})
                )""")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_emb_idx ON {self.table} "
                f"USING hnsw (embedding vector_cosine_ops)")
            conn.commit()

    def add(self, embedded: list[EmbeddedChunk]) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            for e in embedded:
                c = e.chunk
                cur.execute(
                    f"""INSERT INTO {self.table}
                        (chunk_id, doc_id, source, title, text, ordinal, url, year,
                         metadata, embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (chunk_id) DO UPDATE SET embedding = EXCLUDED.embedding""",
                    (c.chunk_id, c.doc_id, c.source, c.title, c.text, c.ordinal, c.url,
                     int(c.metadata.get("year", 0) or 0), json.dumps(c.metadata),
                     np.asarray(e.embedding, dtype=np.float32)))
            conn.commit()

    def search(self, query_vec, k, metadata_filter=None) -> list[RetrievedPassage]:
        where, params = "", [np.asarray(query_vec, dtype=np.float32)]
        clauses = []
        if metadata_filter:
            if "source" in metadata_filter:
                clauses.append("source = ANY(%s)"); params.append(list(metadata_filter["source"]))
            if "year_min" in metadata_filter:
                clauses.append("year >= %s"); params.append(metadata_filter["year_min"])
        if clauses:
            where = "WHERE " + " AND ".join(clauses)
        params.append(k)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""SELECT chunk_id, doc_id, source, title, text, ordinal, url, metadata,
                           1 - (embedding <=> %s) AS score
                    FROM {self.table} {where}
                    ORDER BY embedding <=> %s LIMIT %s""",
                [params[0], *params[1:-1], params[0], params[-1]])
            rows = cur.fetchall()
        out = []
        for i, r in enumerate(rows):
            chunk = Chunk(chunk_id=r[0], doc_id=r[1], source=r[2], title=r[3], text=r[4],
                          ordinal=r[5], url=r[6], metadata=r[7] or {})
            out.append(RetrievedPassage(chunk=chunk, score=float(r[8]), rank=i + 1))
        return out

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return int(cur.fetchone()[0])


def get_store(backend: str, index_dir, dim: int, dsn: str | None = None) -> VectorStore:
    if backend == "pgvector":
        if not dsn:
            raise ValueError("vector_backend=pgvector requires database_url")
        return PgVectorStore(dsn, dim)
    return LocalVectorStore(index_dir)
