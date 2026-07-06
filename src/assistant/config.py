"""Central configuration for the biomedical assistant (pydantic-settings).

All settings are overridable via environment variables (prefix `BIOMED_`) or a
`.env` file, so the same code runs locally (local vector store, no LLM key) and
in production (Neon pgvector, hosted inference) with only env changes.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BIOMED_", env_file=".env", extra="ignore"
    )

    # ---- Retrieval / embeddings ----
    embedding_model: str = "BAAI/bge-small-en-v1.5"        # small, CPU-friendly
    reranker_model: str = "BAAI/bge-reranker-base"          # cross-encoder, optional
    embedding_dim: int = 384                                # bge-small-en-v1.5
    chunk_size: int = 512                                   # chars per chunk (approx)
    chunk_overlap: int = 64
    retrieve_top_k: int = 20                                # candidates before rerank
    rerank_top_k: int = 5                                   # passages kept after rerank
    use_reranker: bool = True
    embedding_provider: str = "local"                       # "local" | "hf_inference"
    #   local        -> sentence-transformers (index building; batch throughput)
    #   hf_inference -> HF Inference feature-extraction (torch-free serving image)

    # ---- Vector store ----
    vector_backend: str = "local"                           # "local" | "pgvector"
    database_url: str | None = None                         # Neon pgvector DSN (prod)
    index_dir: Path = REPO_ROOT / "data" / "index"          # local store persistence
    corpus_dir: Path = REPO_ROOT / "data" / "corpus"        # ingested documents

    # ---- LLM serving (live FT+RAG path) ----
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_repo: str = "Udit013/qwen2.5-7b-medmcqa-qlora-5k"
    inference_provider: str = "local"                       # "local" | "hf_inference"
    hf_token: str | None = None
    max_new_tokens: int = 512

    # ---- Ingestion (NCBI E-utilities) ----
    ncbi_email: str | None = None                           # courtesy header for NCBI
    ncbi_api_key: str | None = None                         # higher rate limit if set

    def ensure_dirs(self) -> None:
        for d in (self.index_dir, self.corpus_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide singleton so config is loaded once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
