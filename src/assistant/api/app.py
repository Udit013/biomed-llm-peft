"""FastAPI backend: live RAG answering + precomputed benchmark data.

Endpoints:
  GET  /health          — liveness + which config is actually served
  POST /query           — live RAG answer for an arbitrary question (Planner →
                          Retrieval → Answer → Citation-Verification). The
                          response's `config` says what served it (base_rag on
                          free tier; ft_rag on a GPU backend with the adapter).
  GET  /benchmark       — precomputed 4-way Benchmark Explorer data (if present)

The 7B FT model is loaded lazily on the FIRST /query (documented cold start).
Config comes from BIOMED_* env vars: on Render/free-tier set
BIOMED_INFERENCE_PROVIDER=hf_inference so a hosted endpoint serves the model.

NOT FOR CLINICAL USE — research/education only.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import get_settings
from ..logging import get_logger

log = get_logger(__name__)
_STATE: dict = {}
_EXPLORER_PATH = Path("results/benchmark_explorer.json")

CONFIG_DISPLAY = {"base_rag": "Base + RAG", "ft_rag": "Fine-tuned + RAG"}


def served_config(cfg) -> str:
    """The config the live backend ACTUALLY serves — a pure function of env.

    hf_inference (free tier) can only serve the base model -> 'base_rag'. Point
    BIOMED_INFERENCE_PROVIDER at a local/GPU backend with the adapter and it
    becomes 'ft_rag' — with no API or UI change, since both read this label.
    """
    return "base_rag" if cfg.inference_provider == "hf_inference" else "ft_rag"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3,
                          examples=["What is first-line therapy for type 2 diabetes?"])


def _build_service():
    """Construct the live RAG AssistantService (provider per config)."""
    from ..agents.graph import AssistantService
    from ..rag.pipeline import RAGPipeline
    from ..serving.providers import HFInferenceProvider, LocalTransformersProvider

    cfg = get_settings()
    pipeline = RAGPipeline(cfg)
    if cfg.inference_provider == "hf_inference":
        # Serverless HF Inference can't serve a custom LoRA adapter, so it serves
        # the BASE model -> honestly "Base + RAG".
        provider = HFInferenceProvider(cfg.base_model, token=cfg.hf_token,
                                       max_new_tokens=cfg.max_new_tokens)
    else:
        # GPU/local backend with the adapter -> true "Fine-tuned + RAG".
        provider = LocalTransformersProvider(cfg.base_model, adapter_dir=cfg.adapter_repo,
                                             max_new_tokens=cfg.max_new_tokens)
    return AssistantService(pipeline, provider, config_label=served_config(cfg), settings=cfg)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    _STATE["cfg"] = cfg
    _STATE["service"] = None  # lazy — built on first /query (cold start)
    if _EXPLORER_PATH.exists():
        _STATE["benchmark"] = json.loads(_EXPLORER_PATH.read_text())
    log.info("api ready", extra={"backend": cfg.vector_backend,
                                 "provider": cfg.inference_provider})
    yield
    _STATE.clear()


app = FastAPI(title="Biomedical AI Research Assistant",
              description="Live RAG QA with citations (Base + RAG on free tier; the "
                          "response's `config` field states what was actually served). "
                          "NOT for clinical use.",
              lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    cfg = _STATE.get("cfg")
    label = served_config(cfg) if cfg else None
    return {"status": "ok",
            "served_config": label,                              # base_rag | ft_rag
            "served_config_display": CONFIG_DISPLAY.get(label),  # "Base + RAG" | ...
            "vector_backend": getattr(cfg, "vector_backend", None),
            "inference_provider": getattr(cfg, "inference_provider", None),
            "model_loaded": _STATE.get("service") is not None,
            "benchmark_available": "benchmark" in _STATE}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if _STATE.get("service") is None:
        log.info("cold start: building RAG service")
        _STATE["service"] = _build_service()
    ans = _STATE["service"].answer(req.question)
    return ans.model_dump()


@app.get("/benchmark")
def benchmark() -> dict:
    if "benchmark" not in _STATE:
        raise HTTPException(404, "No benchmark data. Run scripts/rag_benchmark.py first.")
    return _STATE["benchmark"]
