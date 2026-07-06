"""FastAPI backend: live FT+RAG answering + precomputed benchmark data.

Endpoints:
  GET  /health          — liveness + config summary
  POST /query           — live Fine-tuned + RAG answer for an arbitrary question
                          (Planner → Retrieval → Answer → Citation-Verification)
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


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3,
                          examples=["What is first-line therapy for type 2 diabetes?"])


def _build_service():
    """Construct the live FT+RAG AssistantService (provider per config)."""
    from ..agents.graph import AssistantService
    from ..rag.pipeline import RAGPipeline
    from ..serving.providers import HFInferenceProvider, LocalTransformersProvider

    cfg = get_settings()
    pipeline = RAGPipeline(cfg)
    if cfg.inference_provider == "hf_inference":
        provider = HFInferenceProvider(cfg.adapter_repo, token=cfg.hf_token,
                                       max_new_tokens=cfg.max_new_tokens)
    else:
        provider = LocalTransformersProvider(cfg.base_model, adapter_dir=cfg.adapter_repo,
                                             max_new_tokens=cfg.max_new_tokens)
    return AssistantService(pipeline, provider, config_label="ft_rag", settings=cfg)


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
              description="Live Fine-tuned + RAG QA with citations. NOT for clinical use.",
              lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    cfg = _STATE.get("cfg")
    return {"status": "ok", "vector_backend": getattr(cfg, "vector_backend", None),
            "inference_provider": getattr(cfg, "inference_provider", None),
            "model_loaded": _STATE.get("service") is not None,
            "benchmark_available": "benchmark" in _STATE}


@app.post("/query")
def query(req: QueryRequest) -> dict:
    if _STATE.get("service") is None:
        log.info("cold start: building FT+RAG service")
        _STATE["service"] = _build_service()
    ans = _STATE["service"].answer(req.question)
    return ans.model_dump()


@app.get("/benchmark")
def benchmark() -> dict:
    if "benchmark" not in _STATE:
        raise HTTPException(404, "No benchmark data. Run scripts/rag_benchmark.py first.")
    return _STATE["benchmark"]
