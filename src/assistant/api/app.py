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
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import get_settings
from ..logging import get_logger
from ..serving.providers import UpstreamError

log = get_logger(__name__)
_STATE: dict = {}
_BUILD_LOCK = threading.Lock()   # /query runs in a threadpool: build the service once
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
    question: str = Field(..., min_length=3, max_length=1000,
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
                                       max_new_tokens=cfg.max_new_tokens,
                                       providers=cfg.hf_provider_list,
                                       timeout=cfg.inference_timeout_s)
    else:
        # GPU/local backend with the adapter -> true "Fine-tuned + RAG".
        provider = LocalTransformersProvider(cfg.base_model, adapter_dir=cfg.adapter_repo,
                                             max_new_tokens=cfg.max_new_tokens)
    return AssistantService(pipeline, provider, config_label=served_config(cfg), settings=cfg)


class RateLimiter:
    """Sliding-window limit per client key. In-memory: correct for the single
    free-tier instance; multiple replicas would need a shared store (Redis)."""

    def __init__(self, per_minute: int, window_s: float = 60.0):
        self.per_minute, self.window_s = per_minute, window_s
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> float:
        """Record a hit; return 0 if allowed, else seconds until a slot frees."""
        if self.per_minute <= 0:
            return 0.0
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] >= self.window_s:
                q.popleft()
            if len(q) >= self.per_minute:
                return self.window_s - (now - q[0])
            q.append(now)
            if len(self._hits) > 10_000:          # bound memory under key churn
                self._hits = {k: v for k, v in self._hits.items() if v}
            return 0.0


def _client_key(request: Request) -> str:
    # Render terminates TLS at a proxy; the first X-Forwarded-For hop is the client.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def _get_service():
    svc = _STATE.get("service")
    if svc is None:
        with _BUILD_LOCK:
            svc = _STATE.get("service")
            if svc is None:
                log.info("cold start: building RAG service")
                svc = _STATE["service"] = _build_service()
    return svc


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    _STATE["cfg"] = cfg
    _STATE["service"] = None  # lazy — built on first /query (cold start)
    _STATE["limiter"] = RateLimiter(cfg.rate_limit_per_minute)
    _STATE["last_query"] = None
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
              version="1.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id + server-timing to every response, and log it."""
    rid = (request.headers.get("x-request-id") or "")[:64] or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = str(dt)
    log.info("request", extra={"request_id": rid, "method": request.method,
                               "path": request.url.path,
                               "status": response.status_code, "latency_ms": dt})
    return response


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
            "benchmark_available": "benchmark" in _STATE,
            # Liveness alone hid a total /query outage; expose the last real outcome.
            "last_query": _STATE.get("last_query")}


@app.post("/query")
def query(req: QueryRequest, request: Request) -> dict:
    rid = getattr(request.state, "request_id", "")
    limiter = _STATE.get("limiter")
    retry = limiter.check(_client_key(request)) if limiter else 0.0
    if retry:
        raise HTTPException(429, "Rate limit exceeded; try again shortly.",
                            headers={"Retry-After": str(int(retry) + 1)})
    try:
        svc = _get_service()
    except Exception:
        _record(False, "startup")
        log.error("service build failed", extra={"request_id": rid}, exc_info=True)
        raise HTTPException(503, f"Backend is starting or misconfigured (request {rid}).")
    try:
        ans = svc.answer(req.question)
    except UpstreamError:
        _record(False, "model")
        log.error("generation failed", extra={"request_id": rid}, exc_info=True)
        raise HTTPException(502, f"The hosted language model is unavailable right now "
                                 f"(request {rid}). Please retry in a minute.")
    except Exception:
        # Details (DSNs, provider internals) stay in logs, never in the response.
        _record(False, "internal")
        log.error("query failed", extra={"request_id": rid}, exc_info=True)
        raise HTTPException(502, f"Query failed (request {rid}).")
    _record(True)
    return ans.model_dump()


def _record(ok: bool, stage: str | None = None) -> None:
    _STATE["last_query"] = {"ok": ok, "failed_stage": stage, "at": int(time.time())}


@app.get("/benchmark")
def benchmark() -> dict:
    if "benchmark" not in _STATE:
        raise HTTPException(404, "No benchmark data. Run scripts/rag_benchmark.py first.")
    return _STATE["benchmark"]
