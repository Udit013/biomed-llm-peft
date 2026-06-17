"""FastAPI endpoint: base model + LoRA adapter, MedMCQA-style MCQ answering.

Returns the chosen letter plus normalized, UNCALIBRATED option probabilities
(see src/serve/score.py). Model location is configured via env vars so the same
image serves base or any adapter:

    BASE_MODEL   (default Qwen/Qwen2.5-7B-Instruct)
    ADAPTER_DIR  (path or HF Hub repo id; omit to serve the base model)
    LOAD_IN_4BIT (default "true")

Run:  uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

NOT FOR CLINICAL USE — research/education only.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

_STATE: dict = {}


class MCQRequest(BaseModel):
    question: str = Field(..., examples=["Which vitamin deficiency causes scurvy?"])
    options: list[str] = Field(..., min_length=2, max_length=4,
                               examples=[["Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"]])


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .loader import load_model_and_tokenizer

    base = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    adapter = os.environ.get("ADAPTER_DIR") or None
    load_4bit = os.environ.get("LOAD_IN_4BIT", "true").lower() == "true"
    print(f"[api] loading base={base} adapter={adapter} 4bit={load_4bit}")
    model, tok = load_model_and_tokenizer(base, adapter_dir=adapter, load_in_4bit=load_4bit)
    _STATE.update(model=model, tok=tok, base=base, adapter=adapter)
    yield
    _STATE.clear()


app = FastAPI(
    title="Biomedical MCQ — QLoRA serving",
    description="Base Qwen2.5-7B-Instruct + MedMCQA LoRA adapter. NOT for clinical use.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "base": _STATE.get("base"), "adapter": _STATE.get("adapter")}


@app.post("/predict")
def predict(req: MCQRequest) -> dict:
    from .score import score_options

    return score_options(_STATE["model"], _STATE["tok"], req.question, req.options)
