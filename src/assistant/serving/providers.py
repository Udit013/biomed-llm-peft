"""LLM provider abstraction: one `generate(messages) -> GenerationResult` API,
several backends. Heavy deps are lazy so the module imports on CPU.

  * LocalTransformersProvider — base Qwen2.5-7B (+ optional LoRA adapter) via the
    research pipeline's loader (`src/serve/loader.py`). Used on a GPU session.
  * HFInferenceProvider       — Hugging Face Inference API (serverless / hosted),
    for the free-tier live demo where a 7B can't run on CPU.

Tests/eval inject their own duck-typed provider (see tests/_fakes.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..logging import get_logger

log = get_logger(__name__)


class UpstreamError(RuntimeError):
    """The hosted model could not produce an answer (all providers failed)."""


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def generate(self, messages: list[dict]) -> GenerationResult: ...


class LocalTransformersProvider(LLMProvider):
    def __init__(self, base_model: str, adapter_dir: str | None = None,
                 load_in_4bit: bool = True, max_new_tokens: int = 512):
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.name = "ft" if adapter_dir else "base"
        self._model = self._tok = None

    def _load(self):
        if self._model is None:
            from ...serve.loader import load_model_and_tokenizer  # reuse research loader

            self._model, self._tok = load_model_and_tokenizer(
                self.base_model, adapter_dir=self.adapter_dir,
                load_in_4bit=self.load_in_4bit)
        return self._model, self._tok

    def generate(self, messages: list[dict]) -> GenerationResult:
        import torch

        model, tok = self._load()
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=self.max_new_tokens,
                                 do_sample=False)
        completion = out[0, input_ids.shape[1]:]
        text = tok.decode(completion, skip_special_tokens=True).strip()
        return GenerationResult(text, int(input_ids.shape[1]), int(completion.shape[0]))


class HFInferenceProvider(LLMProvider):
    """HF Inference router with an explicit, ordered provider list.

    `auto` routing can pick a provider whose mapping for the model is broken (in
    Sep 2026 it kept routing Qwen2.5-7B to Together after Together dropped it, so
    every query 502'd while /health stayed green). Pinning an ordered list and
    falling through on failure keeps the demo up when one provider disappears.
    """

    def __init__(self, model: str, token: str | None = None, max_new_tokens: int = 512,
                 providers: list[str] | None = None, timeout: float = 60.0):
        self.model = model
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.providers = providers or ["auto"]
        self.timeout = timeout
        self.name = "hf_inference"
        self._clients: dict = {}

    def _get_client(self, provider: str):
        if provider not in self._clients:
            from huggingface_hub import InferenceClient

            self._clients[provider] = InferenceClient(
                model=self.model, token=self.token, provider=provider,
                timeout=self.timeout)
        return self._clients[provider]

    def generate(self, messages: list[dict]) -> GenerationResult:
        errors = []
        for provider in self.providers:
            try:
                resp = self._get_client(provider).chat_completion(
                    messages=messages, max_tokens=self.max_new_tokens, temperature=0.0)
            except Exception as exc:  # provider down / model unmapped -> try next
                errors.append(f"{provider}: {type(exc).__name__}")
                log.warning("inference provider failed",
                            extra={"provider": provider, "error": str(exc)[:300]})
                continue
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            return GenerationResult(
                text.strip(),
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            )
        raise UpstreamError("all inference providers failed (" + "; ".join(errors) + ")")
