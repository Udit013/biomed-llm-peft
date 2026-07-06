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
    def __init__(self, model: str, token: str | None = None, max_new_tokens: int = 512):
        self.model = model
        self.token = token
        self.max_new_tokens = max_new_tokens
        self.name = "hf_inference"
        self._client = None

    def _get_client(self):
        if self._client is None:
            from huggingface_hub import InferenceClient

            self._client = InferenceClient(model=self.model, token=self.token)
        return self._client

    def generate(self, messages: list[dict]) -> GenerationResult:
        resp = self._get_client().chat_completion(
            messages=messages, max_tokens=self.max_new_tokens, temperature=0.0)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            text.strip(),
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )
