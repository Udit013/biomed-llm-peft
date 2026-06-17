"""Shared model loading for serving + cost benchmarking.

CUDA-only paths (4-bit) are import-guarded so this file imports on macOS/CPU.
"""
from __future__ import annotations


def load_model_and_tokenizer(
    base_model: str,
    adapter_dir: str | None = None,
    load_in_4bit: bool = False,
    dtype: str = "float16",
):
    """Load (model, tokenizer). Attaches a LoRA adapter if `adapter_dir` is given."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = getattr(torch, dtype)
    kwargs = {"torch_dtype": torch_dtype, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tok
