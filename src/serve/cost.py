"""Inference-cost benchmarking across serving configs.

Measures REAL numbers only (tokens/sec, p50/p95 latency, peak VRAM). Accuracy is
pulled from the matching lm-eval run, not re-measured here. Configs:
  * base          : base model, fp16, no adapter
  * lora_fp16     : base fp16 + LoRA adapter attached
  * quantized     : base 4-bit (bnb nf4) + LoRA adapter
  * vllm          : CONDITIONAL — skipped (recorded "not feasible") unless the
                    GPU is Ampere+ (compute capability >= 8.0) and vllm imports.

On a T4 the quantized-vs-fp16 comparison is the must-have; vLLM degrades to a
"not feasible on T4 — PENDING on higher-end GPU" row, never a fabricated number.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from ..utils.env import get_gpu_info

# A few representative MedMCQA-style prompts for timing (content-agnostic).
BENCH_PROMPTS = [
    "Question: Which vitamin deficiency causes scurvy?\nA. A\nB. C\nC. D\nD. K\nAnswer:",
    "Question: The most common cause of bacterial meningitis in neonates?\n"
    "A. S. pneumoniae\nB. Group B Streptococcus\nC. N. meningitidis\nD. H. influenzae\nAnswer:",
]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def benchmark_hf(model, tokenizer, max_new_tokens: int = 64, n_iters: int = 20) -> dict:
    """Time greedy generation on the benchmark prompts; return latency/throughput/VRAM."""
    import torch

    device = next(model.parameters()).device
    latencies: list[float] = []
    total_new_tokens = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # warmup
    warm = tokenizer(BENCH_PROMPTS[0], return_tensors="pt").to(device)
    model.generate(**warm, max_new_tokens=8, do_sample=False)

    for i in range(n_iters):
        prompt = BENCH_PROMPTS[i % len(BENCH_PROMPTS)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        latencies.append(dt)
        total_new_tokens += out.shape[1] - inputs["input_ids"].shape[1]

    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else None)
    return {
        "tokens_per_sec": round(total_new_tokens / sum(latencies), 2),
        "p50_latency_s": round(statistics.median(latencies), 4),
        "p95_latency_s": round(_percentile(latencies, 0.95), 4),
        "peak_vram_gb": round(vram_gb, 2) if vram_gb else None,
        "n_iters": n_iters,
        "max_new_tokens": max_new_tokens,
    }


def vllm_feasible() -> tuple[bool, str]:
    gpu = get_gpu_info()
    if not gpu.cuda_available:
        return False, "no CUDA device"
    if not gpu.vllm_feasible:
        return False, (f"not feasible on {gpu.device_name} (compute "
                       f"{gpu.compute_capability}) — PENDING on higher-end GPU")
    try:
        import vllm  # noqa: F401
    except ImportError:
        return False, "vllm not installed — PENDING on higher-end GPU"
    return True, "feasible"


COST_COLUMNS = ["Accuracy", "Tokens/sec", "p50 latency (s)", "p95 latency (s)", "VRAM (GB)"]
COST_ROWS = ["Base (fp16)", "LoRA fp16", "Quantized (bnb 4-bit)", "vLLM"]


def render_cost_table(path: str | Path = "results/inference_cost.json") -> str:
    """Render the cost table from a measured JSON file (PENDING where absent)."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    lines = ["| Config | " + " | ".join(COST_COLUMNS) + " |",
             "|---|" + "---|" * len(COST_COLUMNS)]
    key_map = {"Base (fp16)": "base", "LoRA fp16": "lora_fp16",
               "Quantized (bnb 4-bit)": "quantized", "vLLM": "vllm"}
    for row in COST_ROWS:
        rec = data.get(key_map[row], {})
        if rec.get("status") == "not_feasible":
            lines.append(f"| {row} | {rec.get('reason', 'not feasible')} "
                         + "| — " * len(COST_COLUMNS) + "|")
            continue
        cells = [
            _fmt(rec.get("accuracy")),
            _fmt(rec.get("tokens_per_sec")),
            _fmt(rec.get("p50_latency_s")),
            _fmt(rec.get("p95_latency_s")),
            _fmt(rec.get("peak_vram_gb")),
        ]
        lines.append(f"| {row} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _fmt(v) -> str:
    return "PENDING RUN" if v is None else str(v)
