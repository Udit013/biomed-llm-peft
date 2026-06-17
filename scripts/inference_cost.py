#!/usr/bin/env python
"""Benchmark inference cost across serving configs -> results/inference_cost.json.

Measures real tokens/sec, p50/p95 latency, and peak VRAM. vLLM is conditional:
on a T4 it records a "not feasible" row rather than a fabricated number.

Usage (on GPU):
    python scripts/inference_cost.py --adapter outputs/qlora_5k \
        --configs base lora_fp16 quantized vllm

Accuracy cells are left None here; fill them from the matching lm-eval runs.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from src.serve.cost import benchmark_hf, render_cost_table, vllm_feasible
from src.serve.loader import load_model_and_tokenizer


def _free():
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--configs", nargs="+",
                    default=["base", "lora_fp16", "quantized", "vllm"])
    ap.add_argument("--out", default="results/inference_cost.json")
    ap.add_argument("--n-iters", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}

    plans = {
        "base": dict(adapter=None, load_in_4bit=False),
        "lora_fp16": dict(adapter=args.adapter, load_in_4bit=False),
        "quantized": dict(adapter=args.adapter, load_in_4bit=True),
    }

    for name in args.configs:
        if name == "vllm":
            ok, reason = vllm_feasible()
            if not ok:
                results["vllm"] = {"status": "not_feasible", "reason": reason}
                print(f"[cost] vllm: {reason}")
                continue
            print("[cost] vLLM feasible on this GPU — benchmark it with your vLLM "
                  "serving script and merge the row into the JSON.")
            continue

        plan = plans[name]
        print(f"[cost] benchmarking '{name}' ...")
        model, tok = load_model_and_tokenizer(
            args.base_model, adapter_dir=plan["adapter"],
            load_in_4bit=plan["load_in_4bit"],
        )
        results[name] = {"status": "ok", **benchmark_hf(model, tok, n_iters=args.n_iters)}
        del model
        _free()

    out.write_text(json.dumps(results, indent=2))
    print(f"[cost] wrote {out}\n")
    print(render_cost_table(out))


if __name__ == "__main__":
    main()
