#!/usr/bin/env python
"""Run the 4-way benchmark (Base / FT / Base+RAG / FT+RAG) → tables + Explorer JSON.

Builds one AssistantService per config over the same RAG index, scores them on the
eval set, and writes:
  * results/rag_benchmark.md          — comparison tables (README)
  * results/benchmark_explorer.json   — per-question data for the Gradio Explorer

Needs a GPU for the real 7B providers. Use --sample for a CPU structural check
(tiny corpus, echo provider — validates the wiring, not model quality).

Usage:
    python scripts/rag_benchmark.py --adapter outputs/qlora_5k
    python scripts/rag_benchmark.py --sample --dry-run
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
from src.assistant.agents.graph import AssistantService
from src.assistant.config import get_settings
from src.assistant.eval.benchmark import run_benchmark
from src.assistant.eval.eval_set import load_eval_set, sample_eval_set
from src.assistant.eval.report import render_markdown, write_explorer_data
from src.assistant.rag.pipeline import RAGPipeline


def _build_services(pipeline, base_provider, ft_provider, cfg):
    mk = lambda prov, label: AssistantService(pipeline, prov, label, settings=cfg)
    return {"base": mk(base_provider, "base"), "ft": mk(ft_provider, "ft"),
            "base_rag": mk(base_provider, "base_rag"), "ft_rag": mk(ft_provider, "ft_rag")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir/repo for FT configs.")
    ap.add_argument("--eval-set", default=None, help="Path to eval questions JSON.")
    ap.add_argument("--price-per-1m", type=float, default=0.0, help="$/1M tokens for cost estimate.")
    ap.add_argument("--out-md", default="results/rag_benchmark.md")
    ap.add_argument("--out-json", default="results/benchmark_explorer.json")
    ap.add_argument("--sample", action="store_true", help="Tiny offline corpus + echo provider.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = get_settings()
    questions = sample_eval_set() if (args.sample or not args.eval_set) else load_eval_set(args.eval_set)

    if args.sample:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from _fakes import FakeEmbedder, FakeProvider  # type: ignore
        from src.assistant.rag.ingest import iter_sample_documents
        from src.assistant.rag.store import LocalVectorStore

        cfg.use_reranker = False
        pipeline = RAGPipeline(cfg, store=LocalVectorStore(cfg.index_dir),
                               embedder=FakeEmbedder(), reranker=None)
        pipeline.build_index(list(iter_sample_documents()))
        base_provider = ft_provider = FakeProvider()
    else:
        from src.assistant.serving.providers import LocalTransformersProvider
        pipeline = RAGPipeline(cfg)
        base_provider = LocalTransformersProvider(cfg.base_model, max_new_tokens=cfg.max_new_tokens)
        ft_provider = LocalTransformersProvider(cfg.base_model, adapter_dir=args.adapter or cfg.adapter_repo,
                                                max_new_tokens=cfg.max_new_tokens)

    services = _build_services(pipeline, base_provider, ft_provider, cfg)
    if args.dry_run:
        print("[benchmark] configs:", list(services), "| questions:", len(questions))
        return

    results = run_benchmark(services, questions, price_per_1m=args.price_per_1m)
    from pathlib import Path
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render_markdown(results))
    write_explorer_data(results, args.out_json)
    print(render_markdown(results))
    print(f"[benchmark] wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
