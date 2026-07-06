"""Render benchmark results into comparison tables + the Explorer JSON.

`render_markdown` produces retrieval / generation / systems tables across the 4
configs (for the README). `write_explorer_data` dumps the per-question records the
Gradio Benchmark Explorer loads so users can inspect precomputed answers,
evidence, citations, and latency without any GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_ORDER = ["base", "ft", "base_rag", "ft_rag"]
CONFIG_LABEL = {"base": "Base", "ft": "Fine-tuned", "base_rag": "Base + RAG",
                "ft_rag": "Fine-tuned + RAG"}


def _cell(v) -> str:
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def _table(title: str, rows: list[str], header: list[str], get) -> str:
    out = [f"### {title}", "| Config | " + " | ".join(header) + " |",
           "|---|" + "---|" * len(header)]
    for cfg in rows:
        out.append(f"| {CONFIG_LABEL.get(cfg, cfg)} | "
                   + " | ".join(_cell(v) for v in get(cfg)) + " |")
    return "\n".join(out) + "\n"


def render_markdown(results: dict) -> str:
    configs = results["configs"]
    present = [c for c in CONFIG_ORDER if c in configs]

    retr_hdr = ["Recall@1", "Recall@3", "Recall@5", "MRR"]
    gen_hdr = ["Citation coverage", "Groundedness", "ROUGE-L"]
    sys_hdr = ["p50 e2e (ms)", "p95 e2e (ms)", "Avg tokens", "Est. cost ($)"]

    def retr(c):
        r = configs[c]["retrieval"]
        return [r.get("recall@1"), r.get("recall@3"), r.get("recall@5"), r.get("mrr")]

    def gen(c):
        g = configs[c]["generation"]
        return [g.get("citation_coverage"), g.get("groundedness"), g.get("rougeL_f")]

    def sysm(c):
        s = configs[c]["systems"]
        return [s.get("e2e_latency_ms_p50"), s.get("e2e_latency_ms_p95"),
                s.get("avg_total_tokens"), s.get("estimated_cost_usd")]

    parts = [
        f"_Benchmark over {results['n_questions']} curated questions._\n",
        _table("Retrieval", present, retr_hdr, retr),
        _table("Generation", present, gen_hdr, gen),
        _table("Systems", present, sys_hdr, sysm),
    ]
    return "\n".join(parts)


def write_explorer_data(results: dict, path: str | Path) -> Path:
    """Per-question records (answers/evidence/citations/latency) for the demo."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    explorer = {
        "n_questions": results["n_questions"],
        "configs": {c: {"records": results["configs"][c]["records"],
                        "retrieval": results["configs"][c]["retrieval"],
                        "generation": results["configs"][c]["generation"],
                        "systems": results["configs"][c]["systems"]}
                    for c in results["configs"]},
    }
    path.write_text(json.dumps(explorer, indent=2, default=str))
    return path
