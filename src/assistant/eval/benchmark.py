"""4-way benchmark runner: score each config on the eval set.

For each (config, question) it runs the AssistantService and records retrieval,
generation, and systems metrics plus the full answer/evidence/citations — the
latter powers the interactive Benchmark Explorer. Aggregates per config feed the
comparison tables.
"""
from __future__ import annotations

from ..agents.graph import AssistantService
from ..schema import GroundedAnswer
from .eval_set import EvalQuestion
from .generation_metrics import citation_coverage, groundedness, rouge_l
from .retrieval_metrics import aggregate_retrieval, recall_at_k, reciprocal_rank
from .systems_metrics import aggregate_systems, answer_total_ms

K_VALUES = (1, 3, 5)


def _record(q: EvalQuestion, ans: GroundedAnswer, uses_rag: bool) -> dict:
    gold = set(q.gold_doc_ids)
    retrieval = {}
    if uses_rag:
        for k in K_VALUES:
            retrieval[f"recall@{k}"] = recall_at_k(ans.passages, gold, k)
        retrieval["mrr"] = reciprocal_rank(ans.passages, gold)
    gen = {
        "citation_coverage": citation_coverage(ans.answer),
        "groundedness": groundedness(ans.claims),
        "rougeL_f": rouge_l(ans.answer, q.reference_answer) if q.reference_answer else None,
    }
    return {
        "id": q.id, "question": q.question, "answer": ans.answer,
        "config": ans.config,
        "retrieval": retrieval, "generation": gen,
        "total_latency_ms": answer_total_ms(ans),
        "citations": [c.model_dump() for c in ans.citations],
        "passages": [p.model_dump() for p in ans.passages],
        "all_claims_supported": ans.all_claims_supported,
    }


def run_config(service: AssistantService, questions: list[EvalQuestion],
               uses_rag: bool, price_per_1m: float = 0.0) -> dict:
    records, answers = [], []
    for q in questions:
        ans = service.answer(q.question)
        answers.append(ans)
        records.append(_record(q, ans, uses_rag))

    retrieval_agg = aggregate_retrieval([r["retrieval"] for r in records]) if uses_rag else {}
    gen_keys = ["citation_coverage", "groundedness"]
    generation_agg = {k: round(sum(r["generation"][k] for r in records) / len(records), 4)
                      for k in gen_keys}
    rouge_vals = [r["generation"]["rougeL_f"] for r in records
                  if r["generation"]["rougeL_f"] is not None]
    if rouge_vals:
        generation_agg["rougeL_f"] = round(sum(rouge_vals) / len(rouge_vals), 4)

    return {
        "config": service.config_label,
        "retrieval": retrieval_agg,
        "generation": generation_agg,
        "systems": aggregate_systems(answers, price_per_1m),
        "records": records,
    }


def run_benchmark(services: dict[str, AssistantService], questions: list[EvalQuestion],
                  price_per_1m: float = 0.0) -> dict:
    """services: {config_label: AssistantService}. Returns per-config results."""
    uses_rag = {"base": False, "ft": False, "base_rag": True, "ft_rag": True}
    return {
        "n_questions": len(questions),
        "configs": {label: run_config(svc, questions, uses_rag.get(label, True), price_per_1m)
                    for label, svc in services.items()},
    }
