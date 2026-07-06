"""Systems metrics: latency, token usage, and estimated inference cost.

Latencies come from `GroundedAnswer.latency_ms` (per-stage timings recorded live).
Cost is an explicit ESTIMATE from token usage × a configurable price per 1M
tokens (0 for self-hosted); we report it as an estimate, never a measured bill.
"""
from __future__ import annotations

import statistics

from ..schema import GroundedAnswer


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


def answer_total_ms(ans: GroundedAnswer) -> float:
    return round(sum(ans.latency_ms.values()), 2)


def estimate_cost_usd(total_tokens: int, price_per_1m: float) -> float:
    return round(total_tokens / 1_000_000 * price_per_1m, 6)


def aggregate_systems(answers: list[GroundedAnswer], price_per_1m: float = 0.0) -> dict:
    if not answers:
        return {}
    totals = [answer_total_ms(a) for a in answers]
    retr = [a.latency_ms.get("retrieve_ms", 0.0) for a in answers]
    gen = [a.latency_ms.get("generate_ms", 0.0) for a in answers]
    tokens = [a.token_usage.get("total_tokens", 0) for a in answers]
    total_tok = sum(tokens)
    return {
        "e2e_latency_ms_p50": _pct(totals, 0.5),
        "e2e_latency_ms_p95": _pct(totals, 0.95),
        "retrieval_latency_ms_p50": _pct(retr, 0.5),
        "generation_latency_ms_p50": _pct(gen, 0.5),
        "avg_total_tokens": round(total_tok / len(answers), 1),
        "total_tokens": total_tok,
        "estimated_cost_usd": estimate_cost_usd(total_tok, price_per_1m),
        "price_per_1m_tokens_usd": price_per_1m,
    }
