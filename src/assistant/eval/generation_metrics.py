"""Generation metrics.

Always available (no heavy deps):
  * citation_coverage — fraction of answer sentences carrying a [n] marker.
  * groundedness      — fraction of answer claims lexically supported by the
                        retrieved evidence (from the verification agent).
Optional (lazy, return None if the library is missing):
  * rougeL_f          — vs a gold reference answer (rouge-score).
  * bertscore_f1      — semantic similarity to the reference (bert-score; heavy).
"""
from __future__ import annotations

import re

_CITE = re.compile(r"\[\d+\]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def citation_coverage(answer: str) -> float:
    sentences = [s for s in _SENT.split(answer.strip()) if re.search(r"[a-zA-Z]", s)]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if _CITE.search(s))
    return round(cited / len(sentences), 4)


def groundedness(claims: list[dict]) -> float:
    if not claims:
        return 0.0
    return round(sum(1 for c in claims if c.get("supported")) / len(claims), 4)


def rouge_l(prediction: str, reference: str) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)


def bertscore_f1(predictions: list[str], references: list[str]) -> float | None:
    try:
        from bert_score import score
    except ImportError:
        return None
    _, _, f1 = score(predictions, references, lang="en", verbose=False)
    return round(float(f1.mean()), 4)
