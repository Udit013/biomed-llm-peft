"""Curated evaluation questions (question + gold doc_ids + reference answer).

Gold `doc_ids` are document-level relevance labels for the retrieval metrics; the
`reference_answer` feeds ROUGE/BERTScore. `sample_eval_set()` matches the tiny
offline corpus (ingest.iter_sample_documents) so the benchmark runs in CI.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class EvalQuestion(BaseModel):
    id: str
    question: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    reference_answer: str = ""


def load_eval_set(path: str | Path) -> list[EvalQuestion]:
    data = json.loads(Path(path).read_text())
    return [EvalQuestion(**q) for q in data]


def sample_eval_set() -> list[EvalQuestion]:
    return [
        EvalQuestion(
            id="q1", question="What is the first-line treatment for type 2 diabetes?",
            gold_doc_ids=["pubmed:0001"],
            reference_answer="Metformin is the first-line pharmacologic treatment for "
                             "type 2 diabetes."),
        EvalQuestion(
            id="q2", question="What blood pressure threshold does WHO use to start "
                              "antihypertensive treatment?",
            gold_doc_ids=["who:hypertension_2021"],
            reference_answer="WHO recommends starting treatment at a systolic blood "
                             "pressure of 140 mmHg or higher."),
        EvalQuestion(
            id="q3", question="Which vaccine does the CDC recommend for adults 50 and older?",
            gold_doc_ids=["cdc:vaccine_schedule"],
            reference_answer="The CDC recommends the recombinant zoster vaccine for "
                             "adults aged 50 and older."),
    ]
