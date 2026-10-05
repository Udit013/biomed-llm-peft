"""Answer-quality fixes: marker normalization, abstention handling, passage
dedupe, and stored-vector reuse in semantic verification. CPU-only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _fakes import FakeEmbedder
from src.assistant.rag.citations import (dedupe_passages, is_abstention,
                                         normalize_markers, verify_claims_semantic)
from src.assistant.schema import Chunk, GroundedAnswer, RetrievedPassage


def _p(text, doc="d", emb=None):
    return RetrievedPassage(chunk=Chunk(chunk_id=doc + text[:5], doc_id=doc, source="pubmed",
                                        title="t", text=text, ordinal=0), score=0.9,
                            embedding=emb)


def test_normalize_markers_expands_ranges_and_lists():
    assert normalize_markers("x [1-3].") == "x [1][2][3]."
    assert normalize_markers("x [1, 3] y [2–4]") == "x [1][3] y [2][3][4]"
    assert normalize_markers("x [2-9]", n_sources=4) == "x [2][3][4]"   # clamp to sources
    assert normalize_markers("x [2] and 1-3 mg") == "x [2] and 1-3 mg"  # untouched


def test_abstention_is_not_a_claim():
    assert is_abstention("The provided sources do not contain enough evidence.")
    assert not is_abstention("Sepsis causes tachycardia and fever [1].")
    emb = FakeEmbedder()
    claims, ok = verify_claims_semantic(
        "The provided sources do not contain enough evidence.", [_p("sepsis text")], emb)
    assert claims == [] and ok is False


def test_dedupe_drops_republished_abstracts():
    same = "Sepsis is a common and life-threatening condition requiring early recognition."
    out = dedupe_passages([_p(same, "a"), _p(same, "b"), _p("Different abstract.", "c"),
                           _p(same.upper(), "d")])
    assert [p.chunk.doc_id for p in out] == ["a", "c"] and [p.rank for p in out] == [1, 2]


def test_semantic_verify_reuses_stored_vectors():
    class Counting(FakeEmbedder):
        calls: list = []
        def embed_documents(self, texts):
            self.calls.append(len(texts)); return super().embed_documents(texts)
    emb = Counting()
    text = "Metformin lowers hepatic glucose production."
    passages = [_p(text, emb=emb._vec(text).tolist())]
    claims, ok = verify_claims_semantic(text + " [1].", passages, emb)
    assert ok and emb.calls == [1]            # only the claim was embedded


def test_embeddings_never_serialized_in_api_payload():
    ans = GroundedAnswer(query="q", answer="a", passages=[_p("x", emb=[0.1] * 4)])
    assert "embedding" not in ans.model_dump()["passages"][0]
    assert "embedding" not in ans.model_dump_json()


def test_stray_abstention_removed_only_when_answer_has_cited_claims():
    from src.assistant.rag.citations import strip_stray_abstention
    mixed = ("DOACs need no routine monitoring [5]. They cause less intracranial "
             "bleeding [3]. The provided sources do not contain enough evidence.")
    assert strip_stray_abstention(mixed) == ("DOACs need no routine monitoring [5]. "
                                             "They cause less intracranial bleeding [3].")
    pure = "The provided sources do not contain enough evidence."
    assert strip_stray_abstention(pure) == pure
    uncited = "Some general statement. The provided sources do not contain enough evidence."
    assert strip_stray_abstention(uncited) == uncited     # no cited claim: keep the refusal
