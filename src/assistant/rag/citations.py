"""Citation construction and lexical grounding checks.

`build_citations` turns the top passages into numbered `[n]` citations the answer
can reference. `verify_claims` provides a dependency-free grounding signal:
for each answer sentence, the maximum token-overlap (Jaccard) against any cited
passage. The LangGraph citation-verification agent (Phase 2) can swap this for an
NLI/embedding check, but this gives a real, deterministic baseline now.
"""
from __future__ import annotations

import re

from ..schema import Citation, RetrievedPassage

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def build_citations(passages: list[RetrievedPassage]) -> list[Citation]:
    citations: list[Citation] = []
    for i, p in enumerate(passages, start=1):
        c = p.chunk
        citations.append(Citation(
            marker=f"[{i}]", doc_id=c.doc_id, source=c.source, title=c.title,
            url=c.url, quote=c.text[:400],
        ))
    return citations


def _sentence_support(sentence: str, passages: list[RetrievedPassage]) -> tuple[float, int]:
    """Return (best Jaccard overlap, index of best passage) for a sentence."""
    s_tokens = _tokens(sentence)
    if not s_tokens:
        return 0.0, -1
    best, best_i = 0.0, -1
    for i, p in enumerate(passages):
        p_tokens = _tokens(p.chunk.text)
        inter = len(s_tokens & p_tokens)
        jacc = inter / len(s_tokens)          # coverage of the claim by the passage
        if jacc > best:
            best, best_i = jacc, i
    return best, best_i


def verify_claims(answer: str, passages: list[RetrievedPassage],
                  threshold: float = 0.35) -> tuple[list[dict], bool]:
    """Per-sentence grounding. Returns (claims, all_supported)."""
    from ..schema import Citation  # noqa (kept local to avoid cycle confusion)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    claims, all_ok = [], True
    for sent in sentences:
        if not _tokens(sent):        # marker-only / punctuation fragment, not a claim
            continue
        score, idx = _sentence_support(sent, passages)
        supported = score >= threshold
        all_ok = all_ok and supported
        claims.append({
            "claim": sent,
            "supported": supported,
            "support_score": round(score, 3),
            "passage_index": idx,
        })
    return claims, (all_ok if claims else False)


def _claim_sentences(answer: str) -> list[str]:
    """Answer sentences that carry an actual claim (skip marker-only fragments)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip())
            if s.strip() and _tokens(s)]


def verify_claims_semantic(answer: str, passages: list[RetrievedPassage],
                           embedder, threshold: float = 0.6) -> tuple[list[dict], bool]:
    """Embedding-based grounding: max cosine of each claim vs any retrieved passage.

    Far more robust than lexical overlap — a correctly *paraphrased* claim still
    scores high, so faithful answers aren't falsely flagged unsupported. `embedder`
    must return L2-normalized vectors (so cosine == dot product).
    """
    import numpy as np

    sentences = _claim_sentences(answer)
    if not sentences or not passages:
        return [], False
    p_vecs = np.asarray(embedder.embed_documents([p.chunk.text for p in passages]))
    s_vecs = np.asarray(embedder.embed_documents(sentences))
    claims, all_ok = [], True
    for i, sent in enumerate(sentences):
        sims = p_vecs @ s_vecs[i]
        best_i = int(np.argmax(sims))
        best = float(sims[best_i])
        supported = best >= threshold
        all_ok = all_ok and supported
        claims.append({"claim": sent, "supported": supported,
                       "support_score": round(best, 3), "passage_index": best_i})
    return claims, (all_ok if claims else False)
