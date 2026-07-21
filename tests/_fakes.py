"""Deterministic CPU fakes for tests: no model downloads, no network."""
from __future__ import annotations

import re
import zlib

import numpy as np

from src.assistant.serving.providers import GenerationResult

_W = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag-of-words -> unit vector. Deterministic across processes.

    Uses zlib.crc32 (stable) rather than Python's per-process-randomized hash(),
    so retrieval order in tests is reproducible.
    """
    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in _W.findall(text.lower()):
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documents(self, texts): return np.vstack([self._vec(t) for t in texts])
    def embed_query(self, query):    return self._vec(query)


class FakeProvider:
    """Echoes the top source's opening as a cited claim, so grounding passes.

    Without RAG context it returns a generic (ungrounded) sentence.
    """
    name = "fake"

    def generate(self, messages) -> GenerationResult:
        user = messages[-1]["content"]
        if "Sources:" in user:
            # grab the first source passage text between the [1] marker and [2]/Question
            body = user.split("[1]", 1)[1]
            body = re.split(r"\n\[2\]|\nQuestion:", body)[0]
            first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[-1].strip() or body.strip()
            text = first_sentence.rstrip(".") + " [1]."      # marker inside the sentence
        else:
            text = "This is a general response without cited evidence."
        return GenerationResult(text=text, prompt_tokens=len(user.split()),
                                completion_tokens=len(text.split()))
