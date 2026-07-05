"""Biomedical AI Research Assistant — production RAG + agent package.

This package is the PRODUCTION system layered on top of the original QLoRA
research pipeline (src/data, src/train, src/eval, src/serve, src/utils). The
research/experiment code is preserved untouched; everything under
`src.assistant` is the new retrieval-augmented, agentic serving system.
"""

__all__ = ["config", "logging", "schema"]
