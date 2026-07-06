# Architecture

Two clearly-separated layers in one repo:

- **Part 1 — Research pipeline** (`src/data`, `src/train`, `src/eval`, `src/serve`,
  `src/utils`, `configs/`, `lm_eval_tasks/`, `notebooks/`): the original QLoRA
  fine-tuning + lm-eval study. **Preserved unchanged.**
- **Part 2 — Production assistant** (`src/assistant/`): the retrieval-augmented,
  agentic serving system built on top of the fine-tuned adapter.

## Production system (`src/assistant/`)

```
                         ┌──────────────────────── HF Space (Gradio) ────────────────────────┐
                         │  Ask (live FT+RAG)          Benchmark Explorer (precomputed 4-way) │
                         └───────────────┬───────────────────────────┬───────────────────────┘
                                         │ POST /query               │ benchmark_explorer.json
                                         ▼                           ▼
                         ┌──────────── FastAPI (Render) ────────────┐
                         │  AssistantService (config = ft_rag)      │
                         │                                          │
                         │  LangGraph:  Planner ─► Retrieval ─►     │
                         │              Answer  ─► CitationVerify   │
                         └───────┬───────────────────┬──────────────┘
                                 │                   │
                   ┌─────────────▼──────┐   ┌────────▼─────────────┐
                   │ RAG pipeline       │   │ LLM provider          │
                   │ embed→store→rerank │   │ HF Inference / local  │
                   └─────────┬──────────┘   │ (Qwen2.5-7B + adapter)│
                             │              └───────────────────────┘
                   ┌─────────▼──────────┐
                   │ Neon Postgres      │
                   │ + pgvector index   │
                   └────────────────────┘
```

### Data flow
1. **Ingest** (`rag/ingest.py`) — PubMed abstracts (NCBI E-utilities) + NIH/WHO/CDC
   guideline files → normalized `Document`s.
2. **Index** (`rag/{chunk,embed,store}.py`) — sentence-aware chunks → bge-small
   embeddings → pgvector (prod) or local numpy (dev/CI).
3. **Serve a query** — the LangGraph agents run Planner (strategy + metadata
   filter) → Retrieval (semantic + rerank) → Answer (grounded, cited generation)
   → Citation-Verification (per-claim lexical grounding), returning a typed
   `GroundedAnswer` with citations, evidence, verification, latency, and tokens.

### Evaluation (`src/assistant/eval/`)
The 4 configs (Base / FT / Base+RAG / FT+RAG) are scored offline on a curated eval
set: retrieval (Recall@k, MRR), generation (citation coverage, groundedness,
ROUGE-L, BERTScore), systems (latency, tokens, estimated cost). Output feeds the
README tables and the interactive Benchmark Explorer.

### Design choices
- **Backend-agnostic vector store** — same interface for local numpy and Neon
  pgvector, so dev/CI need no external services.
- **Framework-agnostic agents** — pure `(state, deps) -> state` functions,
  orchestrated by LangGraph in prod and a sequential loop in tests.
- **Free-tier serving** — the 7B never runs on free CPU; live answers use HF
  Inference, and the 4-way comparison is precomputed (Explorer), keeping the demo
  responsive and honest.
