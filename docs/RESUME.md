# Résumé bullets

Every claim maps to code/tests in this repo or a measured experiment. Items that
still need a GPU run or a live deploy are marked so you don't over-claim.

### Biomedical AI Research Assistant — RAG · Agents · QLoRA · Evaluation
**Stack:** Python · Qwen2.5-7B-Instruct · QLoRA (bitsandbytes) · PEFT/TRL ·
sentence-transformers · Neon Postgres + pgvector · LangGraph · FastAPI · Gradio ·
Hugging Face Hub/Inference · Docker · GitHub Actions

**Backed and ready to claim now (implemented + tested):**
- Built a modular **production RAG system** over PubMed + NIH/WHO/CDC guidelines —
  reproducible ingestion (NCBI E-utilities), sentence-aware chunking, bge-small
  embeddings, a **backend-agnostic vector store** (local numpy for dev/CI, **Neon
  pgvector** for prod), semantic retrieval with metadata filtering, and
  cross-encoder reranking.
- Implemented a **LangGraph 4-agent workflow** (Planner → Retrieval → Answer →
  **Citation-Verification**) that returns grounded, `[n]`-cited answers and flags
  every unsupported claim, with a dependency-free sequential fallback.
- Engineered a **4-way evaluation harness** (Base / Fine-tuned / Base+RAG /
  Fine-tuned+RAG) covering retrieval (Recall@k, MRR), generation (citation
  coverage, groundedness, ROUGE-L, BERTScore), and systems (p50/p95 latency,
  tokens, estimated cost) — validated end-to-end offline with **12 CI tests**.
- Fine-tuned **Qwen2.5-7B-Instruct with 4-bit QLoRA** on MedMCQA (~194K MCQs) on a
  single free T4 and evaluated with EleutherAI lm-eval-harness: in-domain MedMCQA
  **47.5% → 50.0%** (within noise — an honest null showing strong instruction
  tuning already near-saturates) and out-of-domain PubMedQA **48.0% → 64.5%**.
- **Deployed the assistant end-to-end on 100% free-tier infra** — Gradio
  ([Space](https://huggingface.co/spaces/Udit013/biomed-assistant)) → FastAPI
  ([Render](https://biomed-assistant-api.onrender.com/health)) → Neon pgvector
  (3.4K biomedical chunks) → HF Inference (Qwen2.5-7B) — serving live, cited,
  verified **Base + RAG** answers. Honestly labeled: the live config is exposed
  by the API and shown in the UI, and swaps to **Fine-tuned + RAG** via a GPU
  endpoint with **zero UI/API change**.
- **Published** the [MedMCQA QLoRA adapter](https://huggingface.co/Udit013/qwen2.5-7b-medmcqa-qlora-5k)
  to the HF Hub with a complete model card; shipped **Docker**, a free-tier
  deployment blueprint, and **GitHub Actions CI** on every push. Separated
  production code from the original research pipeline for clarity.

**Still pending — only claim after a GPU benchmark run:**
- Fill the 4-way benchmark numbers (`scripts/rag_benchmark.py` over the real
  corpus) → then: "measured Recall@k / groundedness / latency across four configs,
  showing RAG lifts groundedness from X→Y". (Adapter publish + live deploy are
  now DONE and moved above.)
