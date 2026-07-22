# Résumé entry

Every claim below is backed by deployed infrastructure, published artifacts, or
committed + CI-tested code in this repository.

---

### Biomedical AI Research Assistant
**Live:** [huggingface.co/spaces/Udit013/biomed-assistant](https://huggingface.co/spaces/Udit013/biomed-assistant)
**Model:** [Udit013/qwen2.5-7b-medmcqa-qlora-5k](https://huggingface.co/Udit013/qwen2.5-7b-medmcqa-qlora-5k)
**Code:** [Udit013/biomed-llm-peft](https://github.com/Udit013/biomed-llm-peft)
**Stack:** Python · PyTorch · Transformers · PEFT/QLoRA · LangGraph · FastAPI · Gradio · PostgreSQL + pgvector (Neon) · Hugging Face Hub/Inference · Docker · GitHub Actions
**Description:** Biomedical research assistant that answers clinical questions with grounded, cited evidence — RAG over PubMed abstracts and NIH/WHO/CDC guidelines, a LangGraph multi-agent workflow with per-claim citation verification, and a 4-way evaluation harness (Base / Fine-tuned / Base + RAG / Fine-tuned + RAG) — built on a QLoRA-fine-tuned Qwen2.5-7B and deployed end to end on free-tier infrastructure.
**Bullets:**
- Fine-tuned **Qwen2.5-7B-Instruct with 4-bit QLoRA** (**0.92%** of parameters trainable) on **MedMCQA (~194K MCQs)** and evaluated with **EleutherAI lm-evaluation-harness**: in-domain MedMCQA **47.5% → 50.0%** (a within-noise null, since instruction tuning already near-saturates the task) and out-of-domain PubMedQA **48.0% → 64.5%**; published the adapter to the Hugging Face Hub with a full model card
- Built a RAG pipeline over biomedical literature — reproducible ingestion (NCBI E-utilities), sentence-aware chunking, `bge-small` embeddings, semantic retrieval with metadata filtering, cross-encoder reranking, and inline citations — indexing **733 PubMed abstracts into 3,410 vector chunks** in **Neon PostgreSQL + pgvector**
- Implemented a **LangGraph 4-agent workflow** (Planner → Retrieval → Answer → Citation-Verification) returning grounded, `[n]`-cited answers with **semantic (embedding-based) per-claim verification** that flags any unsupported claim, with a dependency-free sequential fallback for testing
- Deployed end to end on **free-tier infrastructure** (Gradio Space → FastAPI on Render → Neon pgvector → HF Inference), serving live cited **Base + RAG** answers at **~4 s**; the API reports the exact config it served, so a swap to **Fine-tuned + RAG** via a GPU endpoint needs zero UI or API change
- Engineered a torch-free serving path for the free tier — a vector-store abstraction (local NumPy for dev/CI, Neon pgvector for prod), local **ONNX query embeddings** (fastembed), and a router-based HF Inference LLM — small enough for a 512 MB instance
- Designed a **4-way evaluation harness** across retrieval (Recall@k, MRR), generation (citation coverage, groundedness, ROUGE-L, BERTScore), and systems (p50/p95 latency, token usage, estimated cost), with comparison tables and an interactive Benchmark Explorer
- Separated the research pipeline from the production system in a modular package, with pinned dependencies, a Dockerized backend, structured JSON logging with request tracing, and **GitHub Actions CI running a 13-test suite** on every push
