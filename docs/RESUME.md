# Résumé entry

Every claim below is backed by deployed infrastructure, published artifacts, or
committed + CI-tested code in this repository.

---

### Biomedical AI Research Assistant
**Live:** [huggingface.co/spaces/Udit013/biomed-assistant](https://huggingface.co/spaces/Udit013/biomed-assistant)
**Model:** [Udit013/qwen2.5-7b-medmcqa-qlora-5k](https://huggingface.co/Udit013/qwen2.5-7b-medmcqa-qlora-5k)
**Code:** [Udit013/biomed-llm-peft](https://github.com/Udit013/biomed-llm-peft)
**Stack:** Python · PyTorch · Transformers · PEFT/QLoRA · LangGraph · FastAPI · Gradio · PostgreSQL + pgvector (Neon) · Hugging Face Hub/Inference · Docker · GitHub Actions
**Description:** A production-grade Biomedical AI Research Assistant that answers clinical/research questions with grounded, cited evidence — retrieval-augmented generation over PubMed abstracts and NIH/WHO/CDC guidelines, a LangGraph multi-agent workflow with per-claim citation verification, and a 4-way evaluation harness (Base / Fine-tuned / Base + RAG / Fine-tuned + RAG) — built on a QLoRA-fine-tuned Qwen2.5-7B and deployed end-to-end on 100% free-tier infrastructure.
**Bullets:**
- Fine-tuned **Qwen2.5-7B-Instruct with 4-bit QLoRA** (only **0.92%** of parameters trainable) on **MedMCQA (~194K** medical MCQs**)** and evaluated with **EleutherAI lm-evaluation-harness**, measuring in-domain MedMCQA **47.5% → 50.0%** (a within-noise null showing strong instruction tuning already near-saturates the task) and out-of-domain PubMedQA **48.0% → 64.5%**; **published** the adapter to the Hugging Face Hub with a full model card.
- Built a **production RAG pipeline** over biomedical literature — reproducible ingestion (NCBI E-utilities), sentence-aware chunking, `bge-small` embeddings, semantic retrieval with metadata filtering, cross-encoder reranking, and inline citation generation — indexing **733 PubMed abstracts into 3,410 vector chunks** in **Neon PostgreSQL + pgvector**.
- Implemented a **LangGraph 4-agent workflow** (Planner → Retrieval → Answer → **Citation-Verification**) that returns grounded, `[n]`-cited answers and flags every factual claim as supported or unsupported, with a dependency-free sequential fallback for testing.
- **Deployed the system end-to-end on 100% free-tier infrastructure** (Gradio Space → FastAPI on Render → Neon pgvector → HF Inference) serving live, cited **Base + RAG** answers at **~4 s** end-to-end; the API returns the exact config it served and the UI displays it, so a **swap to Fine-tuned + RAG via a GPU endpoint requires zero UI or API change**.
- Engineered a **backend-agnostic, torch-free serving path** for the free tier — a vector-store abstraction (local NumPy for dev/CI, Neon pgvector for prod), local **ONNX query embeddings** (fastembed), and a router-based HF Inference LLM — small enough to run on a 512 MB instance.
- Designed a **4-way evaluation harness** comparing Base / Fine-tuned / Base + RAG / Fine-tuned + RAG across **retrieval** (Recall@k, MRR), **generation** (citation coverage, groundedness, ROUGE-L, BERTScore), and **systems** (p50/p95 latency, token usage, estimated cost), rendering comparison tables and an interactive Benchmark Explorer.
- Established **production engineering rigor** — a modular package cleanly separating the research pipeline from the production system, pinned dependencies, a Dockerized backend, structured JSON logging with per-stage latency, and **GitHub Actions CI running a 12-test suite** on every push.
