---
title: Biomedical AI Research Assistant
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Biomedical AI Research Assistant

Grounded, cited biomedical question answering over PubMed abstracts + NIH/WHO/CDC
guidelines, built on `Qwen2.5-7B-Instruct` + a MedMCQA QLoRA adapter.

- **Ask** — live **Fine-tuned + RAG** answers (Planner → Retrieval → Answer →
  Citation-Verification) via the FastAPI backend. Set the `BACKEND_URL` Space
  variable to your Render backend URL.
- **Benchmark Explorer** — precomputed **4-way** comparison (Base / Fine-tuned /
  Base+RAG / Fine-tuned+RAG) from bundled `benchmark_explorer.json`; runs with no
  GPU or backend.

**⚠️ Research/education only — not medical advice.** See `../DEPLOY.md` for setup.
