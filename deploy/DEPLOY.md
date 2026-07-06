# Deployment (free tier)

Architecture: **Neon** (pgvector) ← **FastAPI on Render** (backend) ← **Gradio on
HF Spaces** (frontend). The 7B model is served via **HF Inference** so no GPU is
needed at serving time. Everything below fits free tiers.

```
HF Space (Gradio)  --/query-->  Render (FastAPI)  --SQL-->  Neon (pgvector)
   Ask + Explorer                 FT+RAG agents         biomedical chunks
                                       |
                                   HF Inference (Qwen2.5-7B + adapter)
```

## 1. Vector DB — Neon + pgvector
1. Create a free project at https://neon.tech → copy the connection string (DSN).
2. (Optional) apply `deploy/neon_schema.sql` — the app also creates it on first use.

## 2. Build + push the index (from a machine with the embedding model)
```bash
pip install -r requirements-assistant.txt
export BIOMED_VECTOR_BACKEND=pgvector
export BIOMED_DATABASE_URL="postgres://...neon..."
export BIOMED_NCBI_EMAIL="you@example.com"       # polite NCBI header
python scripts/rag_index.py --config configs/corpus.yaml
```

## 3. Precompute the benchmark (on a GPU session)
```bash
python scripts/rag_benchmark.py --adapter Udit013/qwen2.5-7b-medmcqa-qlora-5k
# -> results/rag_benchmark.md + results/benchmark_explorer.json
```
Commit `results/benchmark_explorer.json` and copy it into `deploy/space/` so the
Explorer works with no backend.

## 4. Backend — FastAPI on Render
1. Push this repo to GitHub (already at `Udit013/biomed-llm-peft`).
2. Render → New → Blueprint → point at `deploy/render.yaml`.
3. Set secrets in the dashboard: `BIOMED_DATABASE_URL` (Neon DSN),
   `BIOMED_HF_TOKEN` (HF token with Inference access).
4. Deploy → note the service URL, e.g. `https://biomed-assistant-api.onrender.com`.
   Check `GET /health`.

## 5. Frontend — Gradio on HF Spaces
```bash
huggingface-cli repo create biomed-assistant --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<you>/biomed-assistant space_deploy
cp deploy/space/app.py deploy/space/requirements.txt deploy/space/README.md space_deploy/
cp results/benchmark_explorer.json space_deploy/
cd space_deploy && git add . && git commit -m "Deploy assistant" && git push
```
In the Space **Settings → Variables**: set `BACKEND_URL` to your Render URL.

## Cold-start note
The Render free instance sleeps when idle; the first request wakes it (~30–60 s),
and the first `/query` also warms the model path. The Benchmark Explorer is fully
precomputed, so it stays instant regardless.
