# Deployment (free tier)

Architecture: **Neon** (pgvector) ← **FastAPI on Render** (backend) ← **Gradio on
HF Spaces** (frontend). The 7B model is served via **HF Inference** so no GPU is
needed at serving time. Everything below fits free tiers.

```
HF Space (Gradio)  --/query-->  Render (FastAPI)  --SQL-->  Neon (pgvector)
   Ask + Explorer               RAG + agents            biomedical chunks
                                       |
        free tier: HF Inference (BASE) → "Base + RAG"   (swap to a GPU
        endpoint with the adapter for live "Fine-tuned + RAG", no UI change)
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

## Swapping in a GPU-backed Fine-tuned + RAG endpoint (later, no UI/API change)

The live config is driven entirely by env — the API returns the config it
*actually* served (`/health.served_config`, `/query.config`) and the UI displays
that verbatim. To upgrade the live demo from **Base + RAG** to **Fine-tuned + RAG**:

1. Run the backend where a GPU is available (e.g. a Colab session) with:
   ```bash
   export BIOMED_INFERENCE_PROVIDER=local          # serve base + LoRA adapter
   export BIOMED_EMBEDDING_PROVIDER=local          # local embeddings
   export BIOMED_VECTOR_BACKEND=pgvector BIOMED_DATABASE_URL=...neon...
   uvicorn src.assistant.api.app:app --host 0.0.0.0 --port 8000
   ```
2. Expose it (e.g. `cloudflared tunnel --url http://localhost:8000`) and set the
   Space's `BACKEND_URL` to that URL.

The Space's badge then automatically reads `Fine-tuned + RAG` — no code, UI, or API
changes. Point `BACKEND_URL` back to Render to return to the always-on Base + RAG.

## Cold-start note
The Render free instance sleeps when idle; the first request wakes it (~30–60 s),
and the first `/query` also warms the model path. The Benchmark Explorer is fully
precomputed, so it stays instant regardless.
