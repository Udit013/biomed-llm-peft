# Deploying the Gradio Space

This folder is self-contained and is pushed to a Hugging Face Space as its own
git repo (separate from the main project repo).

## Where the adapter loads from

The app loads the **base model** (`Qwen/Qwen2.5-7B-Instruct`) and attaches the
**LoRA adapter from the Hugging Face Hub** at runtime. The adapter repo id is
read from the `ADAPTER_REPO` environment variable (a Space *variable*). It is the
repo you created with `scripts/push_to_hub.py`, e.g.
`your-username/qwen2.5-7b-medmcqa-qlora-5k`.

## 1. Create the Space

Via the website: https://huggingface.co/new-space
- **Owner / name:** `your-username/biomed-mcqa-qlora`
- **SDK:** Gradio
- **Hardware:** CPU basic works (slow cold start) — a small GPU is far better for
  a 7B model. Pick hardware deliberately; document the cold-start tradeoff.

Or via CLI:

```bash
pip install huggingface_hub
huggingface-cli login                      # paste an HF token with write access
huggingface-cli repo create biomed-mcqa-qlora --type space --space_sdk gradio
```

## 2. Configure the adapter source

In the Space **Settings → Variables and secrets**, add:
- Variable `ADAPTER_REPO = your-username/qwen2.5-7b-medmcqa-qlora-5k`
- (optional) Variable `BASE_MODEL` if you change the base.

## 3. Push the app

```bash
# from the project root
git clone https://huggingface.co/spaces/your-username/biomed-mcqa-qlora space_deploy
cp space/app.py space/requirements.txt space/README.md space_deploy/
cd space_deploy
git add app.py requirements.txt README.md
git commit -m "Deploy biomedical MCQ QLoRA demo"
git push
```

The Space builds from `requirements.txt` and launches `app.py`. The first query
downloads + loads the model (cold start); later queries are fast.

## Cold-start tradeoff

Lazy loading (current default) keeps boot cheap but makes the first request slow.
To prefer warm latency over boot cost, move the `get_model()` call to import time
in `app.py` so the model loads during build/startup instead of on first request.
