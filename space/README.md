---
title: Biomedical MCQ QLoRA
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.7.1
app_file: app.py
pinned: false
license: apache-2.0
---

# Biomedical MCQ — Qwen2.5-7B + MedMCQA QLoRA (demo)

Enter a 4-option medical multiple-choice question; the app returns the chosen
answer and a one-sentence rationale. The base model is `Qwen/Qwen2.5-7B-Instruct`
with a MedMCQA LoRA adapter loaded from the Hub (set via the `ADAPTER_REPO`
Space secret/variable).

**⚠️ Research/education only — NOT for clinical decisions.**

The model loads lazily on the first request (cold start). See `DEPLOY.md` in the
project repo for exact Space-creation and push steps.
