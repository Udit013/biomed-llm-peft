# Biomedical LLM Adaptation: When Is PEFT Worth It?

A reproducible engineering pipeline for studying **when** 4-bit QLoRA fine-tuning
actually helps an already-strong instruction-tuned LLM on a biomedical benchmark,
**how much data** is needed before it stops paying off, and **at what inference
cost**.

> **This is a systematic engineering investigation, not an accuracy chase.** The
> success criterion is *"did we learn when PEFT is worth it?"* — **not** *"did we
> beat the benchmark?"*. A thin or even negative gap is a legitimate finding:
> strong instruction tuning may already saturate the benchmark. The study is
> designed so any outcome — large gain, marginal gain, or none — is a real result.

> ⚠️ **Not for clinical use.** Research and education only. Nothing here, including
> any model or "option probability", may inform real medical decisions.

---

## 1. Engineering & methodology (the point of this project)

| Concern | How it's handled |
|---|---|
| **Reproducibility** | Fixed seed, every hyperparameter + `% trainable params` logged to `run_metadata.json`; pinned `requirements.txt`; one-command `reproduce.sh`. |
| **Honest evaluation** | All scoring via **EleutherAI lm-evaluation-harness** (`lm_eval_tasks/`), never a hand-rolled loop. Base and fine-tuned models scored under identical 4-bit quantization and identical chat-templated prompts. |
| **Leakage safety** | Train on a seeded subsample of MedMCQA's official `train`; score on the official `validation` split; held-out val tail used for eval-loss only. Provenance stated in [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md). |
| **No fabricated numbers** | Any unmeasured metric is an explicit **`PENDING RUN`** marker. Tables fill only after real Colab T4 runs. |
| **Graceful degradation** | W&B logging is optional — no key ⇒ local JSONL logging, training never blocks. Hub push skips cleanly with no token. vLLM auto-skips on unsupported GPUs. |
| **T4-aware** | Defaults sized for a free Colab T4 (fp16, 4-bit, grad checkpointing, grad accumulation, capped seq length); checkpointed + resumable across disconnects. |

The single source of truth for the MCQ prompt format is
[`src/data/format.py`](src/data/format.py); training and the lm-eval tasks use it
identically so the base-vs-fine-tuned comparison is apples-to-apples.

### Quickstart

```bash
pip install -r requirements.txt
python scripts/smoke_test.py                     # structural check, no GPU, no downloads
bash reproduce.sh configs/qlora_5k.yaml          # full 5K slice on a GPU (Colab T4)
```

On Colab, open [`notebooks/run_colab.ipynb`](notebooks/run_colab.ipynb): it clones
the repo, installs pinned deps, mounts Drive for checkpoint/output persistence, and
runs train → eval → save, resumable across disconnects.

---

## 2. The headline result: data-scaling sweep

The deliverable is **how performance scales with N**, not a single number. Each
QLoRA run is one point in the sweep (`N ∈ {5K, 20K, 50K}`); 50K/full require Colab
Pro or a longer/cloud GPU.

All cells are `PENDING RUN` until the real T4 runs complete (`scripts/results_table.py`
regenerates this from `results/`).

| Model | MedMCQA (in-domain) | PubMedQA (out-of-domain) |
|---|---|---|
| Base 0-shot | PENDING RUN | PENDING RUN |
| Base 5-shot | PENDING RUN | PENDING RUN |
| QLoRA 5K | PENDING RUN | PENDING RUN |
| QLoRA 20K | PENDING RUN | PENDING RUN |
| QLoRA 50K | PENDING RUN | PENDING RUN |

**Reading guide (fill after runs):** Does accuracy rise then plateau with N (PEFT
helps but saturates)? Stay flat (instruction tuning already saturates MedMCQA)?
Does PubMedQA move with MedMCQA training (generalization) or not (in-domain
overfit)? The answer to the central question goes here, grounded in these numbers.

---

## 3. Error analysis & interpretation

MedMCQA accuracy is broken down **by subject** for base vs fine-tuned
(`scripts/error_analysis.py`, from lm-eval `--log_samples`), then each subject is
classified **improved / neutral / worsened**.

> **PENDING RUN.** After the runs, this section states explicitly which subjects
> PEFT improved, left neutral, or worsened, with a one-line hypothesis — e.g.
> whether PEFT mainly strengthens *fact-recall* domains (Pharmacology, Anatomy)
> versus *reasoning-heavy* ones (Medicine, Surgery). The interpretation, not just
> the table, lives here.

---

## 4. Inference cost

Real measured throughput / latency / VRAM across serving configs
(`scripts/inference_cost.py`). The quantized-vs-fp16 comparison is the must-have on
T4; **vLLM is conditional** — on a T4 (compute 7.5) it is recorded as *"not
feasible on T4 — PENDING on higher-end GPU"*, never fabricated.

| Config | Accuracy | Tokens/sec | p50 latency (s) | p95 latency (s) | VRAM (GB) |
|---|---|---|---|---|---|
| Base (fp16) | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN |
| LoRA fp16 | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN |
| Quantized (bnb 4-bit) | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN | PENDING RUN |
| vLLM | not feasible on T4 — PENDING on higher-end GPU | — | — | — | — |

---

## 5. Data

- **MedMCQA** (`openlifescienceai/medmcqa`, ~194K 4-option medical MCQs) — primary,
  train + in-domain eval. Official `validation` split (4183 labeled) for scoring;
  the official `test` split has hidden labels and is not used.
- **PubMedQA** (`qiaojin/PubMedQA`, `pqa_labeled`, 1000) — secondary, **out-of-domain
  eval only, never trained on.**

See [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) for full split provenance.

---

## 6. Repository layout

```
configs/          base + per-N sweep configs (inherit from base.yaml)
src/
  data/           MedMCQA loading/splitting + canonical MCQ prompt format
  train/          4-bit QLoRA SFT (TRL), checkpointed + resumable
  eval/           lm-eval harness wrapper, results table, per-subject error analysis
  serve/          model loader, option-probability scoring, FastAPI, cost benchmark
  utils/          config, seeding/env, graceful W&B, model card
lm_eval_tasks/    EleutherAI lm-eval task YAMLs (medmcqa_val, pubmedqa_ood)
scripts/          CLI entrypoints (train, run_eval, error_analysis, cost, hub, smoke)
notebooks/        run_colab.ipynb (T4, Drive-persisted, resumable)
space/            Gradio app for HF Spaces + DEPLOY.md
docs/             experiment plan
legacy/           archived DeBERTa aphasia-classification project (see below)
reproduce.sh      one-command 5K reproduction
Dockerfile        CUDA image for GPU train/eval/serve
```

---

## 7. Serving & artifacts

- **FastAPI** (`src/serve/api.py`): loads base + adapter, returns the chosen letter
  plus **normalized, uncalibrated option probabilities** derived from generation
  logits — documented as *option probabilities*, **not** "confidence".
- **Gradio Space** (`space/`): enter a medical MCQ, get answer + reasoning; loads on
  demand (cold-start tradeoff documented). See [`space/DEPLOY.md`](space/DEPLOY.md).
- **Hugging Face Hub** (`scripts/push_to_hub.py`): pushes the LoRA adapter with a
  complete model card (task, data, base model, training config, eval + cost tables,
  intended use, and the explicit no-clinical-use limitation).
- **Docker** + pinned deps + `reproduce.sh` for one-command reproduction.

---

## 8. Origin note — from a tiny aphasia dataset to properly-powered open data

This project is the modernized successor to an earlier **DeBERTa-v3
aphasia-classification** study (aphasia vs. control speech transcripts). That work
was fundamentally limited by a tiny dataset — **n = 95** transcripts — which caps
what any conclusion about model adaptation can claim. Rather than chase a number on
95 examples, this successor moves to **properly-powered open biomedical data**
(MedMCQA, ~194K) and reframes the question from *"can we classify?"* to the more
honest engineering question: ***when does domain fine-tuning actually pay off, and
at what cost?*** The original implementation is preserved unchanged under
[`legacy/aphasia-classification/`](legacy/aphasia-classification/) for reference.
