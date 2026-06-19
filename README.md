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

> **Status.** The pipeline is complete and validated end-to-end on a free Colab T4,
> with a real, analyzed **N = 5K** result (below). The multi-point scaling sweep
> (20K/50K) is **scoped as future work** — it is compute-bound, not blocked by the
> code: each larger run is a one-line config change but exceeds a free-tier GPU
> budget (see [§2](#2-the-headline-result-data-scaling-at-n5k)). The engineering and
> evaluation methodology — not the number of sweep points — is the deliverable.

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
the repo, installs pinned deps, runs train → eval → results on local disk (**no
Google Drive**), and persists outputs off-Colab (adapter → HF Hub, result tables →
your laptop).

---

## 2. The headline result: data-scaling at N=5K

The deliverable is **how performance scales with N**. We establish the first point
of that curve at **N = 5K** (the free-T4-feasible slice); the 20K/50K points are
[future work](#future-work-the-scaling-curve).

Measured on a free Colab T4, both models scored under identical 4-bit quantization
and chat-templated prompts. Numbers are a **200-item-per-task subsample** of the
official splits (lm-eval flags `--limit`), so each cell carries **±3.5 pp** standard
error; base and QLoRA were scored on the **same** items for an apples-to-apples
comparison. Full-split scoring (`EVAL_LIMIT=None`) is a longer/Pro-GPU run.

| Model | MedMCQA (in-domain) | PubMedQA (out-of-domain) |
|---|---|---|
| Base 0-shot (n=200) | 47.5 | 48.0 |
| **QLoRA 5K (n=200)** | **50.0** | **64.5** |
| QLoRA 20K | *future work (compute-bound)* | *future work* |
| QLoRA 50K | *future work (Pro/cloud GPU)* | *future work* |

*(Seed 42; LoRA r=16, α=32; 0.92% trainable params; 1 epoch; lr 2e-4. Base 5-shot
is supported but omitted here — it is slow on a T4 and secondary to the 0-shot
base-vs-PEFT contrast.)*

**What this says (at N=5K):**
- **In-domain (MedMCQA): essentially flat.** +2.5 pp is *within* the ±3.5 pp noise —
  5K QLoRA examples do **not** meaningfully improve an already strong
  instruction-tuned model on in-domain MCQs. This is the central finding at 5K, and
  exactly the kind of "thin gap" the study was designed to report honestly: strong
  instruction tuning already sits near the achievable accuracy here.
- **Out-of-domain (PubMedQA): a large jump (+16.5 pp).** This is well outside noise.
  The most likely mechanism is **not** new medical knowledge but **sharpened
  answer-selection behavior** — fine-tuning on MCQ format makes the model commit more
  decisively, which also helps PubMedQA's yes/no/maybe scoring. We flag this as a
  hypothesis to confirm at 20K, not a generalization claim.

**Takeaway:** at 5K, PEFT's value here is about *output behavior/format*, not
in-domain knowledge. Whether a real in-domain gain emerges with more data is the
open question the (compute-bound) 20K/50K points would answer.

---

## 3. Error analysis & interpretation

MedMCQA accuracy is broken down **by subject** for base vs fine-tuned
(`scripts/error_analysis.py`, from lm-eval `--log_samples`), then each subject is
classified **improved / neutral / worsened**. Full per-subject table:
[`results/error_analysis_qlora_5k.md`](results/error_analysis_qlora_5k.md).

**At N=5K (n=200 subsample), the breakdown is:**
- **Improved** (5): Dental, Medicine, Microbiology, Physiology, Surgery
- **Neutral** (10): Anaesthesia, Anatomy, ENT, Forensic Medicine, Ophthalmology,
  Pathology, Pediatrics, Pharmacology, Radiology, Social & Preventive Medicine
- **Worsened** (2): Biochemistry, Gynaecology & Obstetrics

**Interpretation — and an honest limit.** The dominant pattern is **neutral**,
consistent with the flat in-domain headline: at 5K, PEFT moves most subjects little.
Crucially, **the data does *not* support the tempting "PEFT strengthens fact-recall
domains more than reasoning-heavy ones" hypothesis** — reasoning-heavy Medicine and
Surgery *improved* while fact-recall Biochemistry *worsened*. We call that out rather
than fit a story to noise: with only 200 scored items, **per-subject counts are tiny
(1–57; several n≤6)**, so individual subject verdicts are within sampling noise and
should not be over-read. The trustworthy subject-level signal needs full-split scoring
and ideally the larger-N checkpoints — part of the [future work](#future-work-the-scaling-curve).
The methodological point stands: the harness produces a matched, leakage-safe,
per-subject comparison on demand; the 5K data simply isn't powered for subject-level
claims, and we say so.

---

## 4. Inference cost

The harness (`scripts/inference_cost.py`) measures real throughput / latency / VRAM
across serving configs. **These rows are deferred to the same compute-bound future
work** as the scaling curve — under the free-T4 budget we prioritized the train +
eval result, and two rows here are constrained by the hardware itself:

- **fp16 rows don't fit a T4.** A 7B in fp16 is ~15 GB and CPU-offloads on a 16 GB
  T4, so fp16 latency would measure offloading, not serving — it needs a ≥24 GB GPU.
- **vLLM is not feasible on a T4** (compute 7.5); the harness records it as
  *"not feasible on T4 — PENDING on higher-end GPU"* rather than fabricating a row.

| Config | Accuracy | Tokens/sec | p50 latency (s) | p95 latency (s) | VRAM (GB) |
|---|---|---|---|---|---|
| Base (fp16) | future work (needs ≥24 GB GPU) | — | — | — | — |
| LoRA fp16 | future work (needs ≥24 GB GPU) | — | — | — | — |
| Quantized (bnb 4-bit) | future work (T4-feasible) | — | — | — | — |
| vLLM | not feasible on T4 — needs higher-end GPU | — | — | — | — |

The quantized (4-bit) row *is* T4-feasible and is the natural first cost
measurement to add — `scripts/inference_cost.py --configs quantized` produces it.

---

## Future work: the scaling curve

This is a **deliberate scope boundary, not an unfinished sweep.** The code path for
every remaining item exists and is one config/flag away; what's missing is GPU budget
beyond a free Colab T4:

- **20K / 50K QLoRA points** — `scripts/train.py --config configs/qlora_20k.yaml`
  (etc.). 20K fits one long session; 50K needs Colab Pro / a cloud GPU. These turn the
  single 5K point into the actual scaling curve and would test whether the OOD jump is
  format-driven (likely stable) vs knowledge-driven (would grow with N).
- **Full-split scoring** — `EVAL_LIMIT=None` removes the ±3.5 pp subsample noise and
  powers the per-subject analysis for real subject-level claims.
- **Quantized inference-cost row** — T4-feasible; the fp16/vLLM rows need a larger GPU.

A one-off A100 rental (~$1–10) or Colab Pro completes all of the above without any
code changes.

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
notebooks/        run_colab.ipynb (T4, local-disk / Drive-free; Hub + laptop persistence)
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
