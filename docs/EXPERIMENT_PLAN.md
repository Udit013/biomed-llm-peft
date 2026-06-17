# Experiment plan

## Central question

**When** does parameter-efficient domain fine-tuning (4-bit QLoRA) actually help
an already-strong instruction-tuned LLM (Qwen2.5-7B-Instruct), **how much data**
is needed before it stops paying off, and **at what inference cost**?

This is an engineering investigation, not an accuracy chase. The success
criterion is *"did we learn when PEFT is worth it?"* — **not** *"did we beat the
benchmark?"*. A thin or even negative gap is a legitimate, publishable finding:
strong instruction tuning may already saturate MedMCQA. The writeup is designed
so any outcome (large gain, marginal gain, or no gain) is a real result.

## Variables

- **Independent:** training-set size `N ∈ {5K, 20K, 50K, (optional) full}`.
- **Controlled:** base model, seed (42), LoRA config, quantization, prompt format,
  eval protocol. All logged per run in `run_metadata.json`.
- **Dependent:** in-domain accuracy (MedMCQA val), out-of-domain accuracy
  (PubMedQA), per-subject accuracy deltas, and inference cost.

## Data & splits (leakage-safe)

- **MedMCQA** (`openlifescienceai/medmcqa`, ~194K). Train on a seeded subsample of
  the official `train` split; a disjoint `val_size` tail is held out for eval
  loss only. **Scoring uses the official `validation` split (4183 labeled).** The
  official `test` split has hidden labels (cop=-1) and is not used.
- **PubMedQA** (`qiaojin/PubMedQA`, `pqa_labeled`, 1000). **Out-of-domain eval
  only — never trained on.** Tests whether MedMCQA PEFT generalizes or overfits.

## Procedure (build incrementally — verify each layer)

1. **Structural validation (local, Mac/CPU).** Import all modules, build configs,
   dry-run the eval command. No GPU, all metrics PENDING RUN. (`scripts/smoke_test.py`)
2. **5K slice end-to-end (Colab T4).** `bash reproduce.sh configs/qlora_5k.yaml`
   must produce real numbers: train → base 0/5-shot eval → finetuned eval →
   per-subject error analysis → headline table. **Do not scale up until this
   yields real numbers.**
3. **Scale the sweep.** Repeat for 20K (fits a single free T4 session). 50K and
   full require Colab Pro / a longer or cloud GPU, OR resuming across multiple
   checkpointed T4 sessions (training resumes from `outputs/<run>/checkpoint-*`).
4. **Error analysis + interpretation.** Break MedMCQA accuracy down by subject;
   state which subjects PEFT improved / left neutral / worsened, with a one-line
   hypothesis (fact-recall vs reasoning-heavy). Interpretation lives in the README.
5. **Inference cost.** Benchmark base vs LoRA-fp16 vs quantized (and vLLM if the
   GPU supports it). T4 records vLLM as "not feasible — PENDING on higher-end GPU".
6. **Artifacts.** Push adapter + model card to the Hub; FastAPI + Gradio demo.

## Hardware reality (free Colab T4)

- 16 GB, Turing (compute 7.5): **no native bf16** → fp16 + float16 compute dtype.
- ~12h session cap, frequent disconnects → checkpoint often, resume-on-rerun.
- vLLM build commonly unsupported on 7.5 → detect at runtime, skip gracefully.
- 5K and 20K fit a single free session; 50K/full do not (documented, not assumed).

## What "done" looks like

A filled headline table (5 rows × 2 columns), a per-subject interpretation
paragraph, and a measured cost table — each cell either a real number or an
explicit PENDING RUN / not-feasible marker. Plus a one-paragraph answer to the
central question grounded in those numbers.
