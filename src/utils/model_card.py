"""Generate a Hugging Face model card for the trained LoRA adapter.

Pulls real numbers from run_metadata.json and the eval/cost tables when present;
anything not yet measured is rendered as PENDING RUN (never fabricated).
"""
from __future__ import annotations

import json
from pathlib import Path


def build_model_card(adapter_dir: str | Path, base_model: str,
                     headline_table: str, cost_table: str,
                     error_analysis_md: str, hub_repo_id: str) -> str:
    adapter_dir = Path(adapter_dir)
    meta_path = adapter_dir / "run_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    n = meta.get("train_size_actual", "PENDING RUN")
    pct = meta.get("pct_trainable", "PENDING RUN")
    seed = meta.get("seed", 42)
    hp = meta.get("hyperparameters", {})

    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - medical
  - mcqa
  - qlora
  - peft
  - lora
  - medmcqa
datasets:
  - openlifescienceai/medmcqa
library_name: peft
---

# {hub_repo_id}

A 4-bit QLoRA adapter for **{base_model}**, fine-tuned on **MedMCQA** as part of a
systematic study of *when* parameter-efficient domain fine-tuning helps an
already-strong instruction-tuned LLM. This is one point in a data-scaling sweep
(N ∈ {{5K, 20K, 50K}}), not a single-number accuracy claim.

## ⚠️ Intended use & limitation

Research and education only. **This adapter must NOT be used for real clinical
decisions, diagnosis, or treatment.** Outputs (including option probabilities)
are uncalibrated and may be confidently wrong.

## Training

- **Base model:** {base_model}
- **Method:** 4-bit QLoRA (nf4, double quant), LoRA r={meta.get('lora', {}).get('r', 'PENDING RUN')}
- **Training examples (N):** {n}
- **Trainable parameters:** {pct}% of total
- **Seed:** {seed}
- **Key hyperparameters:** lr={hp.get('learning_rate', 'PENDING RUN')},
  epochs={hp.get('num_train_epochs', 'PENDING RUN')},
  effective batch size = {hp.get('per_device_train_batch_size', '?')} ×
  {hp.get('gradient_accumulation_steps', '?')}

## Evaluation

All scoring via EleutherAI lm-evaluation-harness. MedMCQA = in-domain
(official validation split); PubMedQA = out-of-domain (eval only, never trained).

### Data-scaling results

{headline_table}

### Per-subject error analysis

{error_analysis_md}

### Inference cost

{cost_table}

## Reproduction

See the project repository (`scripts/train.py`, `scripts/run_eval.py`,
`reproduce.sh`). Origin note: this project is the modernized successor to a
DeBERTa aphasia-classification study limited by a tiny (n=95) dataset.
"""


def write_model_card(adapter_dir: str | Path, **kwargs) -> Path:
    card = build_model_card(adapter_dir=adapter_dir, **kwargs)
    path = Path(adapter_dir) / "README.md"
    path.write_text(card)
    return path
