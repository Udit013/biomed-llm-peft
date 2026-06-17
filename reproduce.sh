#!/usr/bin/env bash
# One-command reproduction of the 5K slice: train -> eval (base + finetuned) ->
# error analysis -> headline table. Sized for a single free Colab T4 session.
#
# Scale up by swapping the config (configs/qlora_20k.yaml, etc.). 50K/full need
# Colab Pro or a longer/cloud GPU (see docs/EXPERIMENT_PLAN.md).
#
# Env: optional WANDB_API_KEY (graceful if unset), HF_TOKEN (for hub push).
set -euo pipefail

CONFIG="${1:-configs/qlora_5k.yaml}"
ADAPTER_DIR="$(python -c "import yaml,sys; c=yaml.safe_load(open('$CONFIG')); print(c.get('output',{}).get('output_dir','outputs/qlora_5k'))")"
RUN_TAG="$(basename "$ADAPTER_DIR")"

echo "==> Environment"
python -m src.utils.env

echo "==> 1/5 Train QLoRA ($CONFIG)"
python scripts/train.py --config "$CONFIG"

echo "==> 2/5 Eval BASE (0-shot)"
python scripts/run_eval.py --mode base0 --output results/base_0shot

echo "==> 3/5 Eval BASE (5-shot)"
python scripts/run_eval.py --mode base5 --output results/base_5shot

echo "==> 4/5 Eval FINE-TUNED ($RUN_TAG)"
python scripts/run_eval.py --adapter "$ADAPTER_DIR" --output "results/$RUN_TAG"

echo "==> 5/5 Error analysis + headline table"
python scripts/error_analysis.py --base results/base_0shot \
    --finetuned "results/$RUN_TAG" --out "results/error_analysis_$RUN_TAG"
python scripts/results_table.py --out results/headline_table.md

echo "==> Done. See results/ for measured numbers."
