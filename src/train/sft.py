"""4-bit QLoRA supervised fine-tuning of Qwen2.5-7B-Instruct on MedMCQA.

Designed for a single free Colab T4 (16 GB). Heavy CUDA-only imports (bitsandbytes)
happen lazily inside `train` so this module imports cleanly on macOS/CPU for
structural validation. Training is checkpointed and resumable: re-running with
the same output_dir picks up the latest checkpoint, so a Colab disconnect costs
at most `save_steps` of progress.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..data.medmcqa import load_train_val
from ..utils.config import Config
from ..utils.env import env_fingerprint, get_gpu_info, set_seed
from ..utils.tracking import LocalLogger, init_wandb, report_to


def _latest_checkpoint(output_dir: Path) -> str | bool:
    """Return path to resume from, or True to let Trainer auto-detect, else False."""
    if not output_dir.exists():
        return False
    ckpts = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
        default=None,
    )
    return str(ckpts[-1]) if ckpts else False


def train(cfg: Config) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)
    output_dir = Path(cfg.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = LocalLogger(output_dir)

    gpu = get_gpu_info()
    fingerprint = env_fingerprint()
    logger.log({"event": "env", **fingerprint})
    init_wandb(cfg.run_name, {**cfg, "env": fingerprint})

    # ---- data ----
    train_ds, val_ds = load_train_val(
        train_size=cfg.data.train_size, val_size=cfg.data.val_size, seed=cfg.seed
    )
    print(f"[train] N_train={len(train_ds)}  N_val={len(val_ds)}")

    # ---- tokenizer ----
    tok = AutoTokenizer.from_pretrained(cfg.model.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- 4-bit quantized base model ----
    compute_dtype = getattr(torch, cfg.quant.bnb_4bit_compute_dtype)
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.quant.load_in_4bit,
        bnb_4bit_quant_type=cfg.quant.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.quant.bnb_4bit_use_double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
    )

    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=list(cfg.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable, total = _count_trainable(model)
    pct = 100 * trainable / total
    print(f"[train] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")

    # ---- training args ----
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=cfg.train.num_train_epochs,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=cfg.train.fp16,
        bf16=cfg.train.bf16,
        optim=cfg.train.optim,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="steps",
        eval_steps=cfg.train.save_steps,
        max_seq_length=cfg.model.max_seq_length,
        packing=False,
        report_to=report_to(),
        run_name=cfg.run_name,
        seed=cfg.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )

    resume = _latest_checkpoint(output_dir)
    if resume:
        print(f"[train] resuming from {resume}")
    train_result = trainer.train(resume_from_checkpoint=resume)

    # ---- save adapter + metadata ----
    trainer.save_model(str(output_dir))
    tok.save_pretrained(str(output_dir))

    metadata = {
        "run_name": cfg.run_name,
        "base_model": cfg.model.base_model,
        "seed": cfg.seed,
        "train_size_requested": cfg.data.train_size,
        "train_size_actual": len(train_ds),
        "val_size": len(val_ds),
        "trainable_params": trainable,
        "total_params": total,
        "pct_trainable": round(pct, 6),
        "hyperparameters": dict(cfg.train),
        "lora": dict(cfg.lora),
        "quant": dict(cfg.quant),
        "gpu": gpu.__dict__,
        "train_metrics": train_result.metrics,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.log({"event": "done", **metadata})
    print(f"[train] adapter + metadata saved to {output_dir}")
    return metadata


def _count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
