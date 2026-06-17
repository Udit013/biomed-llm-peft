"""Gradio app for Hugging Face Spaces: medical MCQ -> answer + reasoning.

Self-contained (does not import the main repo's `src/`) so this folder can be
pushed to a Space on its own. The model (base Qwen2.5-7B-Instruct + MedMCQA LoRA
adapter from the Hub) is loaded ON DEMAND on the first request.

COLD-START TRADEOFF: lazy loading keeps the Space cheap to boot, but the FIRST
query pays the full model-download + load cost (tens of seconds to minutes on
free CPU/limited hardware). Subsequent queries are fast. To trade cold start for
warm cost, move `get_model()` to import time.

NOT FOR CLINICAL USE — research/education demo only.
"""
from __future__ import annotations

import os

import gradio as gr

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "")  # HF Hub repo id of the adapter
LETTERS = ["A", "B", "C", "D"]
SYSTEM_PROMPT = (
    "You are a medical exam assistant. Choose the single best answer to the "
    "multiple-choice question and briefly explain your reasoning."
)

_MODEL = {}


def get_model():
    """Lazy-load model + tokenizer on first use (cold start happens here)."""
    if _MODEL:
        return _MODEL["model"], _MODEL["tok"]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    if ADAPTER_REPO:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, ADAPTER_REPO)
    model.eval()
    _MODEL.update(model=model, tok=tok)
    return model, tok


def answer(question, opt_a, opt_b, opt_c, opt_d):
    options = [opt_a, opt_b, opt_c, opt_d]
    if not question.strip() or not all(o.strip() for o in options):
        return "Please fill in the question and all four options."
    model, tok = get_model()
    stem = "Question: " + question.strip() + "\n" + "\n".join(
        f"{L}. {o.strip()}" for L, o in zip(LETTERS, options)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": stem + "\n\nGive the letter, then a one-sentence reason."},
    ]
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    prompt = prompt.to(next(model.parameters()).device)
    out = model.generate(prompt, max_new_tokens=160, do_sample=False)
    text = tok.decode(out[0, prompt.shape[1]:], skip_special_tokens=True).strip()
    return text + "\n\n— Research/education demo. NOT for clinical use."


demo = gr.Interface(
    fn=answer,
    inputs=[
        gr.Textbox(label="Question", lines=2),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
    ],
    outputs=gr.Textbox(label="Answer + reasoning", lines=6),
    title="Biomedical MCQ — Qwen2.5-7B + MedMCQA QLoRA",
    description="Enter a 4-option medical MCQ. NOT for clinical use. First query "
                "is slow (cold-start model load); later queries are fast.",
    examples=[[
        "Which vitamin deficiency causes scurvy?",
        "Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K",
    ]],
)

if __name__ == "__main__":
    demo.launch()
