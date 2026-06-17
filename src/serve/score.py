"""Score the 4 options of a medical MCQ and return normalized probabilities.

These are UNCALIBRATED option probabilities derived from the model's generation
logits (length-normalized log-likelihood of each option, softmaxed over the 4).
They are NOT calibrated confidences and must not be read as such.
"""
from __future__ import annotations

from ..data.format import LETTERS, SYSTEM_PROMPT, render_question, render_target


def score_options(model, tok, question: str, options: list[str]) -> dict:
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_question(question, options)},
    ]
    prompt_ids = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    norm_logprobs: list[float] = []
    for i in range(len(options)):
        target = " " + render_target(i, options)
        target_ids = tok(target, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        full = torch.cat([prompt_ids, target_ids], dim=1)
        with torch.no_grad():
            logits = model(full).logits
        logp = F.log_softmax(logits, dim=-1)
        start = prompt_ids.shape[1]
        token_logprobs = [
            logp[0, pos - 1, full[0, pos]] for pos in range(start, full.shape[1])
        ]
        total = torch.stack(token_logprobs).sum()
        norm_logprobs.append((total / target_ids.shape[1]).item())

    probs = F.softmax(torch.tensor(norm_logprobs), dim=0).tolist()
    idx = max(range(len(options)), key=lambda i: probs[i])
    return {
        "answer": LETTERS[idx],
        "answer_text": options[idx],
        "option_probabilities": {LETTERS[i]: round(probs[i], 4) for i in range(len(options))},
        "note": "Uncalibrated option probabilities from generation logits; NOT a calibrated confidence.",
    }
