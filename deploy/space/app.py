"""Gradio Space: Biomedical AI Research Assistant.

Two tabs, matching the hybrid design:
  * Ask — arbitrary questions answered LIVE by the Fine-tuned + RAG pipeline via
    the FastAPI backend (BACKEND_URL). Shows answer, inline citations, retrieved
    evidence, per-claim verification, and latency.
  * Benchmark Explorer — precomputed 4-way comparison (Base / FT / Base+RAG /
    FT+RAG) loaded from bundled benchmark_explorer.json; no GPU/backend needed.

Self-contained (no repo imports) so this folder deploys to a Space on its own.
NOT FOR CLINICAL USE.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
EXPLORER = Path(__file__).parent / "benchmark_explorer.json"
CONFIG_LABEL = {"base": "Base", "ft": "Fine-tuned", "base_rag": "Base + RAG",
                "ft_rag": "Fine-tuned + RAG"}


# ------------------------------- Ask (live) -------------------------------
def ask(question: str):
    if not question.strip():
        return "Enter a question.", "", ""
    if not BACKEND_URL:
        return ("⚠️ No BACKEND_URL configured. Set it to your FastAPI (Render) URL to "
                "enable live answering."), "", ""
    try:
        r = httpx.post(f"{BACKEND_URL}/query", json={"question": question}, timeout=120)
        r.raise_for_status()
        d = r.json()
    except Exception as exc:  # noqa: BLE001
        return f"Backend error: {exc}", "", ""

    supported = d.get("all_claims_supported")
    badge = {True: "✅ all claims supported", False: "⚠️ some claims unsupported",
             None: "—"}[supported]
    lat = d.get("latency_ms", {})
    answer = (f"{d.get('answer','')}\n\n---\n**Verification:** {badge}  |  "
              f"**Latency:** { round(sum(lat.values()),1) } ms  |  "
              f"**Tokens:** {d.get('token_usage',{}).get('total_tokens','?')}")

    cites = "\n".join(
        f"{c['marker']} {'✅' if c.get('supported') else '⚠️'} "
        f"[{c['title']}]({c['url'] or '#'}) — {c['quote'][:160]}…"
        for c in d.get("citations", [])) or "_No citations._"
    evidence = "\n\n".join(
        f"**[{i}] {p['chunk']['title']}** ({p['chunk']['source']})\n{p['chunk']['text'][:300]}…"
        for i, p in enumerate(d.get("passages", []), 1)) or "_No passages retrieved._"
    return answer, cites, evidence


# --------------------------- Benchmark Explorer ---------------------------
def _load_explorer() -> dict:
    return json.loads(EXPLORER.read_text()) if EXPLORER.exists() else {}


def explore(question_id: str):
    data = _load_explorer()
    if not data:
        return "No benchmark data bundled. Run scripts/rag_benchmark.py and add its JSON."
    rows = ["| Config | Answer | Grounded | Citations | Latency (ms) |",
            "|---|---|---|---|---|"]
    for cfg in ("base", "ft", "base_rag", "ft_rag"):
        recs = {r["id"]: r for r in data.get("configs", {}).get(cfg, {}).get("records", [])}
        r = recs.get(question_id)
        if not r:
            continue
        g = r["generation"].get("groundedness", "—")
        rows.append(f"| {CONFIG_LABEL[cfg]} | {r['answer'][:160]}… | {g} | "
                    f"{len(r['citations'])} | {r['total_latency_ms']} |")
    return "\n".join(rows)


def _question_choices():
    data = _load_explorer()
    for cfg in ("ft_rag", "base_rag", "base", "ft"):
        recs = data.get("configs", {}).get(cfg, {}).get("records", [])
        if recs:
            return [(r["question"], r["id"]) for r in recs]
    return []


with gr.Blocks(title="Biomedical AI Research Assistant") as demo:
    gr.Markdown("# 🩺 Biomedical AI Research Assistant\n"
                "Grounded, cited answers over PubMed + NIH/WHO/CDC guidelines. "
                "**Research/education only — not medical advice.**")
    with gr.Tab("Ask (live Fine-tuned + RAG)"):
        q = gr.Textbox(label="Biomedical question", lines=2)
        btn = gr.Button("Ask", variant="primary")
        ans = gr.Markdown(label="Answer")
        with gr.Row():
            cites = gr.Markdown(label="Citations")
            evid = gr.Markdown(label="Retrieved evidence")
        btn.click(ask, inputs=q, outputs=[ans, cites, evid])
    with gr.Tab("Benchmark Explorer (precomputed 4-way)"):
        gr.Markdown("Compare Base / Fine-tuned / Base+RAG / Fine-tuned+RAG on curated "
                    "questions — precomputed metrics, evidence, and latency.")
        qid = gr.Dropdown(choices=_question_choices(), label="Curated question")
        table = gr.Markdown()
        qid.change(explore, inputs=qid, outputs=table)

if __name__ == "__main__":
    demo.launch()
