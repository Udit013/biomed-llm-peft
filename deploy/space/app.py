"""Gradio Space — Biomedical AI Research Assistant (research-grade UI).

Self-contained (no repo imports) so it deploys to a Space on its own. Talks to the
FastAPI backend (BACKEND_URL) for live Base+RAG answers, and reads a bundled
benchmark_explorer.json for the precomputed 4-way comparison. Theme-aware (uses
Gradio CSS variables), with first-class loading / empty / error states.

NOT FOR CLINICAL USE — research/education only.
"""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

import gradio as gr
import httpx

BACKEND_URL = os.environ.get("BACKEND_URL", "").rstrip("/")
EXPLORER = Path(__file__).parent / "benchmark_explorer.json"
REPO = "https://github.com/Udit013/biomed-llm-peft"
MODEL = "https://huggingface.co/Udit013/qwen2.5-7b-medmcqa-qlora-5k"

CONFIG_LABEL = {"base": "Base", "ft": "Fine-tuned", "base_rag": "Base + RAG",
                "ft_rag": "Fine-tuned + RAG"}
SOURCE_META = {"pubmed": ("PubMed", "#2563eb"), "pmc": ("PMC", "#2563eb"),
               "who": ("WHO", "#0d9488"), "cdc": ("CDC", "#16a34a"),
               "nih": ("NIH", "#7c3aed")}
EXAMPLES = [
    "What is dulaglutide and how is it used in type 2 diabetes?",
    "How should sepsis be managed early?",
    "What anticoagulation is recommended for atrial fibrillation?",
    "How is an acute ischemic stroke treated with thrombolysis?",
]

CSS = """
:root { --biomed-radius: 12px; }
.biomed-header h1 { margin: 0 0 4px; font-size: 1.55rem; letter-spacing: -0.02em; }
.biomed-sub { color: var(--body-text-color-subdued); font-size: 0.95rem; margin-bottom: 8px; }
.biomed-disclaimer { font-size: 0.8rem; color: var(--body-text-color-subdued);
  border-left: 3px solid #f59e0b; padding: 2px 10px; margin: 6px 0 2px; }
.card { border: 1px solid var(--border-color-primary); border-radius: var(--biomed-radius);
  background: var(--background-fill-secondary); padding: 16px 18px; margin: 8px 0; }
.metastrip { display:flex; flex-wrap:wrap; gap:8px; font-size:0.78rem;
  color: var(--body-text-color-subdued); margin-bottom:12px; align-items:center; }
.pill { border:1px solid var(--border-color-primary); border-radius:999px;
  padding:2px 10px; white-space:nowrap; }
.pill.mono { font-variant-numeric: tabular-nums; font-family: var(--font-mono); }
.answer-body { font-size:1.0rem; line-height:1.6; white-space:pre-wrap; }
.cite { font-weight:600; font-size:0.72em; vertical-align:super;
  color: var(--link-text-color); }
.verify { display:inline-flex; align-items:center; gap:6px; font-size:0.85rem;
  font-weight:600; border-radius:8px; padding:6px 12px; margin-top:12px; }
.verify.ok   { background: rgba(22,163,74,0.12); color:#16a34a; }
.verify.warn { background: rgba(245,158,11,0.14); color:#d97706; }
.verify.none { background: var(--background-fill-primary); color: var(--body-text-color-subdued); }
.claims { margin-top:10px; font-size:0.85rem; }
.claims summary { cursor:pointer; color: var(--body-text-color-subdued); }
.claim { padding:6px 0; border-top:1px dashed var(--border-color-primary); }
.claim .dot { font-weight:700; }
.ev-card { border:1px solid var(--border-color-primary); border-radius:var(--biomed-radius);
  padding:12px 14px; margin:8px 0; background: var(--background-fill-primary); }
.ev-head { display:flex; align-items:center; gap:8px; margin-bottom:6px; flex-wrap:wrap; }
.badge { color:#fff; border-radius:6px; padding:1px 8px; font-size:0.72rem; font-weight:600; }
.ev-num { font-family:var(--font-mono); font-weight:700; color:var(--body-text-color-subdued); }
.ev-title { font-weight:600; font-size:0.92rem; }
.ev-title a { text-decoration:none; }
.ev-quote { font-size:0.86rem; color:var(--body-text-color-subdued); line-height:1.5; }
.grounded { font-size:0.72rem; }
.empty { text-align:center; color:var(--body-text-color-subdued); padding:36px 12px; }
.empty .big { font-size:2rem; margin-bottom:8px; }
.spinner { display:inline-block; width:16px; height:16px; border:2px solid var(--border-color-primary);
  border-top-color: var(--link-text-color); border-radius:50%; animation: spin 0.8s linear infinite;
  vertical-align:middle; margin-right:8px; }
@keyframes spin { to { transform: rotate(360deg); } }
"""

_CITE_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")

EMPTY_ANSWER = """<div class="empty"><div class="big">🔬</div>
<div><b>Ask a biomedical question</b></div>
<div style="margin-top:6px;font-size:0.9rem">Answers are grounded in retrieved PubMed
+ NIH/WHO/CDC evidence, with inline citations and per-claim verification. Try an
example below.</div></div>"""
EMPTY_EVIDENCE = '<div class="empty" style="padding:20px"><div>Retrieved sources will appear here.</div></div>'
LOADING = """<div class="card"><span class="spinner"></span><b>Analyzing the literature…</b>
<div class="biomed-sub" style="margin-top:8px">Planning → retrieving from pgvector →
generating a grounded answer → verifying each claim. First request after idle can
take ~30–60s (free-tier cold start).</div></div>"""


# ------------------------------- rendering -------------------------------
def _cite_spans(text: str) -> str:
    esc = html.escape(text)
    return _CITE_RE.sub(lambda m: f'<span class="cite">[{m.group(1)}]</span>', esc)


def _render_answer(d: dict) -> str:
    served = CONFIG_LABEL.get(d.get("config"), d.get("config", "?"))
    lat = d.get("latency_ms", {})
    total = round(sum(lat.values()))
    tokens = d.get("token_usage", {}).get("total_tokens", "?")
    claims = d.get("claims", [])
    n, ok = len(claims), sum(1 for c in claims if c.get("supported"))
    supported = d.get("all_claims_supported")

    meta = (f'<span class="pill">🧠 {served}</span>'
            f'<span class="pill mono">⏱ {total} ms</span>'
            f'<span class="pill mono">retrieve {lat.get("retrieve_ms","–")} ms</span>'
            f'<span class="pill mono">generate {lat.get("generate_ms","–")} ms</span>'
            f'<span class="pill mono">{tokens} tokens</span>')

    if supported is True:
        verify = f'<div class="verify ok">✓ All {n} claims grounded in retrieved evidence</div>'
    elif supported is False:
        verify = f'<div class="verify warn">⚠ {ok}/{n} claims grounded — treat the rest with caution</div>'
    else:
        verify = '<div class="verify none">No retrieval — answer is ungrounded</div>'

    claim_rows = ""
    for c in claims:
        dot = ('<span class="dot" style="color:#16a34a">✓</span>' if c.get("supported")
               else '<span class="dot" style="color:#d97706">⚠</span>')
        claim_rows += (f'<div class="claim">{dot} {html.escape(c["claim"])} '
                       f'<span class="ev-num">({c.get("support_score","?")})</span></div>')
    claims_block = (f'<details class="claims"><summary>Per-claim verification '
                    f'({ok}/{n} grounded)</summary>{claim_rows}</details>') if claims else ""

    return (f'<div class="card"><div class="metastrip">{meta}</div>'
            f'<div class="answer-body">{_cite_spans(d.get("answer",""))}</div>'
            f'{verify}{claims_block}</div>')


def _render_evidence(passages: list, citations: list) -> str:
    if not passages:
        return '<div class="empty" style="padding:20px"><div>No sources were retrieved for this query.</div></div>'
    cite_by_doc = {c["doc_id"]: c for c in citations}
    cards = []
    for i, p in enumerate(passages, 1):
        ch = p["chunk"]
        name, color = SOURCE_META.get(ch["source"], (ch["source"].upper(), "#64748b"))
        cit = cite_by_doc.get(ch["doc_id"], {})
        grounded = cit.get("supported")
        gdot = ('<span class="grounded" style="color:#16a34a">● grounded</span>' if grounded
                else '<span class="grounded" style="color:var(--body-text-color-subdued)">○</span>')
        url = ch.get("url") or "#"
        title = html.escape(ch.get("title", "(untitled)"))
        cards.append(
            f'<div class="ev-card"><div class="ev-head">'
            f'<span class="ev-num">[{i}]</span>'
            f'<span class="badge" style="background:{color}">{name}</span>'
            f'<span class="ev-title"><a href="{html.escape(url)}" target="_blank" rel="noopener">{title}</a></span>'
            f'{gdot}</div>'
            f'<div class="ev-quote">{html.escape(ch.get("text","")[:340])}…</div></div>')
    return "".join(cards)


def ask(question: str):
    if not question or not question.strip():
        return EMPTY_ANSWER, EMPTY_EVIDENCE
    if not BACKEND_URL:
        return ('<div class="card">⚠️ <b>No backend configured.</b> Set the '
                '<code>BACKEND_URL</code> Space variable to the API URL.</div>', "")
    try:
        r = httpx.post(f"{BACKEND_URL}/query", json={"question": question.strip()}, timeout=120)
        r.raise_for_status()
        d = r.json()
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        return (f'<div class="card verify warn" style="display:block">⚠ Backend error '
                f'({e.response.status_code}). {html.escape(str(detail)[:200])}</div>', "")
    except Exception:
        return ('<div class="card"><b>⏳ The backend is waking up.</b><div class="biomed-sub" '
                'style="margin-top:6px">Free-tier instances sleep when idle. Please retry in '
                '~30 seconds.</div></div>', "")
    return _render_answer(d), _render_evidence(d.get("passages", []), d.get("citations", []))


# --------------------------- benchmark explorer ---------------------------
def _load_explorer() -> dict:
    return json.loads(EXPLORER.read_text()) if EXPLORER.exists() else {}


def explore(question_id: str):
    data = _load_explorer()
    if not data:
        return ('<div class="empty"><div class="big">📊</div><div><b>Benchmark data not bundled '
                'yet</b></div><div style="margin-top:6px;font-size:0.9rem">Run '
                '<code>scripts/rag_benchmark.py</code> to generate the precomputed 4-way '
                'comparison (Base / Fine-tuned / Base+RAG / Fine-tuned+RAG).</div></div>')
    rows = ['<table style="width:100%;border-collapse:collapse;font-size:0.88rem">',
            '<tr style="text-align:left;color:var(--body-text-color-subdued)">'
            '<th>Config</th><th>Answer</th><th>Grounded</th><th>Cites</th><th>Latency</th></tr>']
    for cfg in ("base", "ft", "base_rag", "ft_rag"):
        recs = {r["id"]: r for r in data.get("configs", {}).get(cfg, {}).get("records", [])}
        r = recs.get(question_id)
        if not r:
            continue
        g = r["generation"].get("groundedness", "–")
        rows.append(f'<tr style="border-top:1px solid var(--border-color-primary)">'
                    f'<td style="padding:8px 6px"><b>{CONFIG_LABEL[cfg]}</b></td>'
                    f'<td style="padding:8px 6px">{html.escape(r["answer"][:180])}…</td>'
                    f'<td style="padding:8px 6px">{g}</td><td style="padding:8px 6px">{len(r["citations"])}</td>'
                    f'<td style="padding:8px 6px">{r["total_latency_ms"]} ms</td></tr>')
    rows.append("</table>")
    return f'<div class="card">{"".join(rows)}</div>'


def _question_choices():
    data = _load_explorer()
    for cfg in ("ft_rag", "base_rag", "base", "ft"):
        recs = data.get("configs", {}).get(cfg, {}).get("records", [])
        if recs:
            return [(r["question"], r["id"]) for r in recs]
    return []


def backend_status() -> str:
    if not BACKEND_URL:
        return '<span class="pill">⚠ backend not configured</span>'
    try:
        h = httpx.get(f"{BACKEND_URL}/health", timeout=15).json()
        disp = h.get("served_config_display") or "unknown"
        return (f'<span class="pill">● serving <b>{disp}</b></span>'
                f'<span class="pill">vector: {h.get("vector_backend")}</span>')
    except Exception:
        return '<span class="pill">● backend asleep — first query will wake it (~30s)</span>'


theme = gr.themes.Soft(primary_hue="blue", secondary_hue="teal",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(title="Biomedical AI Research Assistant", theme=theme, css=CSS,
               analytics_enabled=False) as demo:
    gr.HTML(
        '<div class="biomed-header"><h1>🩺 Biomedical AI Research Assistant</h1>'
        '<div class="biomed-sub">Grounded, cited answers over PubMed + NIH/WHO/CDC '
        'guidelines — retrieval-augmented generation with a multi-agent workflow and '
        'per-claim verification.</div>'
        f'<div class="metastrip" style="margin-top:2px">'
        f'<span class="pill"><a href="{REPO}" target="_blank">Code ↗</a></span>'
        f'<span class="pill"><a href="{MODEL}" target="_blank">Model ↗</a></span></div>'
        '<div class="biomed-disclaimer">⚠ Research & education only — not medical advice.</div></div>')
    status = gr.HTML()   # live "serving Base + RAG" badge, populated on load

    with gr.Tab("Ask"):
        with gr.Row():
            q = gr.Textbox(placeholder="e.g. What anticoagulation is recommended for atrial fibrillation?",
                           label="Biomedical question", lines=2, scale=5, max_length=1000)
            btn = gr.Button("Ask", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=q, label="Example questions (well-supported by the corpus)")
        with gr.Row():
            with gr.Column(scale=3):
                answer_out = gr.HTML(EMPTY_ANSWER)
            with gr.Column(scale=2):
                gr.HTML('<div class="biomed-sub" style="margin:6px 0 0"><b>Retrieved evidence</b></div>')
                evidence_out = gr.HTML(EMPTY_EVIDENCE)

    with gr.Tab("Benchmark Explorer"):
        gr.HTML('<div class="biomed-sub">Precomputed <b>4-way</b> comparison — Base / '
                'Fine-tuned / Base+RAG / Fine-tuned+RAG — on curated questions. Runs with '
                'no GPU or backend.</div>')
        qid = gr.Dropdown(choices=_question_choices(), label="Curated evaluation question")
        table = gr.HTML()
        qid.change(explore, inputs=qid, outputs=table)

    with gr.Accordion("How it works · methodology · limitations", open=False):
        gr.Markdown(
            "**Pipeline.** A LangGraph workflow runs *Planner → Retrieval → Answer → "
            "Citation-Verification*. Queries are embedded and matched against a Neon "
            "Postgres + pgvector index of PubMed abstracts and NIH/WHO/CDC guidelines; "
            "the LLM answers **only** from retrieved passages and cites each claim `[n]`; "
            "a verification pass scores every claim against the evidence (semantic "
            "similarity) and flags unsupported ones.\n\n"
            "**Honest serving.** On free-tier infra the live model is **Base + RAG** "
            "(HF Inference can't serve a custom LoRA adapter). The API reports the exact "
            "config it served and this UI shows it; a GPU endpoint upgrades it to "
            "**Fine-tuned + RAG** with no UI change. The fine-tuned comparison lives in "
            "the Benchmark Explorer.\n\n"
            "**Limitations.** Small curated corpus; retrieval is bi-encoder only in the "
            "free deploy; answers can be incomplete or wrong. **Not medical advice.**\n\n"
            f"[Code]({REPO}) · [Model]({MODEL})")

    btn.click(lambda: (LOADING, EMPTY_EVIDENCE), outputs=[answer_out, evidence_out]) \
       .then(ask, inputs=q, outputs=[answer_out, evidence_out])
    q.submit(lambda: (LOADING, EMPTY_EVIDENCE), outputs=[answer_out, evidence_out]) \
       .then(ask, inputs=q, outputs=[answer_out, evidence_out])
    demo.load(backend_status, outputs=status)

if __name__ == "__main__":
    demo.launch()
