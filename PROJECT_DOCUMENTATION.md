# Project Documentation — Biomedical AI Research Assistant

**The single source of truth for understanding this repository.** Written for four
audiences at once: you preparing for interviews, a recruiter skimming for signal, a
new developer onboarding, and a complete beginner who has never programmed.

> **A note on scope, read this first.** This repository is a **Python AI/backend
> system**, not a React/Next.js web app. It has no client-side JavaScript framework,
> no JWT/OAuth login, no Redux/Zustand, no SSR/hydration. Where a standard
> documentation template asks about those things, this document says so plainly and
> explains what plays the equivalent role instead (e.g. "state management" here means
> how a stateless Python API avoids reloading a 7B model on every request, not how a
> browser re-renders a UI tree). **Nothing below is invented — every claim is traced
> to an actual file in this repository.**

---

## Table of Contents

1. [The Story: What This Project Is](#1-the-story-what-this-project-is)
2. [Who This Is Built For](#2-who-this-is-built-for)
3. [What Happens When a User Opens the App](#3-what-happens-when-a-user-opens-the-app)
4. [High-Level Architecture](#4-high-level-architecture)
5. [The Two Halves of This Repository](#5-the-two-halves-of-this-repository)
6. [Folder Structure, Explained](#6-folder-structure-explained)
7. [Core Concepts, Beginner → Advanced](#7-core-concepts-beginner--advanced)
8. [Part A Deep Dive: The Production Assistant](#8-part-a-deep-dive-the-production-assistant)
   - [8.1 Configuration (`config.py`)](#81-configuration-configpy)
   - [8.2 Logging (`logging.py`)](#82-logging-loggingpy)
   - [8.3 Data Models (`schema.py`)](#83-data-models-schemapy)
   - [8.4 The RAG Pipeline](#84-the-rag-pipeline)
   - [8.5 The Agent Workflow (LangGraph)](#85-the-agent-workflow-langgraph)
   - [8.6 The Serving Layer](#86-the-serving-layer)
   - [8.7 The API (FastAPI)](#87-the-api-fastapi)
   - [8.8 The Evaluation Harness](#88-the-evaluation-harness)
   - [8.9 The Frontend (Gradio)](#89-the-frontend-gradio)
9. [Part B Deep Dive: The QLoRA Research Pipeline](#9-part-b-deep-dive-the-qlora-research-pipeline)
10. [The Database Layer](#10-the-database-layer)
11. [Authentication — What Exists and What Doesn't](#11-authentication--what-exists-and-what-doesnt)
12. [State Management — What Exists and What Doesn't](#12-state-management--what-exists-and-what-doesnt)
13. [Every Dependency, Explained](#13-every-dependency-explained)
14. [Every Configuration File, Explained](#14-every-configuration-file-explained)
15. [The API Reference](#15-the-api-reference)
16. [Execution Traces for Every Major Feature](#16-execution-traces-for-every-major-feature)
17. [Design Decisions and Trade-offs](#17-design-decisions-and-trade-offs)
18. [Testing Strategy](#18-testing-strategy)
19. [Glossary](#19-glossary)
20. [Complete End-to-End Walkthrough](#20-complete-end-to-end-walkthrough)
21. [Interview Question Bank](#21-interview-question-bank)

---

## 1. The Story: What This Project Is

Imagine you are a medical student, a nurse, or a curious person who just read a
headline about a new diabetes drug. You have a specific question — *"What is
dulaglutide and how is it used in type 2 diabetes?"* — and you want a real answer,
not a wall of jargon from a 40-page PDF, and not a made-up answer from a chatbot
that sounds confident but is wrong.

This project is a **biomedical question-answering assistant** that solves that
problem two ways at once:

1. It **never answers from thin air.** Every question is first used to *search* a
   library of real medical abstracts and health-guideline documents. The AI is
   then instructed to answer **only** using what it found, and to attach a
   citation like `[1]` to every sentence, the way a term paper cites its sources.
2. It **checks its own work.** After the AI writes an answer, a second step
   re-reads every sentence and asks: "is this actually supported by the sources I
   found?" If a sentence isn't backed up, it gets flagged, so you can see exactly
   which parts of the answer to trust.

On top of that, the project asks a genuinely research-flavored question: *"Does
teaching a general-purpose AI model extra medical facts (fine-tuning) actually make
it better at this, and how much does that cost to build and run?"* Answering that
honestly — including reporting a "no, not really" result where that's what the data
showed — is the whole point of the [Part B research pipeline](#9-part-b-deep-dive-the-qlora-research-pipeline).

### The one-sentence pitch

> *A biomedical research assistant that retrieves real medical literature, answers
> questions using only that evidence, cites every claim, and automatically checks
> whether each claim is actually supported — built on top of a from-scratch study
> of whether fine-tuning a large language model on medical data is worth the cost.*

### Analogy: a research assistant, not an oracle

Think of a human research librarian. If you ask them "what's the treatment for X,"
a good librarian doesn't answer from memory and hope they're right. They **go to
the shelf, pull three or four relevant papers, read the question again with those
papers open, write an answer that quotes the papers, and hand you both the answer
and the papers** so you can check their work yourself. This project builds a
software version of exactly that librarian, using PubMed (a medical literature
database), plus NIH/WHO/CDC health guidelines, as "the shelf."

---

## 2. Who This Is Built For

- **A person asking a medical research question** — the live demo (see
  [§4](#4-high-level-architecture)) — gets a cited, verified answer in seconds.
- **A researcher or engineer** studying whether parameter-efficient fine-tuning
  (a cheaper alternative to fully retraining an AI model) is worth it for a
  specific domain — the [Part B](#9-part-b-deep-dive-the-qlora-research-pipeline)
  pipeline is a reusable, honest experiment harness for exactly that question.
- **A hiring manager or interviewer** evaluating whether the author can design,
  build, evaluate, and *deploy* (not just prototype) a real retrieval-augmented,
  multi-agent AI system on a real budget (free-tier cloud infrastructure).
- **A new engineer** picking up the codebase — this document, plus the inline
  comments in every file, are written so you can trace any behavior back to the
  exact lines of code that produce it.

> ⚠️ **This is explicitly NOT a clinical tool.** Every layer of the system —
> the prompts, the model cards, the UI — states this. It is a research and
> education project. Nothing here should inform a real medical decision.

---

## 3. What Happens When a User Opens the App

There are actually **two different "apps"** in this repository, and understanding
which one you mean matters:

| | "The live demo" | "The research pipeline" |
|---|---|---|
| What it is | A web page (Gradio Space) anyone can visit | A set of command-line scripts a researcher runs |
| Who uses it | Anyone with a browser | You, running Python scripts on a GPU (e.g. Google Colab) |
| What happens | Ask a question → get a cited answer | Fine-tune a model, then measure if it got better |
| Covered in | [§8](#8-part-a-deep-dive-the-production-assistant) (Part A) | [§9](#9-part-b-deep-dive-the-qlora-research-pipeline) (Part B) |

This section walks through the **live demo**, since that's "the app" in the
everyday sense.

### Step by step, for someone who has never touched a browser dev tool

1. **You open a URL** (a Hugging Face "Space" — a free website that Hugging Face
   hosts for you). Your browser downloads a small Python-powered web page built
   with a library called **Gradio** (think of Gradio as a way to turn a Python
   function into a web form with almost no HTML/CSS/JavaScript writing).
2. **You see a text box** ("Biomedical question") and a button ("Ask"), plus a
   second tab ("Benchmark Explorer") and an expandable "How it works" section.
   This page is defined entirely in one file: `deploy/space/app.py`.
3. **You type a question and click Ask.** The browser sends your question, as
   plain JSON text, over the internet to a **separate server** — a FastAPI
   backend hosted on Render (`src/assistant/api/app.py`). The Gradio page itself
   does no "thinking"; it's just a messenger.
4. **The backend does the real work** (fully detailed in
   [§16.1](#161-live-query-a-user-asks-a-question)):
   it figures out a search strategy, searches a database of medical text for
   relevant passages, asks an AI language model to write an answer using *only*
   those passages, checks whether every sentence in that answer is actually
   backed up by the passages, and packages all of that (answer + sources +
   verification + timing) into one JSON response.
5. **The response travels back** to your browser. The Gradio page's Python code
   turns that JSON into readable HTML: a highlighted answer with `[1]`-style
   citation markers, a green "✓ all claims grounded" badge (or an amber
   warning), and a list of the actual source documents with links.
6. **You read the answer and can click through to PubMed** to read the real
   paper yourself — nothing is hidden.

If the backend has been idle for a while (free hosting tiers "sleep" servers to
save cost), step 3 takes 30–60 seconds the first time as the server "wakes up."
The UI explicitly tells you this is happening instead of leaving you staring at a
frozen screen — a first-class **loading state**, not a silent hang.

---

## 4. High-Level Architecture

```text
                         ┌─────────────────────────────────────────────┐
                         │        Hugging Face Space (Gradio)           │
                         │  "Ask" tab (live)   "Benchmark Explorer" tab │
                         │   deploy/space/app.py                        │
                         └───────────────┬───────────────┬──────────────┘
                                         │ HTTP POST      │ reads a bundled
                                         │ /query         │ JSON file (no network)
                                         ▼                ▼
                         ┌───────────────────────────────────────────────┐
                         │        FastAPI backend (Render, free tier)     │
                         │        src/assistant/api/app.py                │
                         │  GET /health   POST /query   GET /benchmark    │
                         └───────────────┬─────────────────────────────┘
                                         │  AssistantService.answer(question)
                                         ▼
                         ┌───────────────────────────────────────────────┐
                         │     LangGraph agent workflow (4 agents)        │
                         │     src/assistant/agents/                      │
                         │                                                 │
                         │   Planner ─► Retrieval ─► Answer ─► Verify      │
                         └──────┬───────────┬────────────┬────────┬───────┘
                                │           │            │        │
                                │  metadata │  RAG        │ prompt  │ semantic
                                │  filter   │  pipeline    │ + LLM   │ grounding
                                ▼           ▼            ▼        ▼
                    ┌─────────────────┐ ┌───────────────┐ ┌─────────────┐
                    │  (no I/O; pure  │ │ Vector search  │ │ LLM provider │
                    │   heuristic)    │ │ (RAGPipeline)  │ │ (HF Inference│
                    └─────────────────┘ └───────┬────────┘ │  or local)   │
                                                 │          └─────────────┘
                                                 ▼
                                    ┌────────────────────────┐
                                    │  Neon Postgres +        │
                                    │  pgvector (production)  │
                                    │  — or local NumPy store │
                                    │    (dev / CI / tests)   │
                                    └────────────────────────┘
```

**Reading the diagram:** a browser talks only to Gradio; Gradio talks only to the
FastAPI backend over plain HTTP; the backend runs a 4-step agent pipeline; the
pipeline's retrieval step talks to a vector database that stores biomedical text
as searchable numbers ("embeddings" — explained in [§7](#7-core-concepts-beginner--advanced));
the pipeline's answer step talks to a large language model (LLM) hosted either by
Hugging Face (free tier, base model only) or locally on a GPU (with the
fine-tuned adapter).

### Deployment architecture (where each piece actually runs)

```text
┌──────────────────┐      ┌──────────────────────┐      ┌───────────────────┐
│  Hugging Face      │      │  Render.com            │      │  Neon.tech          │
│  Spaces (frontend)  │◄────►│  (backend, Docker)     │◄────►│  (Postgres+pgvector)│
│  $0 / free tier     │HTTP  │  $0 / free tier, 512MB │ SQL  │  $0 / free tier     │
└──────────────────┘      └───────────┬──────────────┘      └───────────────────┘
                                       │ HTTPS
                                       ▼
                          ┌──────────────────────────┐
                          │  Hugging Face Inference     │
                          │  (hosted LLM: Qwen2.5-7B)    │
                          └──────────────────────────┘
```

Every box above is a **free tier** — this system was deliberately engineered to
run for **$0/month**. That constraint drives several real code decisions covered
in [§17](#17-design-decisions-and-trade-offs), most importantly: the backend
container has **no PyTorch installed at all** (`requirements-api.txt`), because a
7-billion-parameter model can't run on a free 512MB server; the LLM instead runs
on Hugging Face's hosted infrastructure, and query embeddings run through a tiny
CPU-only ONNX model (`fastembed`) instead of the heavier `sentence-transformers`
library used for building the search index offline.

---

## 5. The Two Halves of This Repository

This repository literally contains two projects, layered on top of each other,
and keeping them mentally separate is the single most important thing to
understand before reading any code.

### Part A — the production assistant (`src/assistant/`)

Everything described in [§4](#4-high-level-architecture) above: the RAG pipeline,
the agents, the API, the Gradio frontend, the evaluation harness, the deployment
configs. **This is new code, entirely self-contained** — nothing in Part B was
modified to build it.

### Part B — the QLoRA research pipeline (`src/data/`, `src/train/`, `src/eval/`, `src/serve/`)

The original research project this repo grew out of: fine-tune
`Qwen2.5-7B-Instruct` on a medical multiple-choice-question dataset (MedMCQA)
using a cheap technique called **QLoRA**, then honestly measure whether the
fine-tuned model is actually better than the un-fine-tuned ("base") model, using
a real, standardized evaluation tool (EleutherAI's `lm-evaluation-harness`)
instead of hand-rolled scoring.

**Why keep both?** Part A's live demo needs *some* language model to answer
questions with — that's Part B's fine-tuned adapter (or, honestly, the plain base
model on the free deployment; see [§17](#17-design-decisions-and-trade-offs)).
Part B's research question ("did fine-tuning help?") only matters *because* the
model gets used for something real, which is Part A. They are two halves of one
argument: *build the model carefully and evaluate it honestly (Part B), then
build a real system around it and evaluate that honestly too (Part A).*

There is also a third, older, **archived** project: `legacy/aphasia-classification/`
— an earlier, unrelated deep-learning project (classifying aphasia from
speech transcripts) that predates this repository's current focus and is kept
only for historical reference. It is not part of the running system.

---

## 6. Folder Structure, Explained

```text
biomed-llm-peft/
│
├── src/
│   ├── assistant/            ← PART A: the production RAG + agent system
│   │   ├── config.py           Central settings (env-var driven)
│   │   ├── logging.py          Structured JSON logging + timing helper
│   │   ├── schema.py           Typed data models (Pydantic) shared everywhere
│   │   ├── rag/                 Retrieval-Augmented Generation pipeline
│   │   │   ├── ingest.py          Pull documents from PubMed / guideline files
│   │   │   ├── chunk.py           Split documents into searchable passages
│   │   │   ├── embed.py           Turn text into vectors (3 interchangeable backends)
│   │   │   ├── store.py           Vector database (2 interchangeable backends)
│   │   │   ├── retrieve.py        Search the vector database
│   │   │   ├── rerank.py          Re-score search results for precision
│   │   │   ├── citations.py       Build citations + check if claims are grounded
│   │   │   └── pipeline.py        Wires all of the above into one object
│   │   ├── agents/               The LangGraph multi-agent workflow
│   │   │   ├── state.py            Shared data shape passed between agents
│   │   │   ├── planner.py          Agent 1: decide a search strategy
│   │   │   ├── retrieval.py        Agent 2: run the search
│   │   │   ├── answer.py           Agent 3: generate the answer
│   │   │   ├── verify.py           Agent 4: check the answer against evidence
│   │   │   └── graph.py            Wires the 4 agents into a graph + public API
│   │   ├── serving/              How the AI model is actually called
│   │   │   ├── providers.py        2 backends: local GPU vs. hosted API
│   │   │   └── prompts.py          The exact text sent to the language model
│   │   ├── eval/                  The 4-way evaluation harness (Base/FT/+RAG)
│   │   │   ├── eval_set.py          Curated test questions
│   │   │   ├── retrieval_metrics.py Recall@k, MRR
│   │   │   ├── generation_metrics.py Citation coverage, groundedness, ROUGE
│   │   │   ├── systems_metrics.py    Latency, tokens, cost
│   │   │   ├── benchmark.py         Runs all 4 configs over all questions
│   │   │   └── report.py            Renders results into tables + JSON
│   │   └── api/
│   │       └── app.py              The FastAPI web server (3 endpoints)
│   │
│   ├── data/                 ← PART B: dataset loading
│   │   ├── format.py           The exact prompt template for a medical MCQ
│   │   └── medmcqa.py          Loads + splits the MedMCQA dataset safely
│   ├── train/
│   │   └── sft.py              The QLoRA fine-tuning loop itself
│   ├── eval/
│   │   ├── harness.py          Wraps EleutherAI's lm-evaluation-harness
│   │   ├── error_analysis.py   Per-subject accuracy breakdown (base vs fine-tuned)
│   │   └── results.py          Renders the headline results table
│   ├── serve/
│   │   ├── loader.py           Loads the model (+ optional adapter) for inference
│   │   ├── score.py             Turns model output into option probabilities
│   │   ├── api.py                A second, simpler FastAPI app (MCQ scoring only)
│   │   └── cost.py               Benchmarks inference speed/memory
│   └── utils/                ← shared by Part B only
│       ├── config.py           Tiny YAML config loader (with inheritance)
│       ├── env.py              Seed-setting + GPU introspection
│       ├── tracking.py         Weights & Biases logging (optional, graceful)
│       └── model_card.py       Generates the Hugging Face model card text
│
├── scripts/                  Command-line entry points (one per task)
│   ├── rag_index.py             Build the vector database (Part A)
│   ├── rag_benchmark.py         Run the 4-way evaluation (Part A)
│   ├── train.py                 Run QLoRA fine-tuning (Part B)
│   ├── run_eval.py               Run lm-eval-harness scoring (Part B)
│   ├── error_analysis.py         Run the per-subject breakdown (Part B)
│   ├── results_table.py          Render the results table (Part B)
│   ├── push_to_hub.py            Upload the fine-tuned adapter to Hugging Face
│   ├── inference_cost.py         Benchmark serving speed (Part B)
│   ├── smoke_test.py             CPU-only structural self-check (no GPU needed)
│   └── _bootstrap.py             Shared "make imports work" helper
│
├── deploy/                   Everything needed to put Part A on the internet
│   ├── Dockerfile.api           Container image for the FastAPI backend
│   ├── render.yaml               Render.com deployment blueprint
│   ├── neon_schema.sql           The database table definition, for reference
│   ├── DEPLOY.md                 Step-by-step deployment instructions
│   └── space/                    The self-contained Gradio frontend
│       ├── app.py                   The entire UI (one file)
│       ├── requirements.txt          Its (tiny) dependency list
│       └── README.md                 Hugging Face Space metadata
│
├── configs/                   YAML configuration (Part B mostly)
│   ├── base.yaml                 Shared training defaults
│   ├── qlora_5k.yaml              "Train on 5,000 examples" (validated, real)
│   ├── qlora_20k.yaml             "Train on 20,000 examples" (future work)
│   ├── qlora_50k.yaml             "Train on 50,000 examples" (future work)
│   ├── qlora_full.yaml            "Train on the whole dataset" (future work)
│   └── corpus.yaml                Which PubMed searches build Part A's corpus
│
├── lm_eval_tasks/              Custom task definitions for lm-evaluation-harness
│   ├── medmcqa_val.yaml           How to score the medical MCQ benchmark
│   ├── pubmedqa_ood.yaml          How to score the out-of-domain benchmark
│   └── utils.py                    Python helper functions the YAML files call
│
├── notebooks/
│   └── run_colab.ipynb           A ready-to-run Google Colab notebook (Part B)
│
├── tests/                      Automated tests (29 tests, all CPU-only)
│   ├── _fakes.py                  Fake embedder + fake LLM (no downloads needed)
│   ├── test_rag.py                Tests for the RAG pipeline (Part A)
│   ├── test_agents.py             Tests for the 4-agent workflow (Part A)
│   └── test_eval.py               Tests for the evaluation harness (Part A)
│
├── .github/workflows/ci.yml    GitHub Actions: runs all tests on every push
│
├── docs/
│   ├── ARCHITECTURE.md           A shorter architecture summary
│   ├── EXPERIMENT_PLAN.md        Part B's research methodology write-up
│
├── requirements.txt              Part B's dependencies (includes PyTorch)
├── requirements-assistant.txt    Part A's FULL dependencies (index-building machine)
├── requirements-api.txt          Part A's LEAN dependencies (the live server)
├── Dockerfile                    Part B's container image (GPU training/eval)
├── reproduce.sh                  One-command reproduction of Part B's 5K run
├── .env.example                  Template for local secrets (never committed)
└── README.md                     The public-facing project overview
```

### Why is there more than one `requirements*.txt`?

This is a deliberate, load-bearing decision, not clutter. A 7B-parameter model
plus PyTorch plus `sentence-transformers` is **gigabytes** of dependencies and
needs a GPU to run fast. The free Render server has **512MB of RAM and no GPU**.
So the *live backend* (`requirements-api.txt`) intentionally excludes PyTorch
entirely — its embedding model runs through `fastembed` (a tiny CPU-only ONNX
runtime) and its language model runs through Hugging Face's *hosted* Inference
API instead of locally. The *index-building* machine (`requirements-assistant.txt`,
run once, offline, on your own laptop or Colab) uses the fuller
`sentence-transformers` stack because build-time speed matters there and there's
no 512MB constraint. Part B's `requirements.txt` is a third, separate list because
Part B needs PyTorch + `bitsandbytes` + `peft` + `trl` for actual model training,
which the live web server never does.

---

## 7. Core Concepts, Beginner → Advanced

Every advanced term used elsewhere in this document is defined here once, with a
simple explanation, a technical explanation, an analogy, and a pointer to where it
appears in this actual codebase.

### RAG (Retrieval-Augmented Generation)

- **Simple:** Instead of asking an AI a question and hoping it remembers the
  right facts, you first go find the right facts yourself (from a library of real
  documents), hand them to the AI along with the question, and tell it to answer
  using only what you handed it.
- **Technical:** A query is embedded into a vector, compared against a database of
  pre-embedded document chunks using similarity search, and the top-k most similar
  chunks are concatenated into the LLM's prompt as context, conditioning the
  generation on retrieved evidence rather than relying solely on the model's
  parametric (trained-in) knowledge.
- **Analogy:** An open-book exam versus a closed-book exam. RAG makes the AI take
  an open-book exam every time.
- **In this codebase:** `src/assistant/rag/pipeline.py`'s `retrieve_context()`
  method, called by the `retrieval` agent (`src/assistant/agents/retrieval.py`).

### Embeddings

- **Simple:** A way of turning a sentence into a list of numbers such that
  sentences with similar *meaning* end up with similar numbers, even if they use
  completely different words.
- **Technical:** A neural network (here, `BAAI/bge-small-en-v1.5`) maps text to a
  fixed-length vector (384 numbers, i.e. dimensions) in a learned space where
  cosine distance approximates semantic similarity.
- **Analogy:** GPS coordinates for meaning. "Metformin lowers blood sugar" and
  "Metformin reduces glucose levels" land at nearly the same "location" even
  though they share only one word.
- **In this codebase:** `src/assistant/rag/embed.py` — three interchangeable
  implementations (`Embedder`, `HFInferenceEmbedder`, `FastEmbedEmbedder`),
  selected by `config.embedding_provider`.

### Vector database / vector store

- **Simple:** A database specialized in answering "which of these million stored
  items is most similar to this new item?" instead of "find the row where
  `id = 5`."
- **Technical:** Stores high-dimensional vectors and an index structure (this
  project uses **HNSW** — Hierarchical Navigable Small World graphs) that makes
  nearest-neighbor search sub-linear instead of comparing against every row.
- **Analogy:** A library organized by *topic proximity* on the shelf, rather than
  alphabetically by author — so browsing near one relevant book quickly surfaces
  other relevant books.
- **In this codebase:** `src/assistant/rag/store.py` — two backends,
  `LocalVectorStore` (plain NumPy, for dev/tests) and `PgVectorStore` (Neon
  Postgres + the `pgvector` extension, for production).

### Chunking

- **Simple:** Cutting a long document into smaller, self-contained pieces so the
  search engine can find "the one paragraph that answers this" instead of
  returning an entire 10-page document.
- **Technical:** Sentence-boundary-aware splitting into ≤512-character segments
  with a small character overlap between consecutive chunks so a fact split
  across a chunk boundary isn't lost.
- **In this codebase:** `src/assistant/rag/chunk.py`'s `chunk_document()`.

### Reranking (bi-encoder vs. cross-encoder)

- **Simple:** First cast a wide net cheaply, then take a slower, closer look at
  just the top candidates to pick the best ones.
- **Technical:** A **bi-encoder** (the embedding model above) encodes the query
  and every document independently, so documents can be pre-computed and searched
  in milliseconds — but it can't reason about the query and document *together*.
  A **cross-encoder** feeds the query and a candidate document into the model
  *jointly*, producing a much more accurate relevance score, but it's too slow to
  run against an entire database — so it only reranks the ~20 candidates the
  bi-encoder already found.
- **In this codebase:** `src/assistant/rag/rerank.py`'s `Reranker` class, wired
  into `RAGPipeline.retrieve_context()` (on by default in config, **off** in the
  live free-tier deployment — see [§17](#17-design-decisions-and-trade-offs)).

### LLM (Large Language Model)

- **Simple:** A very large next-word-prediction program that, after being shown
  enormous amounts of text, can write coherent, useful answers to questions.
- **Technical:** Here, `Qwen/Qwen2.5-7B-Instruct` — a 7-billion-parameter
  transformer, instruction-tuned to follow chat-style prompts.
- **In this codebase:** Called through `src/assistant/serving/providers.py`.

### Fine-tuning, LoRA, and QLoRA

- **Simple:** Fine-tuning is "teaching an already-smart model a new specific
  skill" instead of building a new model from scratch. LoRA is a cheap way to do
  that teaching without touching (or storing another full copy of) all the
  model's original knowledge. QLoRA does the same thing on a *compressed*
  (quantized) copy of the model so it fits on a much smaller, cheaper GPU.
- **Technical:** LoRA (Low-Rank Adaptation) freezes the pretrained weight
  matrices and injects small, trainable low-rank matrices (`A`, `B`, where the
  update is `ΔW = A·B`) alongside them, drastically reducing the number of
  trainable parameters. QLoRA additionally loads the frozen base model in 4-bit
  precision (here, the `nf4` quant type with double quantization), so a 7B model
  that would normally need ~28GB of GPU memory in full precision fits in the
  16GB of a free Google Colab T4 GPU.
- **In this codebase:** `src/train/sft.py`'s `train()` function. The result: only
  **0.92% of the model's parameters (40.4 million out of 4.39 billion)** were
  actually trained. See [§9](#9-part-b-deep-dive-the-qlora-research-pipeline).

### Quantization

- **Simple:** Storing each number in a model using fewer bits (less precision),
  to save memory, at some small cost in accuracy.
- **Technical:** Converting weights from 16/32-bit floating point down to 4-bit
  representations (via `bitsandbytes`' `BitsAndBytesConfig`), with compute
  happening in a higher-precision "compute dtype" (fp16 here, since the T4 GPU
  has no native bf16 support) even though storage is 4-bit.
- **In this codebase:** `configs/base.yaml`'s `quant:` section;
  `src/train/sft.py` and `src/serve/loader.py` both build a `BitsAndBytesConfig`.

### Multi-agent workflows / LangGraph

- **Simple:** Instead of asking one AI to do everything in one giant, confusing
  instruction ("search, then answer, then double-check yourself, all at once"),
  break the job into separate specialized steps that each do one thing well and
  hand their result to the next step — like an assembly line.
- **Technical:** A directed graph of nodes, where each node is a function that
  receives and returns a partial state dictionary; **LangGraph** compiles this
  into a `StateGraph` that executes nodes in the specified order (here, a purely
  linear chain with no branching or loops, so it is *structurally guaranteed to
  terminate* — it cannot get stuck in an infinite agent loop).
- **In this codebase:** `src/assistant/agents/graph.py`'s `build_langgraph()` and
  `AssistantService`. See [§8.5](#85-the-agent-workflow-langgraph) for the full
  walkthrough of all 4 agents.

### Grounding / hallucination / citation verification

- **Simple:** "Grounding" means checking that every claim an AI makes is actually
  backed up by real evidence, not invented ("hallucinated").
- **Technical:** This project scores each answer sentence against the retrieved
  passages using **cosine similarity between sentence embeddings** (the default,
  "semantic" method) — if a sentence's embedding is close enough (≥0.6 cosine
  similarity) to *any* retrieved passage's embedding, it's marked supported. A
  simpler fallback method compares raw word overlap (Jaccard similarity of
  tokens) instead, which is faster but was found to produce false negatives on
  correctly-paraphrased answers (see [§17](#17-design-decisions-and-trade-offs)
  for the exact story of that upgrade).
- **In this codebase:** `src/assistant/rag/citations.py`'s
  `verify_claims_semantic()` (default) and `verify_claims()` (lexical fallback),
  called by the `verify` agent (`src/assistant/agents/verify.py`).

### REST API

- **Simple:** A set of URL "commands" a computer program exposes over the
  internet, so other programs can ask it to do things by sending simple web
  requests (the same technology your browser uses to load a page).
- **Technical:** An HTTP-based interface where resources are addressed by URL
  paths and manipulated via standard HTTP methods (`GET` to read, `POST` to
  create/act), with structured request/response bodies (JSON here).
- **In this codebase:** `src/assistant/api/app.py` — 3 endpoints, fully documented
  in [§15](#15-the-api-reference).

### Async / synchronous execution

- **Simple:** "Synchronous" means one thing happens, then the next, in strict
  order, each one waiting for the previous to finish. "Async" means a program can
  start a slow task (like a network request) and go do other useful work while
  waiting for it to finish.
- **Technical in this project:** The FastAPI endpoints here are defined as
  regular (synchronous) `def` functions, not `async def` — a deliberate,
  honest simplicity choice, since each request already involves a chain of
  genuinely sequential steps (retrieve → generate → verify) with no independent
  work to parallelize, and FastAPI runs sync route handlers in a thread pool
  automatically, so this doesn't block the server for other requests. The one
  `async` function in the codebase is the CORS/timing middleware
  (`request_context` in `app.py`), which *must* be `async` because it wraps
  FastAPI's own request pipeline.

### CORS (Cross-Origin Resource Sharing) and middleware

- **Simple:** A browser security rule that blocks a web page from one website
  from silently calling an API on a *different* website, unless that API
  explicitly says "it's OK, I allow requests from anywhere/this origin."
  "Middleware" is code that runs automatically before and after *every* request,
  no matter which endpoint was called.
- **In this codebase:** `app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)`
  in `src/assistant/api/app.py` — needed because the Gradio frontend
  (`huggingface.co`) and the FastAPI backend (`onrender.com`) are different
  domains. The custom `request_context` middleware in the same file logs every
  request and stamps a request ID + timing header on every response.

### Latency percentiles (p50 / p95)

- **Simple:** p50 ("median") is "half of all requests were faster than this."
  p95 means "95% of requests were faster than this — this catches the slow,
  unlucky ones the average would hide."
- **In this codebase:** `src/assistant/eval/systems_metrics.py`'s `_pct()`
  helper, used to compute `e2e_latency_ms_p50`/`p95` across benchmark runs.

---

## 8. Part A Deep Dive: The Production Assistant

### 8.1 Configuration (`config.py`)

**File:** `src/assistant/config.py` · **Imports it:** almost every other file in
`src/assistant/` · **Executes:** once per process, lazily, the first time
`get_settings()` is called.

This file defines one class, `Settings`, built on **Pydantic Settings**
(`pydantic_settings.BaseSettings`). Every field has a default value, and every
field can be overridden by an environment variable prefixed `BIOMED_` (e.g.
`BIOMED_VECTOR_BACKEND=pgvector`), or by a local `.env` file. This is why the
exact same Python code runs three different ways — locally with a NumPy store
and no LLM key, in a Colab notebook building the real Neon index, and in
production on Render — with **zero code changes**, only environment variables.

Key settings groups (each explained in context below):
- **Retrieval/embeddings** — which embedding model, chunk size, how many
  candidates to retrieve before/after reranking, and (new this pass)
  `grounding_method`/`grounding_threshold` for the citation-verification agent.
- **Vector store** — `local` (NumPy files on disk) or `pgvector` (Neon Postgres).
- **LLM serving** — which base model, which adapter repo, and whether inference
  runs `local`ly (GPU + adapter) or via `hf_inference` (hosted, base model only);
  for `hf_inference`, `hf_providers` (ordered failover list, default
  `featherless-ai,auto`, parsed by the `hf_provider_list` property) and
  `inference_timeout_s` (default 60).
- **API protection** — `rate_limit_per_minute` (default 20 `/query` calls per
  client per minute; `0` disables).
- **Ingestion** — an optional email/API key for the NCBI PubMed API (raises
  rate limits; not required).

`get_settings()` implements the **singleton pattern**: a module-level `_settings`
variable is created once and reused, so environment variables are read exactly
once per process (important, because re-reading them mid-run could produce
inconsistent behavior if something mutated the environment).

```python
def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

**Why Pydantic Settings instead of hand-rolled `os.environ.get()` calls
everywhere?** Type coercion (a `Path` field automatically becomes a real
`pathlib.Path`, an `int` field is validated as an integer), one central place to
see every configurable knob, and IDE autocomplete on `cfg.embedding_model` instead
of typo-prone string keys.

### 8.2 Logging (`logging.py`)

**File:** `src/assistant/logging.py`.

Rather than pulling in a third-party logging framework, this file implements a
small custom `JsonFormatter` (subclassing Python's built-in `logging.Formatter`)
that turns every log line into a single JSON object — `{"ts": ..., "level": ...,
"logger": ..., "msg": ..., ...any extra fields...}`. This matters in production:
Render's log viewer (and most cloud logging systems) can parse and filter JSON
lines, but not free-form text.

The `timed` class is a **context manager** (implements `__enter__`/`__exit__`) —
used as `with timed(log, "retrieve", query=q) as t: ...` — that measures wall-clock
time around a block of code and automatically logs it on exit. This is how every
`retrieve_ms` and `rerank_ms` timing in the final API response is actually
produced (see `RAGPipeline.retrieve_context()` in the next section).

**Why a context manager instead of manually calling `time.perf_counter()` twice
everywhere?** It guarantees the timing is logged even if an exception is raised
inside the block (`__exit__` still runs), and it removes repetitive
start/stop/log boilerplate from every call site.

### 8.3 Data Models (`schema.py`)

**File:** `src/assistant/schema.py` · every module in `rag/`, `agents/`, `eval/`,
and `api/` imports from here.

This file defines the **typed contracts** between every stage of the pipeline,
using Pydantic `BaseModel` classes:

| Model | Represents | Produced by | Consumed by |
|---|---|---|---|
| `Document` | One raw source (an abstract or a guideline) | `rag/ingest.py` | `rag/chunk.py` |
| `Chunk` | One searchable slice of a `Document` | `rag/chunk.py` | `rag/embed.py`, `rag/store.py` |
| `EmbeddedChunk` | A `Chunk` + its vector | `rag/pipeline.py` | `rag/store.py` |
| `RetrievedPassage` | A `Chunk` returned by search, with a similarity score | `rag/retrieve.py`, `rag/rerank.py` | `rag/citations.py`, agents |
| `Citation` | A numbered `[n]` reference with a supported/unsupported flag | `rag/citations.py` | the API response, the UI |
| `GroundedAnswer` | The final, complete answer object | `agents/graph.py` | the API, the eval harness |

**Why bother with typed models instead of plain Python dictionaries?**
1. **Self-documentation** — anyone reading `GroundedAnswer`'s field list instantly
   knows the shape of an API response, without reading the API code.
2. **Free serialization** — FastAPI and Pydantic automatically turn a
   `GroundedAnswer` object into correct JSON for the HTTP response (this is what
   `ans.model_dump()` does in `api/app.py`).
3. **Validation** — a bug that tries to put a string where a `float` (a
   `support_score`) is expected fails immediately and loudly, rather than
   silently corrupting data three steps later.

This is the same *category* of tool as Zod in a TypeScript project (mentioned
throughout general documentation templates) — Pydantic is Python's equivalent:
schema-validated data at the boundaries of the system.

### 8.4 The RAG Pipeline

The RAG pipeline is the "search the library" half of the system. It lives across
7 small files in `src/assistant/rag/`, each doing exactly one job, tied together
by `pipeline.py`.

#### `ingest.py` — getting real documents

Two data sources, both producing the same `Document` shape:
- **`fetch_pubmed(query, retmax, email, api_key)`** calls the NCBI **E-utilities**
  API in two steps: `esearch` (turn a query like
  `(sepsis[tiab] OR "septic shock"[tiab]) AND (diagnosis[tiab] OR …)` into a list
  of PubMed IDs, ranked by `sort=relevance` — PubMed's "Best Match"; the API's
  default is newest-first) then `efetch` (download the abstract XML for those IDs,
  sent as a POST because hundreds of IDs overflow a GET URL). The XML is parsed
  with Python's built-in `xml.etree.ElementTree` to pull out the title, abstract
  text, year, and journal. `time.sleep(0.34)` between calls respects NCBI's
  ~3-requests/second limit for anonymous callers, and 429/5xx responses are
  retried with backoff.

  **A query-syntax bug that shaped the whole demo.** The original queries were
  written as `"hypertension treatment adults[Title/Abstract]"`. PubMed applies a
  field tag to the *whole* quoted string, so that is an **exact-phrase** search:
  it matched **1** abstract in all of PubMed (diabetes: **3**). Ten topics × a
  400-result cap produced only 733 abstracts, mostly from the few phrases that
  happened to be common — and the demo correctly refused to answer basic
  hypertension or metformin questions because the index had nothing on them.
  The queries are now field-scoped term logic (each term tagged `[tiab]`,
  combined with AND/OR), so the same hypertension topic matches ~22,000 reviews
  and guidelines.
- **`load_guidelines(corpus_dir)`** reads plain `.txt` files placed under
  `data/corpus/guidelines/{nih,who,cdc}/` — a low-tech but honest way to include
  curated guideline text without needing to solve PDF parsing.
- **`iter_sample_documents()`** yields **3 hand-written** documents (about
  metformin, WHO hypertension guidance, and CDC vaccine schedules) — used only by
  the test suite and `--sample` script modes, so CI never needs network access or
  a real API key.

#### `chunk.py` — splitting into searchable pieces

`chunk_document()` uses a greedy, sentence-aware algorithm: split the document
into sentences with a regex (`_SENT`), then pack sentences one at a time into a
running buffer until adding the next sentence would exceed `chunk_size`
(512 characters by default), at which point the current buffer is finalized as a
`Chunk` and a new buffer starts, seeded with the last `overlap` (64) characters
of the *previous* chunk — so a fact split across a chunk boundary still appears,
in part, in both chunks.

#### `embed.py` — turning text into vectors

Three interchangeable classes, all exposing the same `embed_documents(texts)` /
`embed_query(text)` interface (this is the **strategy pattern**: different
algorithms, one shared interface, chosen at runtime by config):

1. **`Embedder`** — wraps `sentence-transformers`' `SentenceTransformer`, loading
   `BAAI/bge-small-en-v1.5` lazily (only on first real use, so importing this
   module doesn't require PyTorch to be installed — important for CI). Used for
   offline index-building.
2. **`HFInferenceEmbedder`** — calls Hugging Face's hosted feature-extraction
   API instead of running a model locally.
3. **`FastEmbedEmbedder`** — runs the *same* `bge-small` model, but through
   `fastembed`, a library that runs models as compiled **ONNX** graphs instead of
   PyTorch — meaning zero PyTorch dependency. This is the one actually used by
   the live, free-tier backend.

All three L2-normalize their output vectors (`_l2()` divides a vector by its own
length), which is what allows the vector store to use a plain dot product as
cosine similarity (`self._vecs @ query_vec` in `LocalVectorStore.search()`) — a
small but important mathematical shortcut: for unit vectors, dot product **is**
cosine similarity, so no extra normalization step is needed at search time.

#### `store.py` — the vector database, two backends

Covered fully in [§10](#10-the-database-layer).

#### `retrieve.py` and `rerank.py` — searching and refining

`Retriever.retrieve()` is a two-line function: embed the query, then call
`store.search()`. All the real complexity lives inside whichever store backend is
active. `Reranker.rerank()` optionally re-scores the top candidates with a
cross-encoder model (`BAAI/bge-reranker-base`) for higher precision, at the cost
of needing PyTorch — which is why it's disabled in the free-tier deployment
(`BIOMED_USE_RERANKER=false` in `render.yaml`).

#### `citations.py` — building citations and verifying them

`build_citations()` turns the top-ranked passages into numbered `Citation`
objects (`[1]`, `[2]`, ...), each carrying the source title, URL, and a short
quoted excerpt. `verify_claims_semantic()` (the default) and `verify_claims()`
(the lexical fallback) are covered in depth under "Grounding" in
[§7](#7-core-concepts-beginner--advanced) — this is the code that produces the
`claims` list and `all_claims_supported` flag on every `GroundedAnswer`.

Three helpers in this file exist because of things seen in live output:

- **`dedupe_passages()`** — PubMed republishes some abstracts under several IDs
  (a review plus its "Points & Pearls" digest, for example). For one sepsis
  question, 4 of the 5 retrieved passages were the same text, so the model was
  effectively reading one source. This keys each passage on its first 300
  normalized characters, drops repeats, and re-numbers ranks. It runs *before*
  the top-k cut, so the 5 slots go to 5 distinct texts.
- **`normalize_markers()`** — rewrites `[1-3]` → `[1][2][3]` and `[1, 3]` →
  `[1][3]`, dropping numbers beyond the source count. The answer agent applies
  it right after generation, so the verifier, the citation-coverage metric, and
  the UI all see one format. Plain text like "1-3 mg" is untouched (only
  bracketed groups match).
- **`is_abstention()`** — "The sources do not contain enough evidence" is not a
  factual claim, yet it scored 0.68 against a sepsis passage (it *talks about*
  sepsis) and was marked "supported". Such sentences are now excluded from
  `claims`; an answer that only abstains gets `all_claims_supported = None`
  (neither pass nor fail) instead of a misleading green check.
- **`strip_stray_abstention()`** — the opposite failure: after the corpus grew,
  the model sometimes gave a full cited answer and then appended "The provided
  sources do not contain enough evidence," contradicting itself. The answer
  agent removes refusal sentences when at least one cited, non-refusal sentence
  exists; a refusal that *is* the whole answer is kept.

**Reusing stored vectors.** Semantic verification compares each claim against
each passage, which used to mean embedding all 5 passages again on every query —
on Render's fraction of a CPU, the most expensive step that wasn't being timed.
The stores now return each passage's stored vector on
`RetrievedPassage.embedding` (pgvector: `SELECT …, embedding`), and
`verify_claims_semantic()` uses them when present, embedding only the claim
sentences. The field is declared `exclude=True`, so vectors never appear in API
responses (a test asserts this).

#### `pipeline.py` — `RAGPipeline`, the single entry point

This class's constructor wires together an embedder, a vector store, and an
optional reranker, all chosen based on the `Settings` object passed in (or the
global singleton if none is given). It exposes exactly two operations the rest
of the system needs:

```python
def build_index(self, documents: list[Document]) -> int:
    # chunk -> embed -> store, for every document
def retrieve_context(self, query: str, metadata_filter=None):
    # embed query -> search store -> dedupe -> (optional) rerank / top-k -> citations
    # returns (passages, citations, {"retrieve_ms": ..., "rerank_ms": ...})
```

Notably, `RAGPipeline` never generates text — it is deliberately
**generation-agnostic**, so retrieval quality can be measured and tested
completely independently of which language model eventually writes the answer.
This separation of concerns is why `tests/test_rag.py` can fully exercise
indexing and retrieval using a fake, instant, deterministic embedder with zero
network calls or model downloads.

### 8.5 The Agent Workflow (LangGraph)

This is the "figure out how to search, search, answer, then fact-check yourself"
half of the system — 4 small agent files plus a shared state definition and a
graph assembler.

#### Why agents instead of one big prompt?

A single giant prompt ("search for X, then write an answer, then check your own
answer, all in one response") asks the language model to do three
fundamentally different jobs at once, with no way to inspect or test any one job
in isolation, and no structural guarantee it won't try to "help more" than asked
(e.g., inventing extra citations). Splitting into 4 agents means:
- Each agent is independently **unit-testable** (see `tests/test_agents.py`).
- The verification step is **not** something the answering model can skip or
  fake — it's a separate, deterministic computation over the same evidence.
- The whole workflow is **structurally bounded** — see below.

#### `state.py` — the shared "clipboard"

`AgentState` is a `TypedDict` (a plain dictionary with a declared, type-checked
shape) carrying everything agents need to pass to each other: the query, the
chosen retrieval strategy, the metadata filter, the retrieved passages and
citations, the generated answer, the per-claim verification results, latency,
and token usage. Each agent function reads what it needs from this dict and
returns a **partial update** (only the new/changed keys), which the graph merges
into the running state.

`Deps` is a small `dataclass` carrying the *runtime objects* agents need but that
aren't data (the `RAGPipeline` instance, the LLM provider, and whether this
particular serving configuration uses RAG at all). Separating "data that flows
through the graph" (`AgentState`) from "tools the graph needs to do its job"
(`Deps`) is a common, clean pattern for keeping agent functions pure and testable
— an agent function's signature is always `(state, deps) -> partial_state`.

#### Agent 1 — `planner.py`

**Purpose:** decide *how* to search, before searching. **No LLM call** — this is
a fast, free, deterministic heuristic, which also makes it trivially testable
(see `test_planner_source_filter` in `tests/test_agents.py`).

```python
def plan(state: AgentState, deps: Deps) -> AgentState:
    ql = state["query"].lower()
    sources = [s for kw, s in _SRC_KEYWORDS.items() if kw in ql]   # "cdc" in text?
    if "guideline" in ql and not sources:
        sources = ["nih", "who", "cdc"]
    metadata_filter = {}
    if sources: metadata_filter["source"] = sources
    if any(w in ql for w in _RECENCY): metadata_filter["year_min"] = 2018
    return {"plan": {"needs_retrieval": deps.use_rag, "strategy": ...},
            "metadata_filter": metadata_filter or None}
```

If you ask *"What does the CDC recommend for adult vaccines?"*, the planner
notices "cdc" in the question and sets `metadata_filter = {"source": ["cdc"]}`
so the retrieval step only searches CDC documents — a real, working example of
metadata-filtered retrieval, not just semantic filtering.

#### Agent 2 — `retrieval.py`

**Purpose:** run the actual search, using the plan from Agent 1.

```python
def retrieve(state: AgentState, deps: Deps) -> AgentState:
    if not deps.use_rag or not state.get("plan", {}).get("needs_retrieval", True):
        return {"passages": [], "citations": [], "latency_ms": ...}   # skip entirely
    passages, citations, lat = deps.pipeline.retrieve_context(
        state["query"], metadata_filter=state.get("metadata_filter"))
    return {"passages": passages, "citations": citations, "latency_ms": {**old, **lat}}
```

For "non-RAG" serving configurations (`base` or `ft` — see
[§8.6](#86-the-serving-layer)), this agent is a no-op: it returns empty passages
immediately, so the answer agent falls back to the model's parametric knowledge.
This one `if` statement is what lets the **same 4-agent graph** serve all four
benchmark configurations (Base / Fine-tuned / Base+RAG / Fine-tuned+RAG) — the
difference is entirely in what `Deps.use_rag` is set to, not in different code
paths.

#### Agent 3 — `answer.py`

**Purpose:** generate the actual answer text.

```python
def answer(state: AgentState, deps: Deps) -> AgentState:
    passages = state.get("passages", [])
    context = deps.pipeline.format_context(passages) if passages else None
    messages = build_answer_messages(state["query"], context)   # see prompts.py
    t0 = time.perf_counter()
    result = deps.provider.generate(messages)                    # the LLM call
    return {"answer": normalize_markers(result.text, len(passages)),  # [1-3] -> [1][2][3]
            "latency_ms": {..., "generate_ms": ...},
            "token_usage": {...}}
```

`format_context()` (in `pipeline.py`) numbers the passages `[1]`, `[2]`, ... in
the exact same order citations were built, so when the LLM is told "cite with
`[n]`," those markers line up with the `Citation` objects the verify step will
check against.

#### Agent 4 — `verify.py`

**Purpose:** the "fact-check yourself" step — covered in depth as "Grounding" in
[§7](#7-core-concepts-beginner--advanced). One implementation detail worth
calling out: after scoring each sentence, this agent also back-annotates the
`Citation` objects themselves (`cit.supported = ...`, `cit.support_score = ...`),
so the final API response's `citations` list — not just its `claims` list — shows
which *sources*, specifically, actually got used to ground the answer. It also
records its own `verify_ms` in `latency_ms`, so every stage of a query is now
timed (`retrieve_ms`, `generate_ms`, `verify_ms`) — before, verification was
the invisible part of the wall-clock time.

#### `graph.py` — wiring it together

```python
def build_langgraph(deps: Deps):
    from langgraph.graph import END, StateGraph
    g = StateGraph(AgentState)
    g.add_node("planner", lambda s: planner_agent.plan(s, deps))
    g.add_node("retrieval", lambda s: retrieval_agent.retrieve(s, deps))
    g.add_node("answer", lambda s: answer_agent.answer(s, deps))
    g.add_node("verify", lambda s: verify_agent.verify(s, deps))
    g.set_entry_point("planner")
    g.add_edge("planner", "retrieval")
    g.add_edge("retrieval", "answer")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    return g.compile()
```

Every edge points strictly forward to the next agent, ending at LangGraph's
built-in `END` sentinel. **There are no cycles.** This means the graph is
structurally incapable of looping forever — a real, common failure mode of more
open-ended "agentic" systems where a model decides for itself when it's "done."

`AssistantService` is the class everything else in the codebase actually talks
to. Its constructor tries to compile a real LangGraph graph; if the `langgraph`
package isn't installed (or fails for any reason), it falls back to
`_sequential()` — a plain Python loop calling the same 4 agent functions in
order. **This fallback exists specifically so the test suite and any CPU-only
environment can exercise the exact same agent logic with zero extra
dependencies** — LangGraph and the sequential loop are behaviorally identical,
because both call the same underlying functions.

```python
CONFIGS = {"base": (False, False), "ft": (True, False),
           "base_rag": (False, True), "ft_rag": (True, True)}
```

This one dictionary is what defines all four benchmark configurations as
combinations of two booleans: "does this configuration use the fine-tuned
adapter?" and "does this configuration use RAG?" — both the live API and the
4-way evaluation harness build their `AssistantService` instances from this same
table.

### 8.6 The Serving Layer

**Files:** `src/assistant/serving/providers.py`, `prompts.py`.

#### `providers.py` — the strategy pattern for "call an LLM"

Both classes below implement the same abstract interface,
`LLMProvider.generate(messages) -> GenerationResult`:

- **`LocalTransformersProvider`** — loads the base model (+ optionally the LoRA
  adapter) via `src/serve/loader.py` (**reused directly from Part B** — a real
  example of not rewriting working code), runs `model.generate(...)` locally, and
  decodes the output. Used when a GPU is available.
- **`HFInferenceProvider`** — calls Hugging Face's hosted `chat_completion` API
  instead. Used on the free-tier deployment, where no GPU exists. **Cannot serve
  a custom LoRA adapter** — HF's free serverless inference only serves published
  base models — which is the single most important honesty constraint in the
  whole system (fully explained in [§17](#17-design-decisions-and-trade-offs)).

  **Provider failover.** Hugging Face's router hands each request to a third-party
  *inference provider* (Together, Featherless, …). Its default, `auto`, picks one
  for you — and in September 2026 it kept routing Qwen2.5-7B to Together after
  Together stopped serving that model, so **every live query failed while
  `/health` stayed green**. `HFInferenceProvider` therefore takes an *ordered*
  list (`BIOMED_HF_PROVIDERS`, default `featherless-ai,auto`), keeps one
  `InferenceClient` per provider, and on any failure logs it and tries the next.
  Each call has a timeout (`BIOMED_INFERENCE_TIMEOUT_S`, default 60 s) so a hung
  provider can't pin a worker thread. If every provider fails it raises
  `UpstreamError`, which the API turns into a clear 502 (see §8.7).

`GenerationResult` is a tiny `dataclass` (`text`, `prompt_tokens`,
`completion_tokens`) — the common return shape both providers normalize to, so
the `answer` agent never needs to know which provider it's talking to.

#### `prompts.py` — the exact instructions given to the model

```python
SYSTEM_PROMPT = (
    "You are a careful biomedical research assistant. Answer the question using "
    "ONLY the information in the provided numbered sources. Give every specific "
    "fact the sources DO support, even if they answer the question only partly, "
    "and then briefly note what they leave out. Cite each factual claim with a "
    "single marker per source, like [1] or [2][3] — never ranges like [1-3]. "
    "Only if no source is relevant at all, reply exactly: 'The provided sources do "
    "not contain enough evidence.' Do not invent citations or facts. ..."
)
```

This single paragraph is doing a lot of work: it forbids answering from
un-cited/general knowledge, mandates the citation format the verification step
expects, still lets the model say "I don't know" — but **only when no source is
relevant** — and repeats the "not medical advice" disclaimer at the source, not
just in the UI.

*Why it was reworded:* the earlier version said "if the sources do not support
an answer, say …not enough evidence". With a strict model that became an
escape hatch: asked for the early signs of sepsis, it refused even though it had
retrieved a relevant sepsis review, and cited the sources as `[1-5]` — a range
the verifier couldn't read. The new wording asks for the *partial* answer the
sources support, plus what's missing, and forbids ranges. The answer agent also
normalizes any range that slips through (see §8.5).
`build_answer_messages()` swaps in a different, un-grounded system prompt
(`SYSTEM_PROMPT_NO_RAG`) when `context is None` — this is what makes the `base`
and `ft` (non-RAG) configurations behave honestly as "answer from memory" rather
than silently reusing the "cite your sources" instruction with no sources given.

### 8.7 The API (FastAPI)

Fully specified in [§15](#15-the-api-reference) as a reference; here is the
narrative walkthrough of `src/assistant/api/app.py`.

**Lifespan and lazy loading.** FastAPI's `lifespan` context manager runs once at
server startup and once at shutdown. On startup, this app loads `Settings` and,
if a precomputed `results/benchmark_explorer.json` file exists, loads it into
memory for the `/benchmark` endpoint. Crucially, it does **not** build the
`AssistantService` (which would load an LLM) at startup — `_STATE["service"] =
None`. The actual 7B-parameter-adjacent service is only constructed the first
time someone calls `POST /query` — this is a deliberate **lazy singleton**
pattern (see [§12](#12-state-management--what-exists-and-what-doesnt)) that keeps
server boot time fast and avoids paying model-load cost for a server that might
never receive a query.

**Middleware, in order of execution.** `CORSMiddleware` is added first so browser
cross-origin requests from the Gradio Space are allowed. The custom
`request_context` middleware then wraps every request: it generates a short
request ID (`uuid.uuid4().hex[:12]`, or reuses an incoming `X-Request-ID` header
if the client supplied one — useful for tracing a single request across services),
times the whole request/response cycle, stamps both onto the response headers,
and logs a structured line. This is the kind of observability infrastructure a
production system needs and a prototype typically skips.

**Building the service exactly once.** FastAPI runs plain `def` endpoints in a
thread pool, so two users hitting a cold server at the same moment could both
see `_STATE["service"] is None` and both build it (two embedding models in
512 MB of RAM). `_get_service()` uses **double-checked locking**: check without
the lock (fast path), take `_BUILD_LOCK`, check again, build. A test fires 8
threads at a cold service and asserts the builder ran once.

**Rate limiting.** `/query` spends the HF token on every call, so a public
endpoint with no limit is a cost hole. `RateLimiter` is an in-memory
**sliding window** per client: a `deque` of timestamps per key; drop those older
than 60 s; reject if `BIOMED_RATE_LIMIT_PER_MINUTE` (default 20, `0` = off)
remain. Rejections get **429** plus a `Retry-After` header. The client key is the
first `X-Forwarded-For` hop (Render terminates TLS at a proxy). Caveat: the
Gradio Space calls the API from its server, so all Space users share one key —
in practice this is a global budget cap, which is what the free tier needs.
In-memory state is correct only for one instance; several replicas would need
Redis.

**Error handling in `/query` — three outcomes, no leaks.**

| Failure | Status | Message | Service rebuilt? |
|---|---|---|---|
| Building the service (DB, embedder) | 503 | "starting or misconfigured (request ID)" | Yes, next call |
| `UpstreamError` — every LLM provider failed | 502 | "hosted language model unavailable (request ID)" | **No** |
| Anything else | 502 | "Query failed (request ID)" | No |

Full tracebacks go to the logs (`exc_info=True`), keyed by the same request ID
that the client sees. *History:* an early version returned opaque 500s, which
hid a real outage; the next version returned `type: message` verbatim, which
diagnosed it — but a psycopg error can contain the database connection string,
password included. The current design keeps both properties: the message is
safe, and the request ID finds the full error in the logs. It also no longer
throws away a working service when only the model is down — rebuilding meant
reloading the embedder and re-running schema DDL on every failed request.

**`last_query` on `/health`.** Liveness alone ("the process is up") is what let
the provider outage go unnoticed. Every `/query` now records
`{"ok", "failed_stage", "at"}`, and `/health` returns it, so a monitor (or a
person) can see that the last real request failed.

**`served_config(cfg)` — the single source of truth for honesty.** This tiny
function is the most important line of business logic in the entire API:

```python
def served_config(cfg) -> str:
    return "base_rag" if cfg.inference_provider == "hf_inference" else "ft_rag"
```

Both `/health` and `/query` call this function (indirectly, via `_build_service`)
to determine which label to report. There is **no separate code path** that
could disagree with reality — the API cannot claim to be serving the fine-tuned
model unless `inference_provider` is actually configured for local/GPU serving
with the adapter attached. See [§17](#17-design-decisions-and-trade-offs) for why
this matters so much.

### 8.8 The Evaluation Harness

**Files:** `src/assistant/eval/*.py`, driven by `scripts/rag_benchmark.py`.

This is the machinery that answers "is RAG actually helping? Is fine-tuning
actually helping?" with numbers instead of vibes, across all four serving
configurations.

- **`eval_set.py`** defines `EvalQuestion` (a question + a set of "gold" relevant
  document IDs + an optional reference answer for text-similarity scoring) and
  ships a tiny 3-question `sample_eval_set()` matching the offline sample corpus,
  so the whole harness runs in CI with zero network access.
- **`retrieval_metrics.py`** implements **Recall@k** (of the documents that
  *should* have been found, what fraction actually appeared in the top k
  results?) and **MRR** (Mean Reciprocal Rank — how high up the list did the
  first correct document appear? `1/rank`, so a correct result at rank 1 scores
  1.0, at rank 2 scores 0.5, etc.).
- **`generation_metrics.py`** implements **citation coverage** (what fraction of
  answer sentences carry a `[n]` marker at all?), **groundedness** (what fraction
  of claims did the verify agent mark supported? — this metric literally reuses
  the same verification result the live app shows the user), and optional,
  lazily-imported **ROUGE-L** and **BERTScore** (only computed if those
  heavier libraries are installed — they gracefully return `None` otherwise, so
  the harness never crashes in a lean environment).
- **`systems_metrics.py`** computes p50/p95 end-to-end latency, average token
  usage, and an **estimated** (never fabricated as "measured") inference cost
  from `total_tokens × price_per_1M_tokens`.
- **`benchmark.py`**'s `run_benchmark(services, questions)` runs *every* question
  through *every* configured `AssistantService` and aggregates all three metric
  families per configuration.
- **`report.py`** renders the aggregated results into three Markdown comparison
  tables (retrieval / generation / systems) for the README, and a JSON file
  (`benchmark_explorer.json`) containing the full per-question records — this
  exact file is what the Gradio "Benchmark Explorer" tab reads.

**Honesty discipline, by design:** nowhere in this harness is a metric ever
invented. If the real benchmark hasn't been run yet, the README table cells stay
literally the string `"PENDING RUN"` (see `src/eval/results.py`'s `render_table`
in Part B for the identical pattern) rather than a plausible-looking guess.

### 8.9 The Frontend (Gradio)

**File:** `deploy/space/app.py` (a second, older, superseded version lives at
`space/app.py` at the repo root — see the note at the end of this section).

#### Why Gradio and not React/Next.js?

This is a deliberate, load-bearing architectural decision, not a limitation. The
entire *point* of this frontend is to visualize a Python-native AI pipeline's
output (citations, evidence cards, per-claim verification, latency). Gradio lets
one Python file define a complete, deployable web UI with no separate build
step, no JavaScript bundler, and free hosting on Hugging Face Spaces — for a
project whose engineering value is in the RAG/agent/eval pipeline, spending
effort on a hand-rolled React frontend would be effort spent on the wrong thing.
(If you *did* want to swap in React later, nothing about the backend would need
to change — it's a plain JSON REST API, framework-agnostic on the client side.)

#### Structure — "components," Gradio's way

Gradio's `Blocks` API is the closest equivalent to "components" here. This app
defines:

- A **header** (`gr.HTML`) with the title, tagline, links, and the clinical-use
  disclaimer.
- A **live status badge** (`gr.HTML`, populated by `demo.load(backend_status, ...)`
  — Gradio's equivalent of a React "on mount" effect) showing which config
  (`Base + RAG` or `Fine-tuned + RAG`) the backend is *actually* serving right now.
- **Tab 1, "Ask"** — a `Textbox` + `Button`, a row of `gr.Examples` (pre-filled
  example questions users can click), and two output panels: the rendered answer
  (`answer_out`) and the retrieved evidence cards (`evidence_out`).
- **Tab 2, "Benchmark Explorer"** — a `Dropdown` of curated questions and a table
  rendering the precomputed 4-way comparison for whichever question is selected.
- An **Accordion** ("How it works") explaining the pipeline, the honest
  Base+RAG-vs-Fine-tuned+RAG serving story, and known limitations, collapsed by
  default so it doesn't clutter the first-time experience.

#### State, event wiring, and interaction (Gradio's answer to "hooks")

Gradio has no React-style hooks; instead, **event bindings** connect a UI
trigger (a click, a dropdown change, page load) to a Python function and a list
of output components to update with that function's return value:

```python
btn.click(lambda: (LOADING, EMPTY_EVIDENCE), outputs=[answer_out, evidence_out]) \
   .then(ask, inputs=q, outputs=[answer_out, evidence_out])
```

This is a genuinely important UX detail: clicking "Ask" first **immediately**
swaps both panels to a loading state (the spinner + "Analyzing the literature…"
copy), *then* (`.then(...)`) calls the real `ask()` function, which can take
several seconds. Without the first step, the UI would appear frozen with no
feedback between the click and the eventual answer — exactly the kind of
first-class **loading state** the task explicitly asked for. `q.submit(...)` does
the identical thing for pressing Enter in the textbox.

#### `ask()` — the function that talks to the backend

```python
def ask(question: str):
    if not question.strip():
        return EMPTY_ANSWER, EMPTY_EVIDENCE                 # empty state
    if not BACKEND_URL:
        return "⚠️ No backend configured...", ""             # misconfiguration state
    try:
        r = httpx.post(f"{BACKEND_URL}/query", json={"question": question.strip()}, timeout=120)
        r.raise_for_status()
        d = r.json()
    except httpx.HTTPStatusError as e:
        return f"⚠ Backend error ({e.response.status_code})...", ""   # error state
    except Exception:
        return "⏳ The backend is waking up...", ""                    # cold-start state
    return _render_answer(d), _render_evidence(d.get("passages", []), d.get("citations", []))
```

Every realistic failure mode has its own explicit, human-readable state: no
question typed, no backend configured, the backend returned an HTTP error, or
the backend is unreachable (which on a free-tier host almost always means
"asleep, waking up now"). This is the difference between a demo that quietly
breaks and one that tells the truth about what's happening.

#### `_render_answer()` and `_render_evidence()` — turning JSON into HTML

These two functions hand-build small, targeted HTML strings (escaped with
Python's `html.escape()` everywhere user- or model-generated text is inserted,
to prevent HTML/script injection from a malicious or malformed model output) —
a metadata strip (served config, total latency, retrieve/generate sub-latencies,
token count), the answer body with `[n]` citation markers styled as superscript
spans, a green/amber verification badge, a collapsible per-claim breakdown, and
source-badged evidence cards (color-coded by source: PubMed, WHO, CDC, NIH) each
linking out to the real source URL.

#### A note on `space/app.py` vs. `deploy/space/app.py`

The repository contains **two** Gradio app files: an older one at
`space/app.py` (a simple single-purpose "answer a 4-option MCQ" demo, left over
from an earlier iteration of the project, before the RAG/agent system existed)
and the current, actually-deployed one at `deploy/space/app.py` (everything
described above). The older file is **not used by the live deployment** — it's a
leftover that should eventually be deleted; documenting this honestly here rather
than silently ignoring it.

---

## 9. Part B Deep Dive: The QLoRA Research Pipeline

This is the original research question the whole project grew from: **"Does
fine-tuning `Qwen2.5-7B-Instruct` on medical multiple-choice questions actually
make it better at answering them, and how much does that cost?"**

### 9.1 Data (`src/data/`)

**`format.py`** defines the *single* canonical way a medical MCQ becomes text —
used identically by both training (`src/train/sft.py`) and evaluation
(`lm_eval_tasks/utils.py` deliberately mirrors it) so the comparison between the
base and fine-tuned model is apples-to-apples:

```python
def render_question(question, options):
    lines = [f"Question: {question.strip()}"]
    for letter, opt in zip(["A","B","C","D"], options):
        lines.append(f"{letter}. {opt.strip()}")
    lines.append("Answer:")
    return "\n".join(lines)
```

**`medmcqa.py`**'s `load_train_val()` loads the `openlifescienceai/medmcqa`
dataset (~194,000 questions total) and does something easy to get wrong:
it filters to only rows with a valid answer index, shuffles under a **fixed
seed** (so the split is reproducible), then carves off a `val_size` slice (for
measuring training loss, never for the accuracy benchmark) from the remaining
pool, and takes the *first* `train_size` items of what's left as the actual
training set. The file's own docstring states the leakage-safety guarantee
explicitly: **training uses MedMCQA's `train` split only; scoring uses the
official `validation` split; there is zero overlap.** The official `test` split
is never used at all, because MedMCQA ships it with hidden labels
(`cop = -1`), so it cannot be scored locally.

### 9.2 Training (`src/train/sft.py`)

`train(cfg)` is the actual fine-tuning loop, using Hugging Face's `transformers`,
`peft`, and `trl` libraries:

1. **Load the 4-bit quantized base model** via a `BitsAndBytesConfig` (nf4
   quantization, double quantization, fp16 compute dtype — T4-safe).
2. **Attach a LoRA adapter** via `peft.LoraConfig` (rank 16, alpha 32, applied to
   all 7 attention/MLP projection matrices — `q_proj`, `k_proj`, `v_proj`,
   `o_proj`, `gate_proj`, `up_proj`, `down_proj`) and `get_peft_model()`.
3. **Print the trainable-parameter count** — this is where the real,
   measured "0.92% of parameters trainable" number comes from
   (`_count_trainable()` sums `p.numel()` over parameters with
   `requires_grad=True` vs. all parameters).
4. **Train** using TRL's `SFTTrainer` (Supervised Fine-Tuning Trainer), with
   gradient checkpointing (trades compute for memory — recomputes activations
   during the backward pass instead of storing them, essential for fitting a 7B
   model's training on a 16GB GPU), fp16 mixed precision, and a paged 8-bit
   AdamW optimizer (another memory-saving trick specific to QLoRA training).
5. **Checkpoint and resume.** `_latest_checkpoint()` scans `output_dir` for the
   highest-numbered `checkpoint-N` folder and, if found, resumes training from
   it. This exists specifically because the target environment (a free Google
   Colab session) can disconnect at any time — resuming means a disconnect costs
   at most `save_steps` of lost progress, not the entire run.
6. **Save the adapter + a `run_metadata.json`** recording every hyperparameter,
   the seed, the exact trainable-parameter counts, GPU info, and the final
   training loss — so every number in this documentation and in the model card
   is traceable back to a real recorded artifact, not a remembered claim.

**Measured result:** trained on a seeded **N=5,000** slice (MedMCQA's total size
is ~194K; 5K was deliberately the first validated point of a planned
data-scaling study — 20K/50K are scoped as future work, one config change away,
gated only by GPU budget beyond a free Colab session).

### 9.3 Evaluation (`src/eval/`)

**`harness.py`** builds a command-line invocation of **EleutherAI's
`lm-evaluation-harness`** — a real, standardized, widely-used evaluation tool —
rather than writing a custom scoring loop. `build_command()` assembles the exact
`lm_eval` CLI arguments (which model, which adapter, which tasks, how many
few-shot examples, whether to apply the chat template) and `run()` executes it
via `subprocess.run()`. Base and fine-tuned models are always scored under
**identical 4-bit quantization** — an important, easy-to-miss methodology
detail: if you evaluated the base model in fp16 and the fine-tuned model in
4-bit, any accuracy difference could be a quantization artifact, not a
fine-tuning effect.

**`error_analysis.py`** breaks MedMCQA accuracy down **per subject** (e.g.
"Pharmacology," "Surgery," "Anatomy"), classifying each subject as *improved*,
*neutral*, or *worsened* based on the accuracy delta between base and
fine-tuned, and renders an honest Markdown report. If the underlying lm-eval
`--log_samples` files don't exist yet, `compare()` returns `{"status": "PENDING
RUN"}` and the rendered report says exactly that — never a guessed table.

**`results.py`** renders the single headline data-scaling table (rows: Base
0-shot, Base 5-shot, QLoRA 5K/20K/50K; columns: MedMCQA, PubMedQA accuracy),
again defaulting every unmeasured cell to the literal string `"PENDING RUN"`.

**Measured results (on a matched 200-item validation subsample, both models
scored on identical items):**

| Model | MedMCQA (in-domain) | PubMedQA (out-of-domain) |
|---|---|---|
| Base 0-shot | 47.5% | 48.0% |
| QLoRA 5K | 50.0% | 64.5% |

**The honest interpretation** (this is the actual research finding, not a
polished-up positive result): the in-domain gain (+2.5 points) is within the
measurement noise for a 200-item sample — the base model was likely already
close to saturating what it can do on MedMCQA. The larger out-of-domain gain
is more likely **sharpened answer-selection behavior** (the model committing to
a clean, confident answer format) than genuinely new medical knowledge
transferring to an unrelated task. Reporting this honestly, rather than only
reporting the flattering number, is a deliberate design principle carried
through the whole project.

### 9.4 Serving and cost benchmarking (`src/serve/`)

**`loader.py`**'s `load_model_and_tokenizer()` is a small, reused-everywhere
function: load a base model (optionally 4-bit quantized), optionally attach a
LoRA adapter via `peft.PeftModel.from_pretrained()`. This exact function is
reused, unmodified, by Part A's `LocalTransformersProvider` — a concrete example
of the "don't rewrite what already works" principle in action.

**`score.py`**'s `score_options()` computes **normalized, uncalibrated option
probabilities** for a 4-choice MCQ: for each option, it appends the option text
to the prompt, computes the model's token-level log-probabilities over exactly
those appended tokens, sums and length-normalizes them (dividing by the number
of tokens, so a longer correct option isn't unfairly penalized), then softmaxes
across the 4 options. The response is explicitly documented (in both code
comments and the returned JSON's `"note"` field) as **not a calibrated
confidence** — a real, correct distinction: a language model's raw output
probability is not the same as "how likely is this to actually be true,"
and claiming otherwise would be misleading.

**`cost.py`** measures real generation throughput (tokens/second), p50/p95
latency, and peak GPU memory across serving configurations, and implements
`vllm_feasible()` — a runtime check (via `get_gpu_info()`'s compute-capability
detection) for whether the current GPU can run vLLM (a high-throughput serving
engine that needs Ampere-or-newer GPUs, compute capability ≥ 8.0). On a T4 GPU
(compute 7.5, used in free Colab), this correctly reports "not feasible" rather
than crashing or fabricating a number.

---

## 10. The Database Layer

There is **no traditional ORM** (Object-Relational Mapper) in this project's
production path — a deliberate choice, explained below. Two interchangeable
vector-store backends implement one shared abstract interface
(`src/assistant/rag/store.py`):

```python
class VectorStore(ABC):
    def add(self, embedded: list[EmbeddedChunk]) -> None: ...
    def search(self, query_vec, k, metadata_filter=None) -> list[RetrievedPassage]: ...
    def count(self) -> int: ...
```

### `LocalVectorStore` — for development, tests, and CI

Stores vectors as one big in-memory NumPy array, persisted to disk as a
compressed `.npz` file (`vectors.npz`) plus a `.jsonl` file (`chunks.jsonl`,
one JSON object per line — the chunk metadata). `search()` computes similarity
against **every** stored vector with a single matrix multiplication
(`self._vecs @ query_vec`), then sorts and applies any metadata filter — perfectly
fine at the scale of a test corpus (a handful to a few thousand chunks), and
requires **zero external services**, which is exactly why the test suite and CI
pipeline can run this whole system with no database account, no credentials,
and no network access.

### `PgVectorStore` — for production (Neon Postgres)

```sql
CREATE TABLE biomed_chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT, source TEXT, title TEXT,
    text TEXT, ordinal INT, url TEXT, year INT,
    metadata JSONB, embedding vector(384)
);
CREATE INDEX biomed_chunks_emb_idx ON biomed_chunks USING hnsw (embedding vector_cosine_ops);
```

- **One table.** `chunk_id` (e.g. `"pubmed:27752546::2"` — doc ID + chunk
  ordinal, deterministic and human-readable) is the primary key. `metadata` is a
  flexible `JSONB` column for anything not worth a dedicated column (journal
  name, PMID, etc.). `year` is pulled out into its own indexed column
  specifically because the Planner agent filters by `year_min` — a query that
  needs to be fast, not buried inside JSON.
- **No relationships/foreign keys** — this is intentionally a single flat table,
  because the access pattern is "similarity search with optional filters," not
  relational joins across entities. There is no "users" table, no "sessions"
  table — see [§11](#11-authentication--what-exists-and-what-doesnt).
- **HNSW index** for approximate nearest-neighbor search, chosen over the
  alternative `IVFFlat` index type because HNSW gives better query-time recall
  at this dataset's scale (~18.5K chunks), at the cost of slower index
  *building* — an acceptable trade since the index is rebuilt rarely, not on
  every request.
- **CRUD, in practice:** *Create* — `PgVectorStore.add()` sends one batched
  (`executemany`), parameterized `INSERT ... ON CONFLICT (chunk_id) DO UPDATE` —
  an "upsert": insert if new, otherwise **overwrite every column**, so re-running
  the index-build script is safe. (An earlier version updated only `embedding`
  on conflict: re-chunking a document then left the *old* text paired with the
  *new* vector — retrieval would match on one passage and show another.)
  *Read* — `search()` runs one parameterized
  `SELECT ..., embedding ORDER BY embedding <=> %s LIMIT %s` query (`<=>` is
  pgvector's cosine-distance operator), with optional `WHERE source = ANY(%s)`
  and `WHERE year >= %s` clauses appended based on the metadata filter. The
  vector is returned too, so verification can reuse it (§8.4). There is
  currently no *Update* or *Delete* path exposed beyond the upsert-on-add
  behavior — the system is built around rebuilding the index from source
  documents, not fine-grained row editing.
- **Why raw SQL instead of an ORM (SQLAlchemy, Prisma-equivalent, etc.)?** The
  access pattern here — vector similarity search with a handful of optional
  filters — doesn't benefit from an ORM's main value proposition (managing
  complex relational joins and migrations across many tables). A hand-written,
  parameterized query is simpler to read, has zero extra dependency weight (an
  ORM would add real size to the deliberately lean `requirements-api.txt`
  free-tier image), and every *value* in every statement goes through `%s`
  placeholders, so user input never reaches the SQL text. (The table name *is*
  f-string-interpolated — safe only because it's a constant set in code, never
  derived from a request.)
- **Connections.** The store keeps **one** psycopg connection and reuses it,
  instead of opening a new TLS connection to Neon for every search. All access
  goes through `_run(fn)`, which holds a lock (a psycopg connection must not be
  used by two threads at once), runs `fn(cursor)`, commits, and on
  `OperationalError` — Neon closes idle connections — reconnects **once** and
  retries. Other errors roll back and re-raise. `connect_timeout=10` stops a
  dead database from hanging a request. The trade-off: the lock serializes
  database calls, which is fine at demo traffic; real concurrency would want a
  pool (`psycopg_pool`).

### How data actually flows in

```text
scripts/rag_index.py
  → fetch_pubmed() / load_guidelines()   (ingest.py, HTTP + local files)
  → chunk_document()                      (chunk.py)
  → embedder.embed_documents()             (embed.py, batch)
  → store.add([EmbeddedChunk, ...])        (store.py — INSERT/upsert)
     or store.rebuild([...])                 (store.py — staging table + atomic swap)
```

This is a **one-way, offline, batch pipeline**, run deliberately by a human
(`python scripts/rag_index.py --config configs/corpus.yaml`), not triggered by
user traffic — there is no "user uploads a document" feature in this system.

---

## 11. Authentication — What Exists and What Doesn't

**Direct answer: there is no authentication, no user accounts, no login, no
sessions, no JWTs, no OAuth, and no cookies anywhere in this codebase.**

This is a correct, deliberate choice for what this system is: a public research
demo with no user-specific data, no paid tiers, and no private information to
protect behind a login. Every endpoint in `src/assistant/api/app.py` is open —
`GET /health`, `POST /query`, and `GET /benchmark` all respond to any caller.

**What *does* exist, adjacent to security:**
- **CORS** (`CORSMiddleware`) — explained in [§7](#7-core-concepts-beginner--advanced) —
  controls which *browsers* are allowed to call the API from a web page, but this
  is not authentication (it doesn't identify who is calling; a `curl` request
  from any machine bypasses CORS entirely, since CORS is enforced by browsers,
  not servers).
- **Input validation** — `QueryRequest.question` has `min_length=3, max_length=1000`
  (Pydantic `Field` constraints), which prevents empty or absurdly long request
  bodies, but is not access control.
- **Secrets management** — real credentials (the Hugging Face token, the Neon
  database connection string) are read from environment variables
  (`BIOMED_HF_TOKEN`, `BIOMED_DATABASE_URL`) and are configured in Render's
  dashboard as secrets, never committed to the repository (`.env` is gitignored;
  see `.env.example` for the *template*, not real values). This protects the
  *backend infrastructure's* credentials — it has nothing to do with
  authenticating end users.

**What a production version of this system, if it needed real user accounts,
would add** (a fair interview question, honestly answered rather than dodged):
an auth layer (e.g. a hosted provider like Auth0/Clerk, or hand-rolled JWT
issuance) in front of the FastAPI app, a `users` table, per-*user* rate limiting
(today the limit is per client IP — 20 `/query` calls a minute — which caps
token spend but can't tell users apart, especially since all Gradio Space
traffic arrives from one IP; see [§8.7](#87-the-api-fastapi)), and role-based checks if different users needed
different capabilities. None of this exists today because none of it was needed
for a free research demo with no persistent user data.

---

## 12. State Management — What Exists and What Doesn't

There is **no client-side state management library** here (no Redux, no
Zustand, no React Context, no TanStack Query) — because there is no client-side
JavaScript framework at all. Gradio's Python process holds whatever UI state
exists (the current contents of the textbox, the currently-selected dropdown
value) internally, managed entirely by the `Blocks`/event-binding system
described in [§8.9](#89-the-frontend-gradio) — this is Gradio's job, not
something this codebase implements.

**What this codebase *does* manage explicitly is server-side state on the
FastAPI backend** — and this is the more interesting engineering answer:

```python
_STATE: dict = {}   # module-level, persists for the life of the server process

@asynccontextmanager
async def lifespan(app: FastAPI):
    _STATE["cfg"] = get_settings()
    _STATE["service"] = None          # NOT built yet — lazy
    if _EXPLORER_PATH.exists():
        _STATE["benchmark"] = json.loads(_EXPLORER_PATH.read_text())
    yield
    _STATE.clear()
```

This is a **lazy singleton** pattern: the expensive `AssistantService` object
(which, once built, holds a loaded RAG pipeline and an LLM provider) is created
**at most once**, the first time `POST /query` is called, and then reused for
every subsequent request for the life of the server process — avoiding the cost
of rebuilding the embedding model / reconnecting to the database on every single
HTTP request. This is conceptually the same problem React Query / TanStack Query
solves on the frontend (cache an expensive resource, don't refetch/rebuild it
unnecessarily) — just solved on the server side here, because the expensive
resource (a language model connection) lives on the server, not in the browser.

**Why a plain module-level dict instead of, say, a proper dependency-injection
framework?** At this scale (one process, one shared service, no per-request or
per-user variation in what's cached), a dict is the simplest tool that is
obviously correct — introducing a DI framework would add a dependency and
indirection to solve a problem that doesn't exist yet. If this system grew to
need multiple concurrently-loaded models or per-tenant caching, that would be
the point to revisit this decision — a good example of not over-engineering for
a hypothetical future.

---

## 13. Every Dependency, Explained

### Part A — the production stack

| Library | Why it's here | Used in | Simpler alternative | Trade-off |
|---|---|---|---|---|
| **FastAPI** | Modern async-capable Python web framework with automatic request validation (via Pydantic) and free OpenAPI docs | `api/app.py` | Flask | Flask is simpler but has no built-in request validation or async support |
| **Pydantic / pydantic-settings** | Typed data models + environment-variable-driven config | `schema.py`, `config.py` | plain dicts + `os.environ` | Plain dicts are zero-dependency but give up validation and self-documentation |
| **Gradio** | Turns a Python function into a full deployable web UI with almost no frontend code | `deploy/space/app.py` | Streamlit, or a hand-built React app | Streamlit is similar; React would need a separate build/deploy pipeline for far more control this project doesn't need |
| **LangGraph** | Compiles a set of Python functions into an explicit, inspectable state graph | `agents/graph.py` | A plain Python function calling 4 functions in a row (the actual fallback!) | The plain-function version is simpler and is literally used as the fallback; LangGraph adds graph visualization/composability for when workflows get more complex than a straight line |
| **sentence-transformers** | Runs the `bge-small` embedding model and the cross-encoder reranker locally via PyTorch | `rag/embed.py`, `rag/rerank.py` | Call a hosted embeddings API for everything | Hosted APIs avoid the PyTorch dependency but add network latency + a paid/rate-limited dependency for every single embedding |
| **fastembed** | Runs the *same* embedding model as ONNX, with zero PyTorch dependency | `rag/embed.py` (`FastEmbedEmbedder`) | sentence-transformers everywhere | sentence-transformers is more full-featured but far too heavy for a 512MB free server |
| **huggingface_hub** | Client library for Hugging Face's Inference API and Hub uploads | `serving/providers.py`, `rag/embed.py`, `scripts/push_to_hub.py` | Hand-rolled `httpx` calls to HF's REST API | Hand-rolled calls work but reimplement retries/auth HF's official client already handles |
| **psycopg + pgvector (Python)** | Postgres driver + vector-type support | `rag/store.py` | `asyncpg`, or an ORM like SQLAlchemy | `asyncpg` is faster for async workloads (not needed here); an ORM adds weight this single-table system doesn't need |
| **numpy** | Vector math (cosine similarity via dot product) for the local store | `rag/store.py`, `rag/embed.py` | Pure Python loops | Would be orders of magnitude slower for vector operations |
| **httpx** | Modern HTTP client (used for NCBI E-utilities calls and the Gradio→FastAPI call) | `rag/ingest.py`, `deploy/space/app.py` | `requests` | `requests` has no native async support; not needed here either, but httpx is the more modern default |
| **rouge-score, bert-score, rank-bm25** | Optional generation-quality metrics | `eval/generation_metrics.py` | Skip these metrics entirely | Each is lazily imported and gracefully skipped if absent — a real engineering choice, not an accident |
| **uvicorn** | The actual ASGI server process that runs the FastAPI app | Dockerfiles, `render.yaml` | Gunicorn+Uvicorn workers | Fine for this single-process, low-concurrency deployment; a busier production system would add Gunicorn as a process manager |

### Part B — the research/training stack

| Library | Why it's here | Trade-off |
|---|---|---|
| **PyTorch** | The deep-learning framework everything else builds on | Required — no lighter alternative for real model training |
| **transformers** | Loads and runs the Qwen2.5 model + tokenizer | The standard choice; alternatives (raw model-loading code) would reimplement a huge surface area |
| **peft** | Implements LoRA adapters on top of a `transformers` model | Purpose-built for exactly this; hand-rolling LoRA is possible but error-prone |
| **trl** | Provides `SFTTrainer`, a ready-made supervised fine-tuning loop | Saves reimplementing the training loop, loss computation, and logging |
| **bitsandbytes** | 4-bit quantization (QLoRA's "Q") | The standard tool for this; version must match the exact CUDA/Triton the runtime ships (a real bug fixed during this project — see [§17](#17-design-decisions-and-trade-offs)) |
| **lm-eval (EleutherAI harness)** | Standardized, peer-reviewed benchmark scoring | Chosen explicitly *instead of* a hand-rolled scoring loop, for scientific credibility |
| **wandb** | Optional experiment tracking | Degrades gracefully to local JSONL logging if no API key is set — training must never be blocked by a missing tracking key |

---

## 14. Every Configuration File, Explained

- **`requirements.txt`** — Part B's dependencies, pinned to exact versions
  (e.g. `torch==2.5.1`) for reproducibility; includes a platform marker
  (`bitsandbytes>=0.45.0 ; platform_system == "Linux"`) so macOS development
  machines don't try (and fail) to install a Linux/CUDA-only package.
- **`requirements-assistant.txt`** — Part A's full dependency set, for
  offline index-building and local development, including the heavier
  `sentence-transformers`.
- **`requirements-api.txt`** — Part A's *lean* dependency set, deliberately
  excluding PyTorch, for the actual deployed free-tier server.
- **`Dockerfile`** — builds a CUDA-enabled image (`nvidia/cuda:12.1.1-...`) for
  Part B's GPU training/evaluation workloads.
- **`deploy/Dockerfile.api`** — builds a plain `python:3.12-slim` image (no
  CUDA, no PyTorch) for Part A's live backend; reads `$PORT` at runtime because
  Render assigns the port dynamically (`CMD ["sh", "-c", "uvicorn ... --port
  ${PORT:-8000}"]`).
- **`deploy/render.yaml`** — a Render "blueprint": declares one web service
  (`biomed-assistant-api`), which Dockerfile builds it, and its environment
  variables — with `sync: false` marking the two real secrets
  (`BIOMED_DATABASE_URL`, `BIOMED_HF_TOKEN`) so they must be entered manually in
  Render's dashboard rather than committed to source control. It also pins
  `BIOMED_HF_PROVIDERS=featherless-ai,auto` and
  `BIOMED_RATE_LIMIT_PER_MINUTE=20` (both match the code defaults, so they're
  documentation as much as configuration).
- **`deploy/neon_schema.sql`** — documents the exact table/index the app
  creates automatically on first connection (`PgVectorStore._ensure_schema()`);
  kept here for a human to read without running Python.
- **`configs/base.yaml` + `configs/qlora_*.yaml`** — Part B's training
  configuration, using a small custom **YAML inheritance** system
  (`src/utils/config.py`'s `load_config()`): a child file declares
  `inherits: base.yaml` and only needs to override the handful of keys that
  actually change (`data.train_size`, `output.output_dir`), via a recursive
  dict merge (`_deep_merge()`). This avoids duplicating 40 lines of shared
  hyperparameters across 5 nearly-identical files.
- **`configs/corpus.yaml`** — the reproducible list of PubMed searches (18
  clinical topics, 300 Best-Match results each) plus a shared `query_filter`
  appended to every query: has an abstract, English, review or guideline
  publication type, published 2010–2026. Re-running
  `scripts/rag_index.py --config configs/corpus.yaml` rebuilds the corpus; it is
  reproducible in method, though PubMed's ranking drifts as papers are added.
- **`lm_eval_tasks/*.yaml` + `utils.py`** — custom task definitions telling
  `lm-evaluation-harness` how to format a MedMCQA/PubMedQA question and how to
  score it (`output_type: multiple_choice` — meaning the harness scores the
  model's log-likelihood of each option text and picks the highest, rather than
  free-form generation). `fewshot_split: train` on the MedMCQA task is a subtle,
  important correctness detail: few-shot example questions are drawn from the
  *training* split, never from the *scored* validation split, avoiding
  contaminating the benchmark with its own answer key.
- **`.github/workflows/ci.yml`** — GitHub Actions: on every push or pull
  request, installs a lean, CPU-only dependency set (no PyTorch), runs Part B's
  structural smoke test (`scripts/smoke_test.py` — checks imports, config
  loading, and command-building without ever touching a GPU or downloading a
  model), then runs Part A's 29-test `pytest` suite. This CI setup is a real
  engineering signal: it proves the whole system's *wiring* is correct on every
  single commit, with zero cost and zero GPU dependency.
- **`.env.example`** — a template showing which environment variables *can* be
  set (all optional for Part B: W&B key, HF token), explicitly never containing
  real values, and `.env` itself is git-ignored.

---

## 15. The API Reference

Base URL (production): `https://biomed-assistant-api.onrender.com`

### `GET /health`

**Purpose:** liveness probe + "what is this server actually configured to do
right now" transparency check.

**Request:** no parameters.

**Response `200`:**
```json
{
  "status": "ok",
  "served_config": "base_rag",
  "served_config_display": "Base + RAG",
  "vector_backend": "pgvector",
  "inference_provider": "hf_inference",
  "model_loaded": true,
  "benchmark_available": false,
  "last_query": {"ok": true, "failed_stage": null, "at": 1791157260}
}
```
`model_loaded: false` means the lazy `AssistantService` singleton hasn't been
built yet (no `/query` has been received since the server started) — the first
`/query` call will trigger the one-time build.

`last_query` is the outcome of the most recent `/query` (`null` until one
arrives). `failed_stage` is `"startup"`, `"model"`, or `"internal"` when it
failed. This exists because `status: "ok"` only proves the process is up — it
stayed `"ok"` through an outage in which every query failed.

### `POST /query`

**Purpose:** the live, grounded question-answering endpoint.

**Request body:**
```json
{ "question": "What is dulaglutide and how is it used in type 2 diabetes?" }
```
`question` is validated by Pydantic: 3–1000 characters (`QueryRequest` in
`api/app.py`). A request with `question` shorter than 3 characters is rejected
automatically by FastAPI with a `422 Unprocessable Entity` *before* any
application code runs — this is Pydantic validation happening at the framework
boundary.

**Processing (full trace in [§16.1](#161-live-query-a-user-asks-a-question)):**
lazily build the `AssistantService` if needed → run the 4-agent LangGraph
workflow → serialize the resulting `GroundedAnswer`.

**Response `200`** (abbreviated; real field names, real shape):
```json
{
  "query": "What is dulaglutide and how is it used in type 2 diabetes?",
  "answer": "Dulaglutide is a glucagon-like peptide-1 receptor agonist (GLP-1RA) [2]. ...",
  "config": "base_rag",
  "citations": [
    {"marker": "[1]", "doc_id": "pubmed:...", "source": "pubmed",
     "title": "Early management of sepsis.", "url": "https://pubmed.ncbi.nlm.nih.gov/...",
     "quote": "...", "supported": true, "support_score": 0.853}
  ],
  "passages": [ { "chunk": {"...": "..."}, "score": 0.71, "rank": 1 } ],
  "claims": [
    {"claim": "Dulaglutide is a GLP-1RA [2].", "supported": true,
     "support_score": 0.84, "passage_index": 1}
  ],
  "all_claims_supported": true,
  "latency_ms": {"retrieve_ms": 794.48, "generate_ms": 1887.7, "verify_ms": "..."},
  "token_usage": {"prompt_tokens": 634, "completion_tokens": 39, "total_tokens": 673}
}
```
The `latency_ms` and `token_usage` values above are from a real live query
(warm instance, Oct 2026). `verify_ms` was added after that measurement.
`passages[].embedding` exists internally but is excluded from the response.

**Possible errors:**
- `422` — `question` fails Pydantic validation (too short/long/missing).
- `429` — per-client rate limit hit; a `Retry-After` header says how many
  seconds to wait.
- `503` — the service couldn't be built (database or embedder unreachable).
- `502` — every LLM provider failed (`"hosted language model is unavailable"`)
  or something else failed mid-pipeline (`"Query failed"`).

Every error `detail` includes the request ID (also returned in the
`X-Request-ID` header) and never the underlying exception text, which can
contain secrets such as a database connection string. The full error is in the
server logs under that request ID — see [§8.7](#87-the-api-fastapi).

### `GET /benchmark`

**Purpose:** serve the precomputed 4-way comparison data for the Gradio
"Benchmark Explorer" tab.

**Response `200`:** the contents of `results/benchmark_explorer.json` (per-config
retrieval/generation/systems aggregates plus every individual question's full
record — answer, passages, citations, claims).

**Possible errors:** `404` if `results/benchmark_explorer.json` was never
generated (i.e. `scripts/rag_benchmark.py` hasn't been run yet against the real
deployed corpus) — the error message says exactly that, rather than returning an
empty or fabricated payload.

---

## 16. Execution Traces for Every Major Feature

### 16.1 Live query — a user asks a question

```text
Browser: types question, clicks "Ask"
   │
   ▼
Gradio (deploy/space/app.py): btn.click → shows LOADING state immediately
   │
   ▼
Gradio: ask(question) → httpx.post(BACKEND_URL + "/query", json={"question": ...})
   │
   ▼
FastAPI (api/app.py): request_context middleware stamps a request ID, starts a timer
   │
   ▼
FastAPI: QueryRequest Pydantic model validates length (3-1000 chars)
   │
   ▼
FastAPI: query() handler — RateLimiter.check(client) → over limit? 429 + Retry-After
   │
   ▼
FastAPI: _get_service() — is _STATE["service"] None?
   │            │
   │  yes ──────┘──► take _BUILD_LOCK, re-check, then _build_service():
   │                  RAGPipeline + LLM provider (BIOMED_INFERENCE_PROVIDER picks
   │                  HFInferenceProvider or LocalTransformersProvider)
   │                  → AssistantService           (build fails → 503)
   ▼
AssistantService.answer(question)
   │
   ▼
LangGraph (or sequential fallback) runs, in strict order:
   │
   ├─► planner.plan()      — no I/O; builds a metadata_filter from keywords
   │
   ├─► retrieval.retrieve() — RAGPipeline.retrieve_context(query, metadata_filter)
   │        │
   │        ├─► Embedder.embed_query(query)         (FastEmbedEmbedder: local ONNX)
   │        ├─► VectorStore.search(vec, k=20, filter) (PgVectorStore: SQL on a reused
   │        │                                          connection; returns stored vectors)
   │        ├─► dedupe_passages(...)                  (drop republished duplicate texts)
   │        ├─► Reranker.rerank(...)                  (skipped — off in free tier → top 5)
   │        └─► build_citations(top passages)
   │
   ├─► answer.answer()      — RAGPipeline.format_context(passages)
   │        │                  build_answer_messages(query, context)
   │        ├─► LLMProvider.generate(messages)         (HFInferenceProvider: tries
   │        │                                          featherless-ai, then auto;
   │        │                                          all fail → UpstreamError → 502)
   │        └─► normalize_markers(text)                ([1-3] → [1][2][3])
   │
   └─► verify.verify()      — for each answer sentence (abstentions skipped):
            embed the sentence, cosine vs the passages' STORED vectors, threshold 0.6
            → per-claim supported/unsupported + overall all_claims_supported
            + verify_ms
   │
   ▼
AssistantService returns a GroundedAnswer (typed Pydantic object)
   │
   ▼
FastAPI: record last_query; ans.model_dump() → JSON (vectors excluded);
         middleware logs the request + adds X-Request-ID / timing headers
   │
   ▼
Gradio: r.json() → _render_answer(d) + _render_evidence(d) → HTML strings
   │
   ▼
Browser: re-renders the answer panel + evidence panel with the real result
```

### 16.2 Building the search index (offline, run by a human)

```text
Human runs: python scripts/rag_index.py --config configs/corpus.yaml [--rebuild | --dry-run]
   │
   ▼
get_settings() loads config (BIOMED_VECTOR_BACKEND, BIOMED_DATABASE_URL, ...)
   │
   ▼
RAGPipeline() constructed — chooses embedder + store backend from config
   │
   ▼
load_guidelines(corpus_dir)  — reads any local NIH/WHO/CDC .txt files
   │
   ▼
for each of the 18 queries in corpus.yaml:
     fetch_pubmed("(" + query + ")" + query_filter, retmax=300)
        → esearch (Best Match PMIDs) → efetch POST (abstracts) → Documents
   │
   ▼
de-duplicate: by doc_id (topics overlap), then by abstract text
              (PubMed republishes some abstracts under several PMIDs)
   │
   ├── default (upsert): build_index() — chunk → embed → store.add() per document
   │
   └── --rebuild / --dry-run: embed_corpus() — chunk + embed EVERYTHING in memory,
          writing nothing (--dry-run stops here and prints counts)
          │
          ▼
       store.rebuild(embedded):
          CREATE TABLE biomed_chunks_new                 (live table untouched)
          INSERT … in batches of 1,000, one commit each
          ── one transaction ──────────────────────────────────────────────
          CREATE INDEX (hnsw, source, year) ON biomed_chunks_new
          DROP TABLE IF EXISTS biomed_chunks_prev
          ALTER TABLE biomed_chunks     RENAME TO biomed_chunks_prev
          ALTER TABLE biomed_chunks_new RENAME TO biomed_chunks   (+ index/pkey renames)
          COMMIT
```

**Why a staging table instead of `TRUNCATE` + reload?** Embedding 18,528
chunks takes ~12 minutes on a laptop CPU and uploading takes more; emptying the
live table first would have left the public demo answering from an empty or
half-loaded index for that whole time. With the swap, live queries read the old
table until the final `COMMIT`, and the rename holds its lock for milliseconds.
The HNSW index is built once after loading, which is much faster than
maintaining it across thousands of inserts.

**Rollback.** The previous index survives as `biomed_chunks_prev` until the next
rebuild. Reverting is one transaction: rename `biomed_chunks` → `_new` and
`biomed_chunks_prev` → `biomed_chunks` (plus their indexes). Neon's 6-hour
history window is a second, coarser safety net.

**Measured result (Oct 2026, this config):** 5,070 unique PMIDs → 41 duplicate
abstracts dropped → **5,029 documents → 18,528 chunks**. The previous,
exact-phrase config produced 733 documents → 3,410 chunks.

### 16.3 Running the 4-way evaluation benchmark

```text
Human runs: python scripts/rag_benchmark.py --adapter <adapter-repo-or-dir>
   │
   ▼
Build ONE shared RAGPipeline (same index for all 4 configs — fair comparison)
   │
   ▼
Build 2 LLM providers: base_provider (no adapter), ft_provider (with adapter)
   │
   ▼
Build 4 AssistantService instances from CONFIGS = {base, ft, base_rag, ft_rag}
   │
   ▼
For each config, for each curated EvalQuestion:
     service.answer(question.question)  → GroundedAnswer
     → compute retrieval metrics (Recall@k, MRR) IF this config uses RAG
     → compute generation metrics (citation coverage, groundedness, ROUGE-L)
     → compute systems metrics (latency, tokens)
   │
   ▼
Aggregate per config → render_markdown() (3 comparison tables)
                      → write_explorer_data() (benchmark_explorer.json)
   │
   ▼
results/rag_benchmark.md + results/benchmark_explorer.json written to disk
(the latter is what deploy/space/app.py's "Benchmark Explorer" tab reads)
```

### 16.4 Fine-tuning the model (Part B)

```text
Human runs (on a GPU, e.g. Colab): python scripts/train.py --config configs/qlora_5k.yaml
   │
   ▼
load_config() — resolves "inherits: base.yaml", merges qlora_5k.yaml's overrides
   │
   ▼
set_seed(42) — Python, NumPy, and Torch RNGs all seeded identically
   │
   ▼
load_train_val(train_size=5000, val_size=1000, seed=42)
     → filter valid rows → shuffle(seed=42) → disjoint val slice → train slice
   │
   ▼
Load Qwen2.5-7B-Instruct in 4-bit (BitsAndBytesConfig: nf4, double-quant, fp16 compute)
   │
   ▼
Attach a LoRA adapter (r=16, alpha=32) via peft.get_peft_model()
   │
   ▼
_latest_checkpoint(output_dir) — resume if a previous run left checkpoints
   │
   ▼
SFTTrainer.train() — gradient checkpointing, fp16, paged AdamW 8-bit
     (every `logging_steps`, loss is logged to W&B if configured, else locally)
     (every `save_steps`, a checkpoint is saved — survives a Colab disconnect)
   │
   ▼
trainer.save_model() + tokenizer.save_pretrained() → the adapter files
   │
   ▼
run_metadata.json written: seed, N, trainable params, hyperparameters, final loss
```

---

## 17. Design Decisions and Trade-offs

This section is the "why," not just the "what" — the questions a technical
interviewer is most likely to ask, answered with the actual reasoning.

### Why pgvector on Postgres instead of a dedicated vector database (Pinecone, Weaviate, etc.)?

At this corpus size (18,528 chunks, 103 MB in Neon), a dedicated vector database's main
advantages — massive horizontal scale, specialized indexing at millions of
vectors — don't apply. Using pgvector means **one datastore** to operate
(no separate service, no separate bill, no separate credentials), plain SQL for
metadata filtering (`WHERE source = ANY(%s)`) instead of a second query
language, and it's free-tier-available on Neon. If the corpus grew to millions
of chunks, this would be the first assumption worth revisiting.

### Why QLoRA instead of full fine-tuning?

Full fine-tuning a 7B model means updating and storing all ~7 billion
parameters, requiring roughly 4× that in GPU memory during training (for
weights, gradients, and optimizer state) — completely out of reach of a free
16GB Colab GPU. QLoRA trains **0.92%** of the parameters while the rest stay
frozen and 4-bit quantized, fitting comfortably in that budget, at the honest
cost of a smaller, lower-rank update to the model's behavior than full
fine-tuning could achieve.

### Why evaluate under matched quantization?

If the base model were scored in full fp16 precision and the fine-tuned model
scored in 4-bit, any accuracy difference could be caused by the quantization
itself, not by the fine-tuning. Both are scored identically 4-bit in this
project's harness — a subtle methodology point that materially affects whether
the reported delta means anything.

### Why does the live demo serve `Base + RAG` and not `Fine-tuned + RAG`?

This is the single most important trade-off in the whole system, and it is
handled with total transparency rather than hidden. **Hugging Face's free,
serverless Inference API can only serve published base models — it cannot load
a custom LoRA adapter.** Running the fine-tuned model live would require a real
GPU server, which costs money continuously (not free-tier). Rather than either
(a) paying for a GPU server, or (b) silently serving the base model while
*claiming* it's the fine-tuned one, the system:
- Has one function, `served_config(cfg)`, as the single source of truth for
  which config is actually active.
- Exposes that truth on `/health` and on every `/query` response's `config`
  field.
- Displays it verbatim in the UI (`backend_status()` in the Gradio app).
- Documents, in the UI itself, exactly how to upgrade to true `Fine-tuned + RAG`
  (point `BIOMED_INFERENCE_PROVIDER` at a GPU-backed server) — a **one
  environment-variable change, zero code change**.
The true 4-way comparison — including the real fine-tuned model — lives in the
**offline, precomputed Benchmark Explorer**, which needed a GPU only once, not
continuously.

### Why semantic grounding instead of lexical (word-overlap) grounding?

The system originally shipped with only the lexical `verify_claims()` method
(raw token-overlap/Jaccard similarity). During real testing, this produced a
**false negative**: a genuinely correct, well-evidenced answer about early
sepsis management was marked "unsupported" purely because it **paraphrased**
the source passage instead of repeating its exact wording — lexical overlap
cannot detect that "source control" and "identifying and addressing the
infection source" mean the same thing. The fix
(`verify_claims_semantic()` in `rag/citations.py`) compares **sentence
embeddings** instead of raw words, so a correct paraphrase still scores high
similarity. The same sepsis answer, re-verified after the fix, correctly scored
`all_claims_supported: True`. The lexical method is kept as a config-selectable
fallback (`grounding_method: "lexical"`), not deleted, since it has zero
external dependency and is useful if embeddings are ever unavailable.

### Why is the reranker disabled in production but enabled by default in config?

The cross-encoder reranker needs PyTorch, and the free-tier server has no
PyTorch installed at all (a torch-free image was a hard requirement for fitting
512MB of RAM). `use_reranker: true` remains the *pipeline's* default because
reranking measurably improves retrieval precision when you *can* afford it (a
GPU-backed or larger deployment); the free-tier `render.yaml` explicitly
overrides it to `false`. This is a real, acknowledged precision trade-off, not
an oversight — documented in the code, the README, and this document.

### Why keep both `space/app.py` and `deploy/space/app.py`?

Only one answer here is fully honest: `space/app.py` is a leftover from an
earlier iteration of the project and is **not** part of the currently deployed
system. It's flagged explicitly in [§8.9](#89-the-frontend-gradio) rather than
silently left for a future reader to be confused by.

### Why LangGraph with a sequential fallback, instead of just always using LangGraph?

If `langgraph` isn't installed (e.g. a minimal CI environment, or a future
deployment target that trims dependencies further), the system should still
work identically — because the actual *logic* lives in 4 independent Python
functions, not inside LangGraph itself. `AssistantService.__init__` tries to
compile a LangGraph graph and falls back to a plain sequential loop over the
same functions if that fails for any reason. This is why the test suite
(`prefer_langgraph=False` in every test) never needs LangGraph installed at all
to fully exercise the agent logic.

---

## 18. Testing Strategy

**29 tests total, 100% CPU-only, zero network calls, zero model downloads** —
verified to run in well under a second.

- **`tests/_fakes.py`** — the shared testing infrastructure. `FakeEmbedder`
  produces deterministic vectors via `zlib.crc32` word-hashing (specifically
  **not** Python's built-in `hash()`, which is randomized per process by design
  for security reasons — using it would have made retrieval order
  non-deterministic and caused real, intermittent test flakiness, which is
  exactly what happened during development and was fixed by switching to
  `crc32`). `FakeProvider` echoes back a sentence from the retrieved context
  with a citation marker attached, so grounding tests can assert a supported
  claim without needing a real LLM.
- **`tests/test_rag.py`** (6 tests) — sentence splitting, chunk sizing/overlap,
  local vector store search + metadata filtering + persistence-across-instances,
  full pipeline retrieval, and semantic grounding correctness.
- **`tests/test_agents.py`** (4 tests) — the planner's keyword-based metadata
  filtering, a full `ft_rag` answer coming back grounded and cited, the `base`
  config correctly skipping retrieval entirely, and the semantic grounding
  function directly.
- **`tests/test_eval.py`** (3 tests) — retrieval metrics (Recall@k, MRR) against
  hand-constructed passage lists with known correct answers, generation metrics
  (citation coverage, groundedness) against hand-written strings, and a full
  4-way benchmark run end-to-end against the offline sample corpus.
- **`tests/test_api.py`** (10 tests) — the production-hardening paths, each
  tied to a real failure: provider failover (first provider raises, second
  answers) and `UpstreamError` when all fail; the sliding-window rate limiter
  (per-client, window slides, `0` disables) and the 429 + `Retry-After`
  response; error bodies that contain the request ID but **not** a planted
  `postgresql://user:pw@…` string; a model outage that does *not* discard the
  service; 8 threads cold-starting at once with the builder running exactly
  once; and the pgvector store reusing one connection, reconnecting once after
  a dropped connection, and upserting every column; and `rebuild()` loading a
  staging table in batches, then building indexes and swapping names in one
  commit. The store tests inject a
  fake `psycopg` module, so it needs no database.
- **`tests/test_citations.py`** (6 tests) — answer-quality fixes from live
  output: `[1-3]` / `[1, 3]` normalization (clamped to the source count, plain
  "1-3 mg" untouched); abstention sentences excluded from claims; duplicate
  abstracts collapsed with ranks renumbered; semantic verification embedding
  only the claim when passages carry stored vectors; and vectors never
  appearing in the serialized API payload.
- **CI (`ci.yml`)** installs a minimal, PyTorch-free dependency set (plus
  pinned `fastapi` + `httpx` for the API tests) and runs
  both Part B's structural smoke test and Part A's full test suite on every
  push/PR — meaning a broken import, a broken config file, or a broken agent
  wiring is caught automatically, without needing a GPU or any paid service, on
  every single commit.

**Why fakes instead of mocking libraries (e.g. `unittest.mock`)?** A fake object
that implements the *real* interface (`embed_documents`, `embed_query`,
`generate`) is more honest than a mock that just returns hardcoded values for
whatever method happens to be called — the fakes here actually compute
something real (a deterministic hash-based vector, a real string manipulation),
so the tests exercise real data-flow logic, not just "was this method called."

---

## 19. Glossary

| Term | Simple definition | Where it appears here |
|---|---|---|
| **API** | A defined way for programs to talk to each other over the network | `src/assistant/api/app.py` |
| **Async** | Code that can do other work while waiting on something slow | FastAPI middleware (see §7) |
| **Chunk** | A small, searchable slice of a larger document | `rag/chunk.py` |
| **CI (Continuous Integration)** | Automatically running tests on every code change | `.github/workflows/ci.yml` |
| **CORS** | Browser rule controlling cross-website API calls | `api/app.py`'s `CORSMiddleware` |
| **Embedding** | Text turned into a list of numbers representing its meaning | `rag/embed.py` |
| **Fine-tuning** | Further training an already-trained model on new, specific data | `src/train/sft.py` |
| **Grounding** | Verifying a claim is actually backed by real evidence | `rag/citations.py` |
| **HNSW** | A fast approximate search algorithm for vector databases | `deploy/neon_schema.sql` |
| **LangGraph** | A library for building explicit, structured multi-step AI workflows | `agents/graph.py` |
| **LLM** | Large Language Model — the AI that writes the answer text | `serving/providers.py` |
| **LoRA** | A cheap way to fine-tune a model by training small extra matrices | `src/train/sft.py` |
| **Middleware** | Code that runs automatically around every web request | `api/app.py` |
| **Pydantic** | A Python library for typed, validated data models | `schema.py`, `config.py` |
| **QLoRA** | LoRA combined with 4-bit model compression | `src/train/sft.py` |
| **Quantization** | Storing model numbers with fewer bits to save memory | `configs/base.yaml`'s `quant:` |
| **RAG** | Search real documents first, then have the AI answer using them | `rag/pipeline.py` |
| **Reranking** | Re-scoring initial search results with a slower, more accurate method | `rag/rerank.py` |
| **Singleton** | An object created exactly once and reused | `api/app.py`'s `_STATE["service"]` |
| **Vector store / vector database** | A database specialized in "find similar items" search | `rag/store.py` |

---

## 20. Complete End-to-End Walkthrough

*Narrated exactly as it happens, from a cold start, using every major feature.*

1. **A visitor opens `huggingface.co/spaces/Udit013/biomed-assistant`.** Hugging
   Face serves the Gradio app (`deploy/space/app.py`) — a lightweight Python
   process, no GPU, no heavy dependencies (just `gradio` + `httpx`). The page
   loads instantly; `demo.load(backend_status, ...)` immediately fires a
   `GET /health` request to the Render backend to populate the live status badge.

2. **The backend has been idle** (Render's free tier sleeps unused services), so
   this first request is slow — the platform is booting a fresh container. The
   badge shows "backend asleep — first query will wake it (~30s)" instead of
   silently hanging.

3. **The visitor reads the header, the disclaimer, and clicks an example
   question:** *"How should sepsis be managed early?"* This fills the textbox
   via `gr.Examples`.

4. **The visitor clicks "Ask."** The button's `.click()` handler fires two
   chained events: first, an instant UI update to the loading state (spinner +
   explanatory copy); then, `ask()` sends the real request.

5. **The FastAPI backend receives `POST /query`.** The `request_context`
   middleware stamps a request ID and starts a timer. Pydantic validates the
   question length. Since this is the very first query since the container
   started, `_STATE["service"]` is `None`, so `_build_service()` runs: it reads
   `BIOMED_INFERENCE_PROVIDER=hf_inference` from the environment and builds an
   `HFInferenceProvider` (not a `LocalTransformersProvider` — no GPU here) and a
   fresh `RAGPipeline` (which connects to Neon Postgres via `PgVectorStore`, and
   builds a `FastEmbedEmbedder` because `BIOMED_EMBEDDING_PROVIDER=fastembed`).

6. **`AssistantService.answer("How should sepsis be managed early?")` runs the
   4-agent graph:**
   - **Planner** finds no source-keyword or recency hints in the question, so
     `metadata_filter` stays `None` — a plain semantic search across the whole
     corpus.
   - **Retrieval** embeds the question locally (ONNX, no network call for
     embedding), then sends one SQL query to Neon: `SELECT ... ORDER BY
     embedding <=> %s LIMIT 20`. The reranker is skipped (disabled on free
     tier), so the top passages are used directly, and `build_citations()`
     numbers them `[1]` through `[5]`.
   - **Answer** builds the exact prompt (`SYSTEM_PROMPT` + the 5 numbered
     passages + the question) and calls Hugging Face's hosted
     `chat_completion` API with `Qwen/Qwen2.5-7B-Instruct`. This is the slowest
     step (real network round-trip to a 7B model) — a few seconds.
   - **Verify** re-embeds every sentence of the returned answer and every
     retrieved passage, computes cosine similarity, and marks each claim
     supported (≥0.6) or not. In this real, tested example, all 5 claims came
     back supported.

7. **The API serializes the `GroundedAnswer`** (`ans.model_dump()`) into JSON
   and returns it; the middleware logs the completed request with its total
   latency and adds `X-Request-ID`/`X-Response-Time-ms` headers.

8. **Gradio receives the JSON**, `_render_answer()` builds the answer HTML (a
   metadata strip showing "Base + RAG," total time, retrieve/generate
   sub-times, and token count; the answer body with styled `[n]` citation
   markers; a green "✓ All 5 claims grounded" badge; a collapsible per-claim
   breakdown with each sentence's similarity score), and `_render_evidence()`
   builds five source cards, each showing the PubMed badge, the real article
   title linking to the real PubMed URL, and a short quoted excerpt.

9. **The visitor reads a real, cited, self-verified answer**, can click through
   to the actual PubMed abstract, and can expand "Per-claim verification" to see
   exactly which sentence maps to which similarity score.

10. **Separately, a researcher** (not this visitor) had earlier run
    `python scripts/rag_index.py --config configs/corpus.yaml --rebuild` to build the 18,528-chunk
    Neon index this query just searched, and — on a GPU session — had run
    `python scripts/train.py --config configs/qlora_5k.yaml` to produce the
    fine-tuned adapter published at
    `huggingface.co/Udit013/qwen2.5-7b-medmcqa-qlora-5k`, and
    `python scripts/rag_benchmark.py` to produce the `benchmark_explorer.json`
    that powers the second tab this visitor could click next.

11. **If the visitor clicks "Benchmark Explorer"** and picks a curated question
    from the dropdown, `explore()` reads the already-downloaded JSON file (no
    network call, no backend involved at all) and renders a table comparing
    Base, Fine-tuned, Base+RAG, and Fine-tuned+RAG side by side for that exact
    question — including the *true* fine-tuned model's answer, computed
    offline, honestly, on a GPU, in advance.

---

## 21. Interview Question Bank

### Beginner

- **Q: What does RAG stand for and why use it?**
  A: Retrieval-Augmented Generation. It lets an AI answer using real, searched
  documents instead of only what it memorized during training, which reduces
  made-up ("hallucinated") answers and lets the answer cite real sources.
- **Q: What is an embedding, in your own words?**
  A: A way to turn a sentence into a list of numbers so that similar-meaning
  sentences end up with similar numbers, letting a computer search by meaning
  instead of exact keyword matches.
- **Q: Why does this project have more than one `requirements.txt` file?**
  A: Because the live, free-tier server can't afford a multi-gigabyte PyTorch
  install, so it has its own lean dependency list (`requirements-api.txt`) that
  deliberately excludes PyTorch, while the offline index-building and training
  machines use fuller dependency lists.

### Intermediate

- **Q: Walk me through what happens between a user clicking "Ask" and seeing an
  answer.**
  A: See [§16.1](#161-live-query-a-user-asks-a-question) — the full, real trace
  through Gradio → FastAPI → the 4-agent LangGraph workflow → back to the UI.
- **Q: Why is the agent workflow split into 4 separate agents instead of one
  prompt?**
  A: Separation of concerns and testability — each agent is independently unit
  tested; and structurally, verification is a real, separate computation, not
  something the answering model could skip. The graph's linear edges also
  guarantee it terminates — it can't loop forever.
- **Q: How do you know your citation verification actually works?**
  A: It was tested on a real failure: the original lexical (word-overlap)
  version incorrectly flagged a correct, paraphrased sepsis-management answer
  as "unsupported." Switching to embedding-based cosine similarity fixed it —
  verified on the same example, which now correctly scores fully grounded.
- **Q: Why pgvector instead of a dedicated vector database?**
  A: At ~3,400 chunks, dedicated vector DBs' main advantage (massive scale)
  doesn't apply; pgvector means one datastore, plain SQL filters, and free-tier
  availability on Neon.

### Advanced

- **Q: Does your live demo actually run the fine-tuned model?**
  A: No, and this is worth stating proactively. Hugging Face's free serverless
  Inference API can't serve a custom LoRA adapter, so the live deployment
  honestly serves the **base** model plus RAG. The API itself reports exactly
  which config it served (`served_config` on `/health`, `config` on `/query`),
  and the UI displays that truthfully. Pointing the backend at a GPU endpoint
  with the adapter loaded flips it to true Fine-tuned+RAG with **zero** UI or
  API code changes — the whole system was deliberately architected around that
  single environment-variable swap.
- **Q: Tell me about a production incident on this project.**
  A: The live demo started failing every query while `/health` stayed green.
  The 502 body (then still verbose) showed Hugging Face's router sending
  Qwen2.5-7B to Together with `model_not_available`; HF's provider mapping API
  confirmed Together's route was in `error` state and Featherless's was `live`.
  The fix was an ordered provider list with failover instead of trusting
  `auto`, plus three things the incident exposed: `/health` now reports the last
  real query outcome, errors are sanitized (that verbose body could have leaked
  a DSN), and a model outage no longer forces a full service rebuild per request.
  After redeploying, testing showed answers had regressed with the new provider
  (a refusal citing `[1-5]`), which led to the passage-dedupe, marker-normalization
  and abstention fixes in §8.4.
- **Q: How many examples did the model actually train on, and why?**
  A: 5,000, out of MedMCQA's ~194,000 total — deliberately the first validated
  point of a planned data-scaling study (5K → 20K → 50K), with the larger runs
  explicitly scoped as future work gated by GPU budget, not a code limitation.
- **Q: What happens if the vector database goes down?**
  A: A *dropped* connection (Neon closes idle ones) is handled transparently:
  the store reconnects once and retries the query. If the database is truly
  down, the reconnect fails, the error reaches the `/query` handler, and the
  client gets a `502 "Query failed (request <id>)"` — no connection string in the
  body — while the full traceback is logged under that ID and `/health`'s
  `last_query` flips to `ok: false`. There's no fallback to the local NumPy
  store or backoff beyond the single retry — a reasonable next step if
  reliability requirements grew.
- **Q: How would you scale this to a much larger user base?**
  A: The retrieval side would swap the single lock-guarded, reused connection
  for a pool (`psycopg_pool`) — the lock serializes database calls, fine for a
  demo but a bottleneck under real concurrency — and probably HNSW tuning or a
  purpose-built vector database at much larger corpus sizes. Generation is
  delegated to Hugging Face's hosted providers (with failover), so LLM
  throughput is largely their problem. The in-memory rate limiter and
  `last_query` state are per-process, so multiple replicas would move them to
  Redis and rate-limit per authenticated user. The lazy singleton is already
  thread-safe (double-checked locking, tested with 8 concurrent cold starts).
- **Q: Why did you choose to evaluate under matched quantization?**
  A: Comparing an fp16 base model to a 4-bit fine-tuned model would confound
  quantization effects with fine-tuning effects — any accuracy difference could
  be caused by either. Scoring both under identical 4-bit quantization isolates
  the fine-tuning effect specifically, which is the actual research question.
