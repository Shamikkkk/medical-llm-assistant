# Cardiovascular PubMed Assistant (Conversational)

This is a simple web app that helps you search **PubMed** using normal English questions and then gives you a concise, evidence-based response grounded in real PubMed abstracts.

**Important:** This assistant is **specialized in cardiovascular (heart and blood vessel) topics only**.  
If you ask about unrelated medical areas, it will politely refuse.

---

## What this app does

1. You type a cardiovascular research question (example: “Do SGLT2 inhibitors reduce heart failure hospitalization?”).
2. The app converts your question into a PubMed-friendly search query.
3. It pulls the most relevant PubMed abstracts.
4. It summarizes what the abstracts collectively suggest and shows the sources (PMIDs).
5. If you ask the same or a very similar question again, it can respond faster using its query cache.

---

## Who is it for?

- Medical students learning evidence-based medicine
- Clinicians doing quick literature checks
- Biomedical researchers doing hypothesis exploration or rapid review

---

## How to run it (step-by-step)

### 1) Install Python
Make sure you have **Python 3.10+** installed.

### 2) Download this project
Place the project folder on your computer (for example, on Desktop).

### 3) Create a virtual environment (recommended)
Open a terminal in the project folder and run:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

### 4) Install dependencies

```bash
pip install -r requirements.txt
```

This includes the in-app theme toggle component:
- `streamlit-component-theme-changer`

### 5) Set environment variables
Copy the example env file and update it as needed:

```bash
cp .env.example .env
```

At minimum, set:
- `NVIDIA_API_KEY` (required for LLM-driven query rewriting and RAG answers)

Optional:
- `NVIDIA_MODEL` (defaults to `meta/llama-3.1-8b-instruct`)
- `USE_RERANKER=true` (if you have Flashrank installed)

### 6) Run the app

```bash
streamlit run app.py
```

---

## Notes

- The app uses PubMed E-utilities and retrieves abstracts in real time.
- Answers are grounded only in retrieved abstracts and include PMID citations.
- If the NVIDIA key is missing, the app still retrieves papers but skips LLM-based answering.
- In `AGENT_MODE=true`, the app runs a LangGraph-backed tool orchestrator.
- In `EVAL_MODE=true`, the app runs online RAG evaluation sampling and stores results locally.

---

## Project structure

- `app.py`: Streamlit entry point
- `src/core/`: baseline RAG pipeline and chain logic
- `src/agent/`: agent orchestrator and tool wrappers
- `src/integrations/`: PubMed, Chroma, NVIDIA integrations
- `src/ui/`: rendering/formatting helpers
- `eval/`: evaluation runner, metrics, store, and dashboard helpers
- `data/`: local persistence (Chroma vector stores)

---

## Agent mode (optional)

Enable in `.env`:

```bash
AGENT_MODE=true
AGENT_USE_LANGGRAPH=true
```

Behavior:
- `AGENT_MODE=false` (default): existing baseline pipeline runs unchanged.
- `AGENT_MODE=true`: an agent orchestrator routes tool calls for guardrails, query refinement, PubMed retrieval, retrieval, answer synthesis, and citation formatting.

---

## Evaluation mode (optional)

Enable in `.env`:

```bash
EVAL_MODE=true
EVAL_SAMPLE_RATE=0.25
EVAL_STORE_PATH=./data/eval/eval_results.jsonl
OPENAI_API_KEY=<NVIDIA_API_KEY_FOR_OPENAI_COMPAT>
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
# optional:
RAGAS_JUDGE_MODEL=meta/llama-3.1-8b-instruct
```

Behavior:
- Runs online evaluation on sampled user turns.
- Shows an **Evaluation Dashboard** tab in Streamlit.
- Persists per-query metrics to JSONL.
- RAGAS judge uses NVIDIA's OpenAI-compatible endpoint, so no OpenAI key is required.
- If `OPENAI_API_KEY`/`OPENAI_BASE_URL` are missing or RAGAS errors, evaluation falls back to heuristic metrics.

### Offline evaluation

Use the template dataset:

```bash
python -m eval.run_offline --dataset data/eval_set.json --out data/eval/eval_results.jsonl
```

Run in agent mode:

```bash
python -m eval.run_offline --dataset data/eval_set.json --out data/eval/eval_results.jsonl --agent-mode
```

### Smoke test

```bash
python scripts/smoke_agent_eval.py
```
