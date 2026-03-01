# Created by Codex — Session Prompt Reference

# CODEX PROMPT — Full Rewrite: Angular Frontend + Backend Enhancements

```
You are working on the `medical-llm-assistant` repository. Below is a complete reference snapshot of the repo (you already have it). Your job is to implement four major sets of changes. Work through them sequentially. For each change, list every file you create, modify, or delete before writing any code.

---

## OVERVIEW OF CHANGES

1. **Replace the Streamlit frontend with an Angular 17+ single-page application**
2. **Convert the Python backend into a proper FastAPI REST/streaming API**
3. **Improve PubMed paper retrieval quality**
4. **Improve LLM answer generation quality**

Additionally, the new Angular UI must have:
- Light / Dark mode toggle (persisted in localStorage)
- Claude-style branching UX (edit a message → branch opens inline, old thread preserved)
- Highly creative, polished, user-friendly design

Work through each section completely before moving to the next.

---

## SECTION 1 — FastAPI Backend (replaces Streamlit as the serving layer)

### Why
Streamlit's runtime is tightly coupled to the UI. Angular needs a proper HTTP/streaming API. The existing pipeline logic (`src/`) stays almost entirely intact — we are only replacing the Streamlit presentation layer (`app.py`, `src/ui/`) with FastAPI routes.

### Files to CREATE

**`api/main.py`**
- FastAPI app entry point
- Configure CORS to allow `http://localhost:4200` (Angular dev server) and the production origin
- Include all routers from `api/routers/`
- Mount static files from `frontend/dist/medical-llm-assistant/browser` at `/` for production serving
- Add a `/health` endpoint returning `{"status": "ok"}`
- Startup: call `load_dotenv()` and `setup_logging()`

**`api/routers/chat.py`**
Implement these endpoints:

```
POST /api/chat/invoke
Body: {
  query: str,
  session_id: str,
  branch_id: str = "main",
  top_n: int = 10,
  agent_mode: bool = False,
  follow_up_mode: bool = True,
  chat_messages: list[dict],
  show_papers: bool = True,
  conversation_summary: str = ""
}
Response: PipelineResponse (JSON)
```

```
POST /api/chat/stream
Same body as above.
Response: text/event-stream (SSE)
  - Each chunk: data: {"type": "chunk", "text": "..."}\n\n
  - Final: data: {"type": "done", "payload": <full PipelineResponse>}\n\n
  - Errors: data: {"type": "error", "message": "..."}\n\n
```

Use `stream_chat_request` from `src/chat/router.py` for streaming.
Use `invoke_chat_request` from `src/chat/router.py` for invoke.

**`api/routers/sessions.py`**
```
GET  /api/sessions          → list all chats [{chat_id, title, created_at, branch_count}]
POST /api/sessions          → create new chat → {chat_id, branch_id: "main"}
DELETE /api/sessions/{chat_id} → delete chat and all its branches
GET  /api/sessions/{chat_id}/branches → list branches for a chat
POST /api/sessions/{chat_id}/branches → create branch from edit:
  Body: { parent_branch_id: str, fork_message_index: int, edited_query: str }
  Response: { branch_id: str }
GET  /api/sessions/{chat_id}/branches/{branch_id}/messages → list messages for a branch
```

Implement a lightweight in-memory session store (dict keyed by chat_id). Add disk persistence via a JSON file at `data/sessions.json` — load on startup, save on every mutation.

**`api/routers/config.py`**
```
GET /api/config → return config.masked_summary() as JSON
```

**`api/routers/papers.py`**
```
GET /api/papers/{pmid} → fetch and return PaperContent (from PaperStore or fetch live)
```

**`api/routers/eval.py`** (only if EVAL_MODE=true)
```
GET /api/eval/results → return list of eval records from EvalStore
GET /api/metrics → return summarize_metric_events(read_metric_events(...))
```

**`api/dependencies.py`**
- Provide a `get_config()` FastAPI dependency that returns `load_config()` (cached with `functools.lru_cache`)
- Provide `get_session_store()` dependency

**`api/models.py`**
- Pydantic v2 request/response models mirroring `src/types.py`:
  - `ChatRequest`
  - `ChatResponse` (wraps `PipelineResponse`)
  - `SessionRecord`, `BranchRecord`, `MessageRecord`
  - `BranchCreateRequest`

**`api/session_store.py`**
- `SessionStore` class with methods:
  - `create_chat(title: str) → ChatRecord`
  - `get_chats() → list[ChatRecord]`
  - `delete_chat(chat_id: str) → bool`
  - `get_branches(chat_id: str) → list[BranchRecord]`
  - `create_branch(chat_id, parent_branch_id, fork_message_index, edited_query) → BranchRecord`
  - `get_messages(chat_id, branch_id) → list[MessageRecord]`
  - `append_message(chat_id, branch_id, message: MessageRecord) → None`
- Auto-persist to `data/sessions.json` after every write
- Load from disk on init if file exists

### Files to MODIFY

**`requirements.txt`** — add:
```
fastapi>=0.115,<0.116
uvicorn[standard]>=0.30,<0.32
sse-starlette>=1.8,<2.0
pydantic>=2.7,<3.0
httpx>=0.27,<0.28
```

**`Dockerfile`** — update CMD to:
```dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Add a build step that runs `ng build` inside `frontend/` before the Python image layer, and copies `frontend/dist/` into the container.

**`.env.example`** — add:
```
API_PORT=8000
FRONTEND_ORIGIN=http://localhost:4200
```

**`.github/workflows/ci.yml`** — add a frontend job:
```yaml
frontend:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: "20"
        cache: "npm"
        cache-dependency-path: frontend/package-lock.json
    - run: cd frontend && npm ci
    - run: cd frontend && npm run build -- --configuration production
    - run: cd frontend && npm test -- --watch=false --browsers=ChromeHeadless
```

### Files to DELETE (Streamlit-specific, no longer needed)
- `app.py`
- `src/ui/render.py`
- `src/ui/metrics_dashboard.py`
- `src/ui/formatters.py` — KEEP only `strip_reframe_block`, `beautify_text`, `pubmed_url`, `doi_url` by moving them to `src/utils/text.py`
- `src/ui/loading_messages.py`
- `src/ui/__init__.py`
- `src/app_state.py`
- `.streamlit/config.toml`

Keep all of `src/core/`, `src/agent/`, `src/chat/`, `src/integrations/`, `src/papers/`, `src/validators/`, `src/observability/`, `src/history.py`, `src/types.py`, `src/logging_utils.py`, `src/intent.py`, `src/validator.py`.

---

## SECTION 2 — Angular Frontend

### Bootstrap

Run these commands from the repo root:
```bash
npm install -g @angular/cli@17
ng new frontend \
  --directory frontend \
  --routing true \
  --style scss \
  --ssr false \
  --skip-tests false \
  --standalone true
cd frontend
npm install @angular/cdk @angular/material lucide-angular marked
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init
```

Configure `frontend/tailwind.config.js`:
```js
module.exports = {
  content: ["./src/**/*.{html,ts}"],
  darkMode: "class",        // class-based so we can toggle via JS
  theme: { extend: {} },
  plugins: [],
};
```

Add to `frontend/src/styles.scss`:
```scss
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Set `frontend/src/proxy.conf.json`:
```json
{
  "/api": {
    "target": "http://localhost:8000",
    "secure": false,
    "changeOrigin": true
  }
}
```

Add to `angular.json` under `serve > options`:
```json
"proxyConfig": "src/proxy.conf.json"
```

---

### Design Direction

**Aesthetic: "Clinical Precision" — a refined, editorial medical-tech interface.**

- **Dark mode base**: deep navy `#08101e` background, with subtle blue-grey panels `#0f1c2e`. Crisp white text `#f0f4f8`.
- **Light mode**: warm off-white `#f7f9fc` background, soft grey panels `#eef1f6`, near-black text `#0d1117`.
- **Accent**: electric teal `#00d4b4` — used for interactive elements, active states, streaming cursors.
- **Typography**: `DM Serif Display` for headings/app name (gives it a medical-journal gravitas), `JetBrains Mono` for PMID citations and query text, `Lato` for body/UI copy. Load from Google Fonts.
- **Panels**: subtle glassmorphism on cards — `backdrop-filter: blur(12px)`, `background: rgba(255,255,255,0.04)`.
- **Animations**: message bubbles slide up and fade in on entry (150ms, cubic-bezier). Streaming cursor is a pulsing teal block. Branch switch transition: a 200ms horizontal slide.
- **Branch tree**: displayed as a vertical thread in the sidebar — each branch is a node connected by a thin teal line, just like Claude's interface.

---

### File Structure to Create under `frontend/src/app/`

```
app/
├── app.component.ts/html/scss       ← root shell with router-outlet
├── app.config.ts                    ← standalone app config + providers
├── app.routes.ts                    ← routes
│
├── core/
│   ├── services/
│   │   ├── chat.service.ts          ← HTTP + SSE calls to FastAPI
│   │   ├── session.service.ts       ← session/branch CRUD
│   │   ├── theme.service.ts         ← light/dark toggle, localStorage persistence
│   │   └── config.service.ts        ← fetch /api/config on startup
│   ├── models/
│   │   ├── chat.models.ts           ← Message, Branch, ChatSession, PipelineResponse
│   │   └── source.models.ts         ← SourceItem
│   └── interceptors/
│       └── error.interceptor.ts
│
├── features/
│   ├── chat/
│   │   ├── chat.component.ts/html/scss          ← main chat view
│   │   ├── message-bubble/
│   │   │   ├── message-bubble.component.ts/html/scss
│   │   ├── source-card/
│   │   │   ├── source-card.component.ts/html/scss
│   │   ├── branch-composer/
│   │   │   ├── branch-composer.component.ts/html/scss  ← edit + fork UI
│   │   └── streaming-cursor/
│   │       └── streaming-cursor.component.ts/html/scss
│   └── sidebar/
│       ├── sidebar.component.ts/html/scss
│       ├── session-list/
│       │   └── session-list.component.ts/html/scss
│       └── branch-tree/
│           └── branch-tree.component.ts/html/scss      ← Claude-style branch graph
│
└── shared/
    ├── components/
    │   ├── theme-toggle/
    │   │   └── theme-toggle.component.ts/html/scss
    │   ├── loading-indicator/
    │   │   └── loading-indicator.component.ts/html/scss
    │   └── empty-state/
    │       └── empty-state.component.ts/html/scss
    └── pipes/
        ├── markdown.pipe.ts        ← uses `marked` to render answer markdown
        └── relative-time.pipe.ts
```

---

### Detailed Component Specs

#### `ThemeService` (`core/services/theme.service.ts`)
```typescript
// Signal-based service
// - `isDark = signal<boolean>(...)` — initialize from localStorage('theme') or prefers-color-scheme
// - `toggle()` — flips the signal, writes to localStorage, adds/removes class 'dark' on document.documentElement
// - `init()` — call in app.config.ts APP_INITIALIZER
```

#### `ThemeToggleComponent` (`shared/components/theme-toggle/`)
- A button in the top-right of the app header
- Shows a sun icon in dark mode, moon icon in light mode (use lucide-angular icons)
- Smooth 300ms transition on the icon swap using Angular animations
- Pill-shaped, teal accent border on hover

#### `ChatService` (`core/services/chat.service.ts`)
```typescript
// streamChat(request: ChatRequest): Observable<StreamEvent>
//   - Opens EventSource to POST /api/chat/stream
//   - Since EventSource doesn't support POST, use fetch() with ReadableStream
//   - Parse SSE lines manually: lines starting with "data: " → JSON.parse
//   - Emit { type: 'chunk', text: string } or { type: 'done', payload: PipelineResponse }
//
// invokeChat(request: ChatRequest): Observable<PipelineResponse>
//   - POST /api/chat/invoke → returns full PipelineResponse
```

#### `ChatComponent` (`features/chat/chat.component.ts`)
- Maintains `messages = signal<Message[]>([])`
- Maintains `isStreaming = signal<boolean>(false)`
- On submit: appends user message, starts streaming, accumulates chunks into a growing assistant message
- Implements auto-scroll to bottom on each new chunk (use `@ViewChild` + `scrollIntoView`)
- Shows `<app-loading-indicator>` while waiting for first chunk
- Shows `<app-streaming-cursor>` at the end of a streaming message

#### `MessageBubbleComponent`
- `@Input() message: Message`
- `@Input() messageIndex: number`
- User messages: right-aligned, teal-tinted background, `JetBrains Mono` font
- Assistant messages: left-aligned, glassmorphism card
- Answer text rendered through `MarkdownPipe`
- "Edit" button (pencil icon, appears on hover) for user messages — emits `(editRequested)` event up to ChatComponent
- Timestamp shown in muted small text using `RelativeTimePipe`
- If `message.answer_cache_hit`, show a small "⚡ Cached" badge

#### `BranchComposerComponent` (THE KEY BRANCHING UX)

This replicates Claude's branching behavior exactly:

**Trigger**: User clicks the pencil/edit icon on any past user message bubble.

**Behavior**:
1. The original message thread stays visible above the edit point, grayed out / at reduced opacity (0.5).
2. Below the fork point, a text area slides in (animated, 200ms ease-in) pre-filled with the original query text.
3. A "Create Branch" button and a "Cancel" button appear.
4. On "Create Branch":
   - Call `POST /api/sessions/{chat_id}/branches` with `{ parent_branch_id, fork_message_index, edited_query }`
   - Receive new `branch_id`
   - Switch active branch to the new one
   - The messages above the fork point are copied into the new branch's history
   - Immediately submit the edited query as the first message in the new branch
5. The branch tree in the sidebar updates to show the new node branching off the parent at the correct message index.
6. Cancel: composer slides out, original thread returns to full opacity.

**Implementation notes**:
- `ChatComponent` holds `editingMessageIndex = signal<number | null>(null)`
- When non-null, pass it to each `MessageBubbleComponent` so messages after the index know to show at reduced opacity
- `BranchComposerComponent` is injected between the message at `editingMessageIndex` and the messages below it (use `@for` with index tracking)

#### `BranchTreeComponent` (sidebar — Claude-style visual)
```
Main ──●── Message 1
       ├── Message 2
       │    └─[branch A]──●── Edited Message 2
       │                   └── Message 3
       └── Message 3 (main continues)
```
- Render as an SVG or styled div tree
- Active branch node: filled teal circle
- Inactive branch nodes: hollow circle with teal border
- Connecting lines: 1px teal lines
- Click any node → switches to that branch (calls `SessionService.switchBranch()`)
- Animate active node with a subtle pulse

#### `SidebarComponent`
- Collapsible (toggle button, animated slide)
- Top section: "New Chat" button, search/filter input
- Middle: `<app-session-list>` — list of recent chats, each showing title + branch count badge
- Bottom per-chat: `<app-branch-tree>`
- Controls section:
  - Top-N slider (1–20)
  - Follow-up mode toggle
  - Show papers toggle
  - Compute device selector (auto / cpu / gpu)
  - Export buttons (Markdown, JSON)
  - Cache clear buttons (query cache, answer cache, paper cache)
  - Response time metric (last response ms, shown as a subtle badge)

#### `SourceCardComponent`
- Compact card: title (truncated), journal + year, PMID badge (monospace teal pill), DOI link
- Expandable to show abstract snippet on click
- Rank badge (1, 2, 3...) with teal background
- External link icons open PubMed and DOI in new tab

#### `StreamingCursorComponent`
- A blinking teal block cursor rendered at the end of a streaming assistant message
- CSS animation: `opacity 0.7s ease-in-out infinite alternate`
- Removed from DOM when `isStreaming` becomes false

#### `LoadingIndicatorComponent`
- Three dots with staggered bounce animation
- Shown in the assistant message bubble position while waiting for the first chunk
- Uses a cycling medical-topic loading message (port the Python `pick_loading_message` logic to TypeScript)

---

### Routing (`app.routes.ts`)
```typescript
[
  { path: '', redirectTo: 'chat', pathMatch: 'full' },
  { path: 'chat', component: ChatComponent },
  { path: 'chat/:sessionId', component: ChatComponent },
  { path: 'chat/:sessionId/branch/:branchId', component: ChatComponent },
]
```

On route params change, `ChatComponent` loads messages for that session+branch from the session service.

---

### App Shell (`app.component.html`)
```html
<div [class.dark]="themeService.isDark()">
  <div class="app-root min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] flex">
    <app-sidebar />
    <main class="flex-1 flex flex-col">
      <header class="app-header">
        <!-- App name, subtitle, theme toggle -->
        <app-theme-toggle />
      </header>
      <router-outlet />
    </main>
  </div>
</div>
```

Define all color variables in `styles.scss`:
```scss
:root {
  --bg-primary: #f7f9fc;
  --bg-panel: #eef1f6;
  --text-primary: #0d1117;
  --text-muted: #6b7280;
  --accent: #00d4b4;
  --border: #dde3ed;
}
.dark {
  --bg-primary: #08101e;
  --bg-panel: #0f1c2e;
  --text-primary: #f0f4f8;
  --text-muted: #8b9ab0;
  --accent: #00d4b4;
  --border: #1e2d42;
}
```

---

## SECTION 3 — Improve PubMed Paper Retrieval

### Problems to fix
1. Single-pass keyword search often misses relevant papers
2. No MeSH term expansion
3. Negative caching is coarse
4. Reranking is disabled by default and uses a simplistic lexical heuristic

### Changes to make in `src/integrations/pubmed.py` and `src/core/retrieval.py`

**In `src/integrations/pubmed.py`:**

Add `build_multi_strategy_queries(user_query: str, llm) -> list[str]`:
- Uses the LLM to generate 3 complementary PubMed queries from the same user question:
  1. The original rewritten query (already exists)
  2. A MeSH-term focused variant (prompt: "Rewrite this as a PubMed query using MeSH terms only: ...")
  3. A broader concept variant with Boolean OR expansions (prompt: "Write a broader PubMed query using OR to capture related concepts: ...")
- Return all 3 as a list

Add `multi_strategy_esearch(queries: list[str], retmax_each: int = 15) -> list[str]`:
- Run all 3 queries in parallel using `ThreadPoolExecutor`
- Merge the resulting PMID lists, de-duplicate, preserve order of first occurrence
- Cap at `retmax_each * 2` total PMIDs to avoid over-fetching

Update the main fetch path in both `src/core/pipeline.py` and `src/agent/tools.py`:
- Replace the single `pubmed_esearch(pubmed_query, retmax=top_n)` call with:
  ```python
  queries = build_multi_strategy_queries(user_query=effective_query, llm=llm)
  pmids = multi_strategy_esearch(queries, retmax_each=max(top_n, 12))
  ```
- Then `pubmed_efetch(pmids[:top_n * 2])` to fetch more candidates than `top_n` for reranking

**In `src/core/retrieval.py`:**

Improve `hybrid_rerank_documents()`:
- Currently uses a simplistic token overlap. Replace with a proper BM25 + semantic hybrid:
  1. BM25 score: implement lightweight BM25 (pure Python, no extra library — `rank_bm25` is not in requirements but you can implement the formula directly in ~30 lines)
  2. Semantic score: use the existing Chroma cosine similarity score from document metadata (field `_distance` or `score` if available; otherwise fall back to token overlap)
  3. Combine: `final_score = alpha * semantic_score + (1 - alpha) * bm25_score` where `alpha = config.hybrid_alpha` (default 0.5)
- Always run this reranking regardless of `USE_RERANKER` flag when more than `top_n` candidate docs are available
- Add `explain_reranking: bool = False` parameter — if True, log which documents were promoted/demoted and why

Add `deduplicate_by_semantic_similarity(docs: list[Document], threshold: float = 0.92) -> list[Document]`:
- Before returning the final context, remove documents whose abstracts have >92% token Jaccard similarity to a document already selected
- Prevents near-duplicate abstracts from padding the context

**In `src/core/config.py`:** add:
```python
multi_strategy_retrieval: bool  # default True, env: MULTI_STRATEGY_RETRIEVAL
retrieval_candidate_multiplier: int  # default 3, env: RETRIEVAL_CANDIDATE_MULTIPLIER
```

**In `.env.example`:** add:
```
MULTI_STRATEGY_RETRIEVAL=true
RETRIEVAL_CANDIDATE_MULTIPLIER=3
```

---

## SECTION 4 — Improve Answer Generation Quality

### Problems to fix
1. The system prompt is too generic and doesn't enforce structure well enough
2. LLM sometimes omits citations or cites non-existent PMIDs
3. Answers can be too verbose or poorly organized
4. No evidence-quality signaling to the user

### Changes to make in `src/core/chains.py`

**Rewrite the RAG chain system prompt** (replace the existing one in `build_rag_chain()`):

```python
SYSTEM_PROMPT = """You are a rigorous clinical literature assistant. Your sole source of truth is the numbered list of PubMed abstracts provided in the context. Follow these rules without exception:

STRUCTURE (always use this exact structure):
## Direct Answer
One to three sentences directly answering the question. Be concrete. Do not hedge unless the evidence genuinely conflicts.

## Evidence Summary
2–5 bullet points. Each bullet MUST end with the PMID in brackets, e.g. [PMID: 12345678].
Only cite PMIDs present in the provided abstracts. Never invent or guess a PMID.
Each bullet should state a specific finding, not a vague generality.

## Evidence Quality
One sentence assessing overall evidence quality: label it as one of — Strong (≥3 concordant RCTs), Moderate (observational or mixed), Preliminary (case reports or single studies), or Insufficient (no relevant abstracts).

## Caveats (optional)
Include only if evidence is weak, contradictory, or the question is outside scope.

STRICT RULES:
- Never say "I don't have access to full text" — just use what is in the abstracts.
- Never include a "Reframe:" section.
- Never describe your reasoning process to the user.
- If the abstracts contain no relevant information, respond only with: "The provided abstracts do not contain sufficient evidence to answer this question."
- Keep the total response under 450 words unless evidence complexity genuinely requires more.
- Use plain, clinician-accessible language. Avoid jargon unless it is in the source material.
"""
```

**Add an answer post-processing step** in both `src/core/pipeline.py` and `src/agent/orchestrator.py`:

Add `validate_citations_in_answer(answer: str, source_pmids: list[str]) -> tuple[str, list[str]]`:
- Regex-extract all `[PMID: XXXXXXXX]` patterns from the answer
- Check each against `source_pmids`
- If any cited PMID is NOT in source_pmids, replace it in the answer with `[PMID: UNAVAILABLE]` and log a warning
- Return `(cleaned_answer, list_of_invalid_pmids)`
- Store the invalid PMID list in the response payload as `invalid_citations: list[str]`

**Add evidence quality extraction** to the pipeline response:
- Parse the "## Evidence Quality" section from the answer
- Store the label (Strong / Moderate / Preliminary / Insufficient) in `payload["evidence_quality"]`
- The Angular UI will display this as a colored badge on the assistant message

**Improve follow-up contextualization** in `src/chat/contextualize.py`:
- The current LLM rewrite prompt is not included in the code shown. Add/replace with:
```python
REWRITE_PROMPT = """You are rewriting a follow-up question so it is fully self-contained for a PubMed literature search.

Previous topic: {topic_summary}
Conversation so far: {history}
Follow-up question: {query}

Rules:
- Produce exactly one rewritten question
- Include the medical condition, intervention, and outcome from prior context
- Do NOT include phrases like "as discussed" or "as mentioned"
- Maximum 25 words
- Output ONLY the rewritten question, nothing else

Rewritten question:"""
```

**In `src/integrations/pubmed.py`**, improve `rewrite_to_pubmed_query()`:
```python
PUBMED_REWRITE_PROMPT = """Convert this medical question into a concise PubMed search query.
Use MeSH terms where applicable. Use AND to combine concepts. Do not use quotes. Maximum 12 words.
Question: {query}
PubMed query:"""
```

---

## SECTION 5 — Integration Checklist

After implementing all sections above, verify:

- [ ] `uvicorn api.main:app --reload` starts without error
- [ ] `GET http://localhost:8000/health` returns `{"status": "ok"}`
- [ ] `POST http://localhost:8000/api/chat/invoke` with a test payload returns a valid `PipelineResponse`
- [ ] `POST http://localhost:8000/api/chat/stream` streams SSE chunks correctly
- [ ] `ng serve` in `frontend/` starts without error, proxies `/api` to FastAPI
- [ ] Theme toggle persists across page refreshes
- [ ] Editing a message creates a new branch and the branch tree updates in the sidebar
- [ ] Source cards render with correct PMID links
- [ ] Evidence quality badge appears on assistant messages
- [ ] `pytest -q` still passes (backend tests unchanged or updated for new API surface)
- [ ] CI pipeline passes both frontend and backend jobs

---

## SECTION 6 — What NOT to change

- `src/core/pipeline.py` invoke/stream logic beyond the specific edits above
- `src/agent/orchestrator.py` beyond citation validation
- `src/integrations/storage.py` — Chroma store logic stays intact
- `src/validators/` — validation pipeline stays intact
- `eval/` — evaluation pipeline stays intact
- All existing pytest tests in `tests/` — update mocks only if signatures changed
- `src/history.py` — LangChain message history stays as-is
- `data/` directory structure

---

## OUTPUT FORMAT EXPECTATION

For each file you create or modify, output:
1. The full file path
2. The complete file contents (no truncation)
3. A one-line comment at the top of each file: `# Created by Codex — <section name>`

When done, output a summary table:
| File | Action (created/modified/deleted) | Section |
|------|----------------------------------|---------|
| ...  | ...                              | ...     |
```

---
