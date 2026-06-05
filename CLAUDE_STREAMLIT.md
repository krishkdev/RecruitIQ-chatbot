# ApplyAI RAG Chatbot — Streamlit Extension

## What this adds
A Streamlit chatbot UI on top of the existing `src/` RAG pipeline. Three modes,
persistent chat history, pre-built FAISS index (no rebuild on startup).

---

## New files to create

```
rag-pipeline/
├── chatbot.py               # NEW — Streamlit entrypoint (replaces running app.py)
├── src/
│   └── chat_history.py      # NEW — disk-persisted chat history manager
├── chat_logs/               # NEW — auto-created at runtime, stores JSON chat logs
└── .streamlit/
    └── config.toml          # NEW — theme config
```

Do NOT modify any existing `src/` modules or `app.py`.

---

## Module Specs

### `src/chat_history.py`
Manages chat history as a JSON file per session.

```python
# Interface:
class ChatHistory:
    def __init__(self, log_dir: str = "chat_logs"):
        # creates log_dir if not exists
        # generates session_id = datetime string (YYYYMMDD_HHMMSS)
        # self.filepath = f"{log_dir}/{session_id}.json"
        # self.messages = []  — list of {"role": "user"|"assistant", "content": str, "mode": str, "timestamp": str}

    def add(self, role: str, content: str, mode: str): ...
        # appends to self.messages and immediately writes to self.filepath as JSON

    def load_session(self, filepath: str): ...
        # loads a previous session from a JSON file path

    @staticmethod
    def list_sessions(log_dir: str = "chat_logs") -> list[str]: ...
        # returns sorted list of .json filepaths in log_dir, newest first
```

### `chatbot.py`
Full Streamlit app. Use `st.session_state` for all runtime state.

#### Layout
- **Sidebar:**
  - ApplyAI logo text (`st.title` or `st.markdown` with styled header)
  - Mode selector: `st.radio` with three options:
    - `"💬 Recruiter Q&A"` — free-form question over resume corpus
    - `"🎯 Candidate Matching"` — paste a JD, get ranked candidates
    - `"🔍 Gap Analysis"` — paste a JD + select a candidate, get ✅/⚠️/❌ breakdown
  - `st.divider()`
  - **Past Sessions** section: `st.selectbox` listing previous session files from `chat_logs/`; a "Load" button loads that session's messages into the chat
  - `st.divider()`
  - "Clear Chat" button — clears `st.session_state.messages` and starts a new `ChatHistory` session

- **Main area:**
  - `st.title("ApplyAI Recruiting Assistant")`
  - Mode-specific input area **above** the chat (not inside the chat):
    - **Recruiter Q&A:** just the standard chat input at bottom (`st.chat_input`)
    - **Candidate Matching:** `st.text_area("Paste Job Description", height=150)` + `st.button("Find Matches")` — results stream into chat as assistant message
    - **Gap Analysis:** two columns — `st.text_area("Job Description", height=120)` | `st.selectbox("Select Candidate", options=candidate_names)` + `st.button("Run Gap Analysis")`
  - Chat history display: loop `st.session_state.messages` and render each with `st.chat_message(role)`
  - For Q&A mode: `st.chat_input("Ask about candidates...")` at bottom

#### Initialization (runs once via `st.session_state` guard)
```python
if "rag" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        store = ApplyAIVectorStore("faiss_store")
        store.load("resumes")
        store.load("jds")
        st.session_state.rag = RAGSearch()   # RAGSearch loads store + Groq internally
        st.session_state.chat_history = ChatHistory()
        st.session_state.messages = []
        # Pre-populate candidate names for gap analysis dropdown
        # Query all resumes from FAISS metadata or load from data/resumes/ filenames
        st.session_state.candidate_names = _get_candidate_names("data/resumes")
```

#### Helper
```python
def _get_candidate_names(resumes_dir: str) -> list[str]:
    # returns sorted list of filename stems from data/resumes/
    # e.g. ["aisha_patel", "carlos_mendez", "wei_zhang"]
```

#### Mode behavior

**Recruiter Q&A:**
- User types in `st.chat_input`
- Display user message, call `rag.recruiter_qa(query)`, stream response with `st.write_stream` if Groq supports it, else just `st.markdown`
- Append both messages to `st.session_state.messages` and `chat_history.add(...)`

**Candidate Matching:**
- User pastes JD and clicks "Find Matches"
- Call `rag.match_candidates(jd_text, top_k=5)`
- Format results as a markdown table: `| Rank | Candidate | Score |`
- Display as assistant message in chat
- Save to history with `mode="matching"`

**Gap Analysis:**
- User selects candidate from dropdown, pastes JD, clicks "Run Gap Analysis"
- Load that candidate's resume text from `data/resumes/<name>.txt` (or `.pdf`)
- Call `rag.gap_analysis(jd_text, resume_text, candidate_name)`
- Display result as assistant message
- Save to history with `mode="gap_analysis"`

---

## `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#800020"
backgroundColor = "#0F0F0F"
secondaryBackgroundColor = "#1A1A1A"
textColor = "#F5F5F5"
font = "sans serif"
```
(Maroon primary matches ApplyAI's theme)

---

## `requirements.txt` additions
Add these lines to the existing `requirements.txt` (do not remove anything):
```
streamlit
```
That's the only new dependency. Everything else (`langchain-groq`, `faiss-cpu`, etc.) is already there.

---

## Coding conventions (match existing repo)
- Load `.env` at top of `chatbot.py` with `load_dotenv()`
- `st.session_state` for all mutable state — never use global variables
- Keep all RAG logic in `src/` — `chatbot.py` only calls `RAGSearch` methods, never LangChain directly
- Error handling: wrap RAG calls in `try/except`, show `st.error(...)` on failure, never crash the app
- Show `st.spinner("Thinking...")` during all RAG calls

---

## How to run
```bash
streamlit run chatbot.py
```

---

## Success criteria
1. App loads without rebuilding FAISS (uses existing `faiss_store/`)
2. All three modes produce visible output in chat
3. Chat history saves a `.json` file in `chat_logs/` after each message
4. Loading a past session from the sidebar restores the full conversation
5. No import errors — all imports resolve against current `requirements.txt` + `streamlit`
