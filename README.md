# RecruitIQ — AI-Powered Recruiting Assistant

A modular RAG (Retrieval-Augmented Generation) pipeline and Streamlit chatbot for intelligent resume screening. Match candidates to job descriptions, run recruiter Q&A over a resume corpus, and generate structured gap analyses — all powered by FAISS vector search and Groq LLMs.

---

## Features

- **Candidate Matching** — paste a job description, get ranked candidates by semantic similarity
- **Recruiter Q&A** — ask free-form questions over the resume corpus ("Who has production ML experience?")
- **Gap Analysis** — structured ✅/⚠️/❌ breakdown of a candidate against a specific JD
- **Persistent Chat History** — every session saved as JSON in `chat_logs/`, loadable from the sidebar
- **Maroon/dark UI** — ApplyAI brand theme via Streamlit config

---

## Tech Stack

| Layer | Library |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) via `langchain-groq` |
| Embeddings | `all-MiniLM-L6-v2` via `langchain-community` HuggingFaceEmbeddings |
| Vector store | FAISS (local, persistent) via `langchain-community` |
| Document loading | `PyMuPDFLoader`, `TextLoader` via `langchain-community` |
| UI | Streamlit |
| Orchestration | LangChain (`langchain`, `langchain-core`, `langchain-community`) |

---

## Project Structure

```
RecruitIQ-chatbot/
├── src/
│   ├── data_loader.py      # Load resumes + JDs, chunk with RecursiveCharacterTextSplitter
│   ├── embedding.py        # HuggingFaceEmbeddings wrapper
│   ├── vectorstore.py      # FAISS two-collection store (resumes, jds)
│   ├── retriever.py        # LangChain retriever from FAISS collection
│   └── search.py           # RAGSearch — match_candidates, recruiter_qa, gap_analysis
├── data/
│   ├── resumes/            # .txt or .pdf resume files
│   └── jds/                # .txt job description files
├── chatbot.py              # Streamlit app (three modes + chat history)
├── app.py                  # CLI entrypoint — index builder + demo
├── .streamlit/
│   └── config.toml         # Maroon/dark theme
├── .env.example            # API key template
└── requirements.txt
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/krishkdev/RecruitIQ-chatbot.git
cd RecruitIQ-chatbot
pip install -r requirements.txt
```

### 2. Set your Groq API key

```bash
cp .env.example .env
# edit .env and add your key
GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Add your data

Drop resume files (`.txt` or `.pdf`) into `data/resumes/` and job descriptions (`.txt`) into `data/jds/`. Sample files for three candidates and two JDs are included.

### 4. Build the FAISS indexes

```bash
python app.py
```

This loads all documents, builds two FAISS indexes (`faiss_store/resumes/` and `faiss_store/jds/`), and runs a quick demo of candidate matching and recruiter Q&A.

### 5. Launch the chatbot

```bash
streamlit run chatbot.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Chatbot Modes

### 💬 Recruiter Q&A
Ask natural-language questions over the resume corpus.

> "Which candidates have production ML experience?"  
> "Who has worked with PyTorch and GCP?"  
> "Does anyone have a research background?"

### 🎯 Candidate Matching
Paste any job description and get candidates ranked by semantic similarity score.

### 🔍 Gap Analysis
Select a candidate from the dropdown, paste a JD, and get a structured breakdown:

```
✅ Strong match — PyTorch experience clearly demonstrated  
⚠️  Partial match — GCP mentioned but limited production depth  
❌ Missing — No fraud detection experience found  
```

---

## Sample Data

Three sample resumes are included out of the box:

| Candidate | Role | Highlights |
|---|---|---|
| `alice_chen` | Senior ML Engineer | Fraud detection at Stripe, PyTorch, GCP Vertex AI |
| `bob_kumar` | Senior Backend Engineer | Distributed systems at Databricks/Lyft, Go, FastAPI |
| `carol_rodriguez` | AI Research Scientist | LLMs, RLHF at Cohere, 12 publications |

Two sample JDs are included: `senior_ml_engineer` and `backend_engineer`.

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Required. Groq API key for LLM inference. |

---

## Notes

- `faiss_store/` and `chat_logs/` are git-ignored — they are generated at runtime.
- Run `python app.py` once to build indexes before launching the chatbot.
- To add new candidates, drop files into `data/resumes/` and re-run `python app.py`.
