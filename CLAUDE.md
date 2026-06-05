# ApplyAI RAG — Claude Code Instructions

## Project Overview
Build a modular RAG pipeline for ApplyAI on top of the existing repo structure (krishnaik06/RAG-Tutorials style). The pipeline covers two use cases:
1. **Semantic Matching** — given a Job Description, rank candidate resumes by relevance
2. **Recruiter Q&A** — answer natural-language recruiter queries over the resume + JD corpus

---

## Stack (match existing repo exactly — do NOT introduce new packages)
```
langchain
langchain-core
langchain-community
langchain-groq
sentence-transformers
faiss-cpu
pypdf
pymupdf
python-dotenv
langgraph         # only for agentic extension, not core pipeline
```

LLM provider: **Groq** via `langchain-groq` (use `ChatGroq`)  
Embeddings: **HuggingFace sentence-transformers** via `langchain-community` (`HuggingFaceEmbeddings`, model: `all-MiniLM-L6-v2`)  
Vector store: **FAISS** (local, persistent) via `langchain-community` (`FAISS`)  
Document loading: `langchain-community` loaders (`PyMuPDFLoader`, `TextLoader`, `JSONLoader`)  
Text splitting: `langchain.text_splitter.RecursiveCharacterTextSplitter`

---

## Target Directory Structure
Preserve the existing repo layout exactly. Add only inside `src/` and at root:

```
rag-pipeline/
├── src/
│   ├── __init__.py          # already exists — do not modify
│   ├── data_loader.py       # REPLACE with ApplyAI version (see spec below)
│   ├── embedding.py         # REPLACE with ApplyAI version
│   ├── vectorstore.py       # REPLACE with ApplyAI version
│   ├── search.py            # REPLACE with ApplyAI version
│   └── retriever.py         # NEW — dual-collection retriever logic
├── data/
│   ├── resumes/             # NEW — put sample .txt or .pdf resumes here
│   └── jds/                 # NEW — put sample .txt job descriptions here
├── app.py                   # REPLACE with ApplyAI entrypoint
├── .env                     # NEW — API keys (never commit)
├── .env.example             # NEW — template
├── requirements.txt         # DO NOT change packages, only pin versions if needed
└── CLAUDE.md                # this file
```

---

## Module Specs

### `src/data_loader.py`
- Function `load_resumes(directory: str) -> list[Document]`
  - Loads all `.txt` and `.pdf` files from `data/resumes/`
  - Uses `TextLoader` for `.txt`, `PyMuPDFLoader` for `.pdf`
  - Adds metadata: `{"source_type": "resume", "candidate_name": <filename stem>}`
- Function `load_jds(directory: str) -> list[Document]`
  - Loads all `.txt` files from `data/jds/`
  - Adds metadata: `{"source_type": "jd", "job_title": <filename stem>}`
- Function `load_all_documents(data_dir: str) -> dict`
  - Returns `{"resumes": [...], "jds": [...]}`
- Use `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` on all docs before returning

### `src/embedding.py`
- Class `ApplyAIEmbeddings`
  - `__init__`: loads `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`
  - Property `embeddings`: returns the embedding model instance
  - Keep it a thin wrapper — same pattern as the original repo

### `src/vectorstore.py`
- Class `ApplyAIVectorStore`
  - `__init__(self, store_path: str)`: sets local FAISS store path
  - `build_from_documents(self, docs: list[Document], collection: str)`: builds and saves FAISS index to `{store_path}/{collection}` (use `FAISS.from_documents` then `.save_local`)
  - `load(self, collection: str)`: loads FAISS index from `{store_path}/{collection}` (use `FAISS.load_local`)
  - `query(self, query_text: str, collection: str, top_k: int = 3) -> list[Document]`: similarity search
  - Two collections: `"resumes"` and `"jds"` — always pass collection name explicitly

### `src/retriever.py` (NEW)
- Function `get_retriever(store: ApplyAIVectorStore, collection: str, top_k: int = 3)`
  - Returns a LangChain retriever from the loaded FAISS collection
  - Use `store.load(collection).as_retriever(search_kwargs={"k": top_k})`

### `src/search.py`
- Class `RAGSearch`
  - `__init__`: instantiates `ApplyAIVectorStore("faiss_store")`, loads both collections, instantiates `ChatGroq(model="llama-3.1-8b-instant")` from env var `GROQ_API_KEY`
  - `match_candidates(self, jd_text: str, top_k: int = 3) -> list[dict]`
    - Queries the `"resumes"` collection with `jd_text`
    - Returns list of `{"candidate_name": ..., "score": ..., "content": ...}`
  - `recruiter_qa(self, query: str, top_k: int = 3) -> str`
    - Retrieves top-k resume chunks for `query`
    - Builds a prompt with retrieved context + query
    - Calls `ChatGroq` and returns the answer string
  - `gap_analysis(self, jd_text: str, resume_text: str, candidate_name: str) -> str`
    - Calls `ChatGroq` with a structured gap analysis prompt
    - Returns ✅/⚠️/❌ bullet analysis

### `app.py`
Replace with:
```python
from src.data_loader import load_all_documents
from src.vectorstore import ApplyAIVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    # Step 1: Load & index (run once, then comment out build step)
    docs = load_all_documents("data")
    store = ApplyAIVectorStore("faiss_store")
    store.build_from_documents(docs["resumes"], "resumes")
    store.build_from_documents(docs["jds"], "jds")

    # Step 2: Use cases
    rag = RAGSearch()

    # Use case 1: Match candidates to a JD
    jd_text = "Senior ML Engineer with PyTorch, GCP, and fraud detection experience."
    matches = rag.match_candidates(jd_text, top_k=3)
    print("\n--- Top Candidates ---")
    for m in matches:
        print(f"  {m['candidate_name']} | score: {m['score']:.4f}")

    # Use case 2: Recruiter Q&A
    answer = rag.recruiter_qa("Which candidates have production ML experience?")
    print("\n--- Recruiter Q&A ---")
    print(answer)
```

---

## Environment Variables
Create `.env` at root:
```
GROQ_API_KEY=your_groq_key_here
```

Create `.env.example`:
```
GROQ_API_KEY=
```

Load in every module that needs it with:
```python
from dotenv import load_dotenv
load_dotenv()
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

---

## Sample Data
Create 3 sample resume `.txt` files in `data/resumes/` and 2 JD `.txt` files in `data/jds/` with realistic ApplyAI-style content (ML engineers, backend engineers, AI researchers). This lets the pipeline run end-to-end immediately.

---

## Coding Conventions (match original repo style)
- No type annotations beyond basic hints — keep it readable like the tutorial code
- Each `src/` module is self-contained and importable independently
- No dataclasses, no Pydantic — plain dicts for return values
- Print progress messages in `app.py` (e.g. `"Loading documents..."`, `"Building FAISS index..."`)
- `if __name__ == "__main__":` guard in `app.py`
- All file paths relative to project root

---

## What NOT to do
- Do NOT use `chromadb` — the repo uses FAISS
- Do NOT use `langchain-openai` or Anthropic SDK — use `langchain-groq` + `ChatGroq`
- Do NOT use `langchain` v0.0.x import paths — use the modern split packages (`langchain-community`, `langchain-core`)
- Do NOT modify `notebook/`, `agenticrag/`, or any existing `.ipynb` files
- Do NOT add Docker, FastAPI, or any web layer — this is the prototype module only
- Do NOT pin package versions unless a version conflict actually occurs

---

## Success Criteria
Running `python app.py` should:
1. Load sample resumes and JDs from `data/`
2. Build and persist two FAISS indexes in `faiss_store/`
3. Print top-3 matched candidates for a sample JD
4. Print a recruiter Q&A answer from Groq

No errors, no missing imports.
