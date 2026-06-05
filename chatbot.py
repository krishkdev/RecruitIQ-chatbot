import os
import fitz  # PyMuPDF
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from src.search import RAGSearch
from src.vectorstore import ApplyAIVectorStore
from src.chat_history import ChatHistory

load_dotenv()


def _get_candidate_names(resumes_dir):
    p = Path(resumes_dir)
    if not p.exists():
        return []
    return sorted(f.stem for f in p.iterdir() if f.suffix in (".txt", ".pdf"))


# ── Session-state init (runs once per browser session) ──────────────────────
if "rag" not in st.session_state:
    with st.spinner("Loading RAG pipeline..."):
        store = ApplyAIVectorStore("faiss_store")
        store.load("resumes")
        store.load("jds")
        st.session_state.rag = RAGSearch()
        st.session_state.chat_history = ChatHistory()
        st.session_state.messages = []
        st.session_state.candidate_names = _get_candidate_names("data/resumes")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 ApplyAI")
    st.markdown("**Recruiting Intelligence Platform**")
    st.divider()

    mode = st.radio(
        "Mode",
        ["💬 Recruiter Q&A", "🎯 Candidate Matching", "🔍 Gap Analysis"],
    )

    st.divider()

    st.markdown("**Past Sessions**")
    sessions = ChatHistory.list_sessions()
    if sessions:
        selected_session = st.selectbox(
            "Sessions",
            options=sessions,
            format_func=lambda x: Path(x).stem,
            label_visibility="collapsed",
        )
        if st.button("Load Session"):
            st.session_state.chat_history.load_session(selected_session)
            st.session_state.messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history.messages
            ]
            st.rerun()
    else:
        st.caption("No past sessions yet.")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatHistory()
        st.rerun()


# ── Main area ────────────────────────────────────────────────────────────────
st.title("ApplyAI Recruiting Assistant")

rag = st.session_state.rag

# Mode-specific input widgets (above chat history)
if mode == "🎯 Candidate Matching":
    jd_input = st.text_area("Paste Job Description", height=150, key="jd_matching")
    if st.button("Find Matches"):
        if jd_input.strip():
            user_msg = f"**Find candidates for JD:**\n\n{jd_input}"
            st.session_state.messages.append({"role": "user", "content": user_msg})
            st.session_state.chat_history.add("user", user_msg, "matching")

            with st.spinner("Thinking..."):
                try:
                    matches = rag.match_candidates(jd_input, top_k=5)
                    rows = ["| Rank | Candidate | Score |", "|------|-----------|-------|"]
                    for i, m in enumerate(matches, 1):
                        rows.append(f"| {i} | {m['candidate_name']} | {m['score']:.4f} |")
                    response = "\n".join(rows)
                except Exception as e:
                    response = f"Error running candidate matching: {e}"
                    st.error(str(e))

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history.add("assistant", response, "matching")
            st.rerun()
        else:
            st.warning("Please paste a job description first.")

elif mode == "🔍 Gap Analysis":
    col1, col2 = st.columns(2)
    with col1:
        jd_gap = st.text_area("Job Description", height=120, key="jd_gap")
    with col2:
        candidate = st.selectbox(
            "Select Candidate",
            options=st.session_state.candidate_names,
        )

    if st.button("Run Gap Analysis"):
        if jd_gap.strip() and candidate:
            resume_path_txt = Path(f"data/resumes/{candidate}.txt")
            resume_path_pdf = Path(f"data/resumes/{candidate}.pdf")
            resume_text = ""

            if resume_path_txt.exists():
                resume_text = resume_path_txt.read_text(encoding="utf-8")
            elif resume_path_pdf.exists():
                doc = fitz.open(str(resume_path_pdf))
                resume_text = "\n".join(page.get_text() for page in doc)
                doc.close()
            else:
                st.error(f"Resume file not found for '{candidate}'.")

            if resume_text:
                user_msg = f"**Gap Analysis:** {candidate}\n\n**JD:** {jd_gap}"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                st.session_state.chat_history.add("user", user_msg, "gap_analysis")

                with st.spinner("Thinking..."):
                    try:
                        response = rag.gap_analysis(jd_gap, resume_text, candidate)
                    except Exception as e:
                        response = f"Error running gap analysis: {e}"
                        st.error(str(e))

                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.add("assistant", response, "gap_analysis")
                st.rerun()
        else:
            st.warning("Please fill in the job description and select a candidate.")

# Chat message display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Q&A chat input — always at bottom, only in Recruiter Q&A mode
if mode == "💬 Recruiter Q&A":
    if query := st.chat_input("Ask about candidates..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.chat_history.add("user", query, "qa")

        with st.spinner("Thinking..."):
            try:
                response = rag.recruiter_qa(query)
            except Exception as e:
                response = f"Error: {e}"
                st.error(str(e))

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history.add("assistant", response, "qa")
        st.rerun()
