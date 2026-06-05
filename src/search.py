import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import ApplyAIVectorStore

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class RAGSearch:
    def __init__(self):
        self.store = ApplyAIVectorStore("faiss_store")
        print("Loading resume index...")
        self.resume_index = self.store.load("resumes")
        print("Loading JD index...")
        self.jd_index = self.store.load("jds")
        self.llm = ChatGroq(model="llama-3.1-8b-instant")
        print("RAGSearch ready.")

    def match_candidates(self, jd_text, top_k=3):
        results = self.resume_index.similarity_search_with_score(jd_text, k=top_k)
        matches = []
        for doc, score in results:
            matches.append({
                "candidate_name": doc.metadata.get("candidate_name", "unknown"),
                "score": score,
                "content": doc.page_content,
            })
        return matches

    def recruiter_qa(self, query, top_k=3):
        docs = self.resume_index.similarity_search(query, k=top_k)
        context = "\n\n".join(
            f"[{d.metadata.get('candidate_name', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        prompt = (
            f"You are a recruiting assistant. Answer the question using only the candidate "
            f"information provided below.\n\n"
            f"Candidate excerpts:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        response = self.llm.invoke(prompt)
        return response.content

    def gap_analysis(self, jd_text, resume_text, candidate_name):
        prompt = (
            f"You are a senior recruiter. Analyze how well {candidate_name} fits the job below.\n\n"
            f"Job Description:\n{jd_text}\n\n"
            f"Candidate Resume:\n{resume_text}\n\n"
            f"Provide a structured gap analysis using:\n"
            f"✅ Strong match — requirement clearly met\n"
            f"⚠️  Partial match — some evidence, but gaps remain\n"
            f"❌ Missing — no evidence of this requirement\n\n"
            f"Gap Analysis for {candidate_name}:"
        )
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    rag = RAGSearch()
    jd = "Senior ML Engineer with PyTorch, GCP, and fraud detection experience."
    matches = rag.match_candidates(jd, top_k=3)
    for m in matches:
        print(f"{m['candidate_name']} | score: {m['score']:.4f}")
