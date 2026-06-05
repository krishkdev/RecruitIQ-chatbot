from src.data_loader import load_all_documents
from src.vectorstore import ApplyAIVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    # Step 1: Load & index (run once, then comment out build step)
    print("Loading documents...")
    docs = load_all_documents("data")

    print("\nBuilding FAISS indexes...")
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
