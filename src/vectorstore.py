import os
from langchain_community.vectorstores import FAISS
from src.embedding import ApplyAIEmbeddings


class ApplyAIVectorStore:
    def __init__(self, store_path):
        self.store_path = store_path
        self._embeddings = ApplyAIEmbeddings().embeddings

    def build_from_documents(self, docs, collection):
        path = os.path.join(self.store_path, collection)
        print(f"Building FAISS index for '{collection}' ({len(docs)} chunks)...")
        index = FAISS.from_documents(docs, self._embeddings)
        index.save_local(path)
        print(f"Saved index to {path}/")

    def load(self, collection):
        path = os.path.join(self.store_path, collection)
        return FAISS.load_local(path, self._embeddings, allow_dangerous_deserialization=True)

    def query(self, query_text, collection, top_k=3):
        index = self.load(collection)
        return index.similarity_search(query_text, k=top_k)


if __name__ == "__main__":
    from src.data_loader import load_all_documents
    docs = load_all_documents("data")
    store = ApplyAIVectorStore("faiss_store")
    store.build_from_documents(docs["resumes"], "resumes")
    results = store.query("PyTorch fraud detection", "resumes", top_k=2)
    for r in results:
        print(r.metadata, r.page_content[:80])
