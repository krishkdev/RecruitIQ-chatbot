def get_retriever(store, collection, top_k=3):
    index = store.load(collection)
    return index.as_retriever(search_kwargs={"k": top_k})


if __name__ == "__main__":
    from src.vectorstore import ApplyAIVectorStore
    store = ApplyAIVectorStore("faiss_store")
    retriever = get_retriever(store, "resumes", top_k=2)
    docs = retriever.invoke("machine learning engineer")
    for d in docs:
        print(d.metadata.get("candidate_name"), d.page_content[:60])
