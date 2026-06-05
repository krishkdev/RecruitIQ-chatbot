from langchain_community.embeddings import HuggingFaceEmbeddings


class ApplyAIEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self._embeddings = HuggingFaceEmbeddings(model_name=model_name)

    @property
    def embeddings(self):
        return self._embeddings


if __name__ == "__main__":
    emb = ApplyAIEmbeddings()
    test = emb.embeddings.embed_query("hello world")
    print(f"Embedding dim: {len(test)}")
