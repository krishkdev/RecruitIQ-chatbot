from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_resumes(directory):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    resume_dir = Path(directory)

    for f in sorted(resume_dir.iterdir()):
        if f.suffix == ".txt":
            loader = TextLoader(str(f), encoding="utf-8")
        elif f.suffix == ".pdf":
            loader = PyMuPDFLoader(str(f))
        else:
            continue

        loaded = loader.load()
        chunks = splitter.split_documents(loaded)
        for chunk in chunks:
            chunk.metadata["source_type"] = "resume"
            chunk.metadata["candidate_name"] = f.stem
        docs.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {f.name}")

    return docs


def load_jds(directory):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = []
    jd_dir = Path(directory)

    for f in sorted(jd_dir.iterdir()):
        if f.suffix != ".txt":
            continue

        loader = TextLoader(str(f), encoding="utf-8")
        loaded = loader.load()
        chunks = splitter.split_documents(loaded)
        for chunk in chunks:
            chunk.metadata["source_type"] = "jd"
            chunk.metadata["job_title"] = f.stem
        docs.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {f.name}")

    return docs


def load_all_documents(data_dir):
    data_path = Path(data_dir)
    print("Loading resumes...")
    resumes = load_resumes(data_path / "resumes")
    print("Loading job descriptions...")
    jds = load_jds(data_path / "jds")
    print(f"Total: {len(resumes)} resume chunks, {len(jds)} JD chunks")
    return {"resumes": resumes, "jds": jds}


if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Resumes: {len(docs['resumes'])} chunks")
    print(f"JDs: {len(docs['jds'])} chunks")
