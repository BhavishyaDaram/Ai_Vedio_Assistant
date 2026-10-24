import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def build_vector_store(transcript: str, persist_dir: str = CHROMA_PATH):
    """Split transcript text and build a Chroma vector store."""
    cleaned_transcript = (transcript or "").strip()
    if not cleaned_transcript:
        cleaned_transcript = "No spoken content was detected in this recording."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([cleaned_transcript])
    if not docs:
        from langchain_core.documents import Document
        docs = [Document(page_content=cleaned_transcript)]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_store


def load_vector_store(persist_dir: str = CHROMA_PATH):
    """Load an existing Chroma vector store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )


def get_retriever(vector_store=None, k: int = 4):
    """Return a retriever from the vector store."""
    if vector_store is None:
        vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
