"""
Content-addressable, per-document vector store management.

Each document gets a unique ID derived from its content + the chunking/
embedding config used to index it. If that exact combination was already
indexed, we load the existing collection instead of rebuilding it.
If chunk_size, chunk_overlap, or the embedding model ever changes,
the ID changes too — so we never silently serve a stale-shaped index.
"""
import hashlib
import json
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

STORE_ROOT = Path(config.CHROMA_DIR)
STORE_ROOT.mkdir(parents=True, exist_ok=True)


def _file_fingerprint(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _document_id(path: str) -> str:
    """Hash of file content + every parameter that affects the index shape.
    Changing chunk_size/overlap/embedding_model produces a new ID —
    so we never accidentally load an index built with different settings."""
    meta = {
        "file_hash": _file_fingerprint(path),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


def _load_document(path: str):
    if path.lower().endswith(".pdf"):
        return PyPDFLoader(path).load()
    elif path.lower().endswith(".txt"):
        return TextLoader(path).load()
    return []


def get_or_build_document_store(path: str) -> tuple[Chroma, str]:
    """
    Returns (vectorstore, document_id) for a single uploaded file.
    Loads an existing isolated collection if this exact file+config
    combo was already indexed; otherwise builds it fresh.
    """
    doc_id = _document_id(path)
    doc_dir = STORE_ROOT / doc_id
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

    if doc_dir.exists():
        print(f"[cache hit] Loading existing index for {os.path.basename(path)} (id={doc_id})")
        vectordb = Chroma(persist_directory=str(doc_dir), embedding_function=embeddings)
        return vectordb, doc_id

    print(f"[cache miss] Building new index for {os.path.basename(path)} (id={doc_id})")
    raw_docs = _load_document(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)

    doc_dir.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=str(doc_dir)
    )
    (doc_dir / "meta.json").write_text(json.dumps({
        "source_file": os.path.abspath(path),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "embedding_model": config.EMBEDDING_MODEL,
        "num_chunks": len(chunks),
    }, indent=2))
    return vectordb, doc_id


def get_retriever_for_document(path: str, k: int = None):
    vectordb, doc_id = get_or_build_document_store(path)
    retriever = vectordb.as_retriever(search_kwargs={"k": k or config.TOP_K})
    return retriever, doc_id