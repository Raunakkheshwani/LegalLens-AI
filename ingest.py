"""
Ingest legal documents (PDF or .txt) into a Chroma vector store.

Run: python ingest.py
Drop sample contracts into data/raw_contracts/ before running.
"""
"""
Right now your contract chunks sit in Chroma, but nothing can query them yet.

"""
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community .vectorstores import Chroma

import config


def load_documents(directory: str):
    docs = []
    for fname in os.listdir(directory):
        path = os.path.join(directory, fname)
        if fname.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())
        elif fname.lower().endswith(".txt"):
            docs.extend(TextLoader(path).load())
    return docs

def build_vector_store():
    os.makedirs(config.RAW_DOCS_DIR, exist_ok= True)
    raw_docs= load_documents(config.RAW_DOCS_DIR)

    if not raw_docs:
        raise FileNotFoundError(
            f"No .pdf or .txt files found in {config.RAW_DOCS_DIR}/ — add a sample contract first."
        )
    
    splitter= RecursiveCharacterTextSplitter(
        chunk_size= config.CHUNK_SIZE,
        chunk_overlap= config.CHUNK_OVERLAP

    )

    chunks= splitter.split_documents(raw_docs)
    
    embeddings = HuggingFaceEmbeddings(model_name= config.EMBEDDING_MODEL)

# this is my RAG memory 
    vectordb= Chroma.from_documents(
         documents= chunks,
        embedding= embeddings,
        persist_directory= config.CHROMA_DIR # baar baar is directory ko dalna ni padega ek baar yeh krdia then stay on the app if reloads then also it will be present there 
    )

    print (f"indexes {len(chunks)} chunks from {len(raw_docs)} source documents into {config.CHROMA_DIR}")
    return vectordb

if __name__ == "__main__":
    build_vector_store()