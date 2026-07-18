"""
Right now your contract chunks sit in Chroma, but nothing can query them yet. This file wraps the vector store into a clean, reusable retrieval function that graph.py will call later.


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config

def get_retriever(k: int= config.TOP_K):

    embeddings = HuggingFaceEmbeddings(model_name= config.EMBEDDING_MODEL)

    vectordb= Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function= embeddings
    )

    return vectordb.as_retriever(search_kwargs= {"k": k})

if __name__ == "__main__":
    # quick manual test — run: python retriever.py
    retriever = get_retriever()
    query = "what is the security amount?"
    results = retriever.invoke(query)

    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source', 'unknown')}) ---")
        print(doc.page_content[:300])
        print()

        """

"""
Wraps vector store access as retrieval functions.

get_retriever()              -> old behavior: single shared collection,
                                 useful for quick manual testing (python retriever.py)
get_retriever_for_file(path) -> new behavior: isolated, cached per-document
                                 collection, used by the real pipeline
"""


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

import config
from document_store import get_retriever_for_document


def get_retriever(k: int = config.TOP_K):
    """Legacy: queries the single shared collection built by the old
    ingest.py. Kept only for quick manual sanity checks."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
    )
    return vectordb.as_retriever(search_kwargs={"k": k})


def get_retriever_for_file(path: str, k: int = None):
    """Real path: isolated, cached, per-document retriever.
    Returns (retriever, document_id)."""
    return get_retriever_for_document(path, k=k)


if __name__ == "__main__":
    retriever = get_retriever()
    query = "Is there a penalty for late payment?"
    results = retriever.invoke(query)
    print(f"\nQuery: {query}\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source', 'unknown')}) ---")
        print(doc.page_content[:300])
        print()