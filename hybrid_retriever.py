"""
Hybrid retrieval: combines vector search (semantic) with BM25 (keyword)
search, fused manually via Reciprocal Rank Fusion (RRF).

We implement RRF directly instead of relying on LangChain's
EnsembleRetriever, since its import path changed across LangChain's
1.0 restructuring — this keeps the fusion logic dependency-free
and fully under our control.
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

import config
from document_store import get_or_build_document_store


def reciprocal_rank_fusion(ranked_lists: list[list[Document]], k: int = 60, top_n: int = 4) -> list[Document]:
    """
    Standard RRF: for each document, sum 1 / (k + rank) across every
    ranked list it appears in. Documents ranked highly by MULTIPLE
    retrievers naturally rise to the top of the fused result.
    k=60 is the constant from the original RRF paper — it dampens the
    influence of any single very-high rank, so score differences are
    driven more by cross-retriever agreement than by one lucky rank 1.
    """
    scores: dict[str, float] = {}
    doc_lookup: dict[str, Document] = {}

    for ranked_docs in ranked_lists:
        for rank, doc in enumerate(ranked_docs):
            key = doc.page_content  # use content as the dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[key] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_lookup[key] for key, _ in fused[:top_n]]


def get_hybrid_retriever_docs(path: str, query: str, k: int = None, candidate_k: int = 10) -> tuple[list, str]:
    """
    Returns (fused_top_k_docs, document_id) for a single query.
    Runs vector search and BM25 search independently, then fuses
    their rankings with RRF.
    """
    k = k or config.TOP_K
    vectordb, doc_id = get_or_build_document_store(path)

    # Pull the exact chunks already stored for this document, so BM25
    # indexes the SAME chunks as the vector store.
    raw = vectordb.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    vector_results = vectordb.as_retriever(search_kwargs={"k": candidate_k}).invoke(query)

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = candidate_k
    bm25_results = bm25.invoke(query)

    ## debug area inserrted for checking
    print(f"\n[DEBUG] Vector top-{candidate_k} previews:")
    for i, d in enumerate(vector_results, 1):
        print(f"  {i}. {d.page_content[:80]}")

    print(f"\n[DEBUG] BM25 top-{candidate_k} previews:")
    for i, d in enumerate(bm25_results, 1):
        print(f"  {i}. {d.page_content[:80]}")

##-----------------------------------------------------------------------
    fused = reciprocal_rank_fusion([vector_results, bm25_results], top_n=k)
    return fused, doc_id


class HybridRetrieverWrapper:
    """Thin wrapper so this can be used with `.invoke(query)`,
    matching the interface graph.py already expects from retriever objects."""
    def __init__(self, path: str, k: int = None):
        self.path = path
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        docs, _ = get_hybrid_retriever_docs(self.path, query, k=self.k)
        return docs


def get_hybrid_retriever_for_file(path: str, k: int = None):
    _, doc_id = get_or_build_document_store(path)
    return HybridRetrieverWrapper(path, k=k), doc_id


if __name__ == "__main__":
    retriever, doc_id = get_hybrid_retriever_for_file(
        "data/raw_contracts/Rent Agreement Km 51 104.pdf"
    )
    for query in ["what is the monthly rent", "what is the security deposit amount"]:
        results = retriever.invoke(query)
        print(f"\nQuery: {query} (doc_id={doc_id})\n")
        for i, doc in enumerate(results, 1):
            print(f"--- Chunk {i} ---")
            print(doc.page_content[:200])
            print()