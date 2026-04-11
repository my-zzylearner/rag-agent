"""
向量检索：给定 query，返回最相关的 top-k chunks
"""
from typing import List, Dict
from .indexer import get_embedder, get_collection

TOP_K = 4  # 返回最相关的 4 个片段


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    返回 [{text, source, score}, ...]
    score 越接近 1 表示越相关（cosine similarity）
    """
    embedder = get_embedder()
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embedder.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "score": round(1 - dist, 4),  # cosine distance → similarity
        })

    return chunks
