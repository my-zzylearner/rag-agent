"""
混合检索：BM25 + 向量检索，RRF 融合排序
"""
import re
import functools
from typing import List, Dict, Tuple
from .indexer import get_embedder, _count, _get_all, _query
from utils.logger import get_logger

_logger = get_logger(__name__)

TOP_K = 4
RRF_K = 60  # RRF 平滑常数，标准值 60


def _tokenize(text: str) -> List[str]:
    """简单分词：按空白和标点切分，保留长度 >= 1 的词元。"""
    return [t for t in re.split(r"[\s\u3000\uff0c\u3002\uff01\uff1f\u300a\u300b\u3010\u3011\u2014\u2026，。！？《》【】—…、]+", text.lower()) if t]


def _rrf_score(rank: int) -> float:
    """倒数排名融合得分：1 / (k + rank)，rank 从 1 开始。"""
    return 1.0 / (RRF_K + rank)


_bm25_cache: Tuple = None  # (bm25, all_ids, all_texts, all_metas)


def _build_bm25() -> Tuple:
    """构建 BM25 索引，返回 (bm25, all_ids, all_texts, all_metas)。"""
    from rank_bm25 import BM25Okapi
    all_data = _get_all()
    all_ids = all_data["ids"]
    all_texts = all_data["documents"]
    all_metas = all_data["metadatas"]
    tokenized_corpus = [_tokenize(t) for t in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, all_ids, all_texts, all_metas


def _get_bm25() -> Tuple:
    """获取 BM25 索引缓存，未初始化时构建。"""
    global _bm25_cache
    if _bm25_cache is None:
        _bm25_cache = _build_bm25()
    return _bm25_cache


def invalidate_bm25() -> None:
    """主动失效 BM25 缓存，知识库更新后调用。"""
    global _bm25_cache
    _bm25_cache = None


def warm_up_bm25() -> None:
    """预热 BM25 索引，在应用启动时调用以避免首次查询延迟。"""
    try:
        if _count() > 0:
            _get_bm25()
    except Exception:
        pass  # 知识库尚未初始化时静默跳过


def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    """
    混合检索：向量 + BM25，RRF 融合后返回 top_k 结果。
    返回 [{text, source, score}, ...]，score 为 RRF 融合分（越高越相关）。
    """
    count = _count()
    if count == 0:
        return []

    # 从缓存获取 BM25 索引和语料（知识库未变化时复用，无需重建）
    bm25, all_ids, all_texts, all_metas = _get_bm25()

    if not all_texts:
        return []

    fetch_k = min(top_k * 2, count)

    # ── 向量检索 ────────────────────────────────────────────
    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()[0]
    vec_results = _query(query_embedding, fetch_k)
    _logger.debug("retrieve: vec_results=%d bm25_corpus=%d fetch_k=%d",
                  len(vec_results["ids"][0]), len(all_texts), fetch_k)
    # id → rank（从 1 开始）
    vec_id_set = vec_results["ids"][0]
    vec_ranks: Dict[str, int] = {doc_id: rank + 1 for rank, doc_id in enumerate(vec_id_set)}

    # ── BM25 检索 ────────────────────────────────────────────
    query_tokens = _tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)

    # 取 BM25 top fetch_k 的 index
    import heapq
    top_bm25_indices = heapq.nlargest(fetch_k, range(len(bm25_scores)), key=lambda i: bm25_scores[i])
    bm25_ranks: Dict[str, int] = {all_ids[i]: rank + 1 for rank, i in enumerate(top_bm25_indices)}

    # ── RRF 融合 ─────────────────────────────────────────────
    candidate_ids = set(vec_ranks) | set(bm25_ranks)
    rrf: Dict[str, float] = {}
    for doc_id in candidate_ids:
        score = 0.0
        if doc_id in vec_ranks:
            score += _rrf_score(vec_ranks[doc_id])
        if doc_id in bm25_ranks:
            score += _rrf_score(bm25_ranks[doc_id])
        rrf[doc_id] = score

    top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]

    # ── 组装结果 ─────────────────────────────────────────────
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
    chunks = []
    for doc_id in top_ids:
        idx = id_to_idx[doc_id]
        meta = all_metas[idx] or {}
        chunks.append({
            "text": all_texts[idx],
            "source": meta.get("source", "unknown"),
            "score": round(rrf[doc_id], 6),
        })

    return chunks
