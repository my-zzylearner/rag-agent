"""
混合检索：BM25 + 向量检索，RRF 融合排序
"""
import re
from typing import List, Dict, Tuple

from utils.logger import get_logger
from .indexer import get_embedder, _count, _get_all, _query

_logger = get_logger(__name__)

TOP_K = 4
RRF_K = 60  # RRF 平滑常数，标准值 60

# ── jieba 模块级初始化（懒加载，首次使用时触发，词典只加载一次）────────────
# 利用 Python sys.modules 缓存：import 同一模块多次只有第一次真正加载词典
_pseg = None       # jieba.posseg，加载成功后缓存
_jieba_ok = None   # True=可用，False=未安装，None=未检测

# 词性过滤常量：保留实词（名词/动词/形容词/数字/英文），过滤虚词
_KEEP_PREFIX = {"n", "v", "a", "m"}
_KEEP_FLAG = {"eng", "x"}
_SINGLE_SYMBOL = re.compile(r'^[_\-\+\*/\\|<>=~`@#$%^&]+$')
_FALLBACK_SPLIT = re.compile(
    r"[\s\u3000\uff0c\u3002\uff01\uff1f\u300a\u300b\u3010\u3011\u2014\u2026，。！？《》【】—…、]+"
)


def _init_jieba():
    """首次调用时尝试加载 jieba，结果缓存到模块变量，后续调用直接返回。"""
    global _pseg, _jieba_ok
    if _jieba_ok is not None:
        return _jieba_ok
    try:
        import jieba
        import jieba.posseg as pseg
        jieba.setLogLevel("ERROR")  # 静默模式，只需设置一次
        _pseg = pseg
        _jieba_ok = True
        _logger.debug("jieba initialized successfully")
    except ImportError:
        _jieba_ok = False
        _logger.warning("jieba not installed, BM25 will use simple tokenizer (Chinese recall degraded)")
    return _jieba_ok


def _tokenize(text: str) -> List[str]:
    """
    中英文混合分词，基于 jieba 词性标注过滤虚词：
    - 保留实词：名词(n*/nr/ns/nt/nz)、动词(v*/vn/vd)、形容词(a*)、英文(eng/x)、数字(m)
    - 过滤虚词：助词(u*/uj)、连词(c)、介词(p)、语气词(y)、叹词(e)、标点符号等
    - 额外过滤：单个符号（如 _ ）、空白词元
    jieba 词典只在首次调用 _init_jieba() 时加载一次，后续复用 sys.modules 缓存。
    未安装 jieba 时自动降级为按标点切分。
    """
    if _init_jieba():
        words = _pseg.lcut(text.lower())
        return [
            w.word for w in words
            if w.word.strip()
            and not _SINGLE_SYMBOL.match(w.word)
            and (w.flag[:1] in _KEEP_PREFIX or w.flag in _KEEP_FLAG)
        ]
    # 降级：按标点/空白切分
    return [t for t in _FALLBACK_SPLIT.split(text.lower()) if t]


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

    # ── 组装结果 + 来源统计 ──────────────────────────────────
    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
    chunks = []
    vec_only = bm25_only = both = 0
    for doc_id in top_ids:
        idx = id_to_idx.get(doc_id)
        if idx is None:
            # 向量库有但 BM25 缓存没有（缓存未及时失效），跳过并触发缓存失效
            invalidate_bm25()
            continue
        # 统计来源
        in_vec = doc_id in vec_ranks
        in_bm25 = doc_id in bm25_ranks
        if in_vec and in_bm25:
            both += 1
        elif in_vec:
            vec_only += 1
        else:
            bm25_only += 1

        meta = all_metas[idx] or {}
        chunks.append({
            "text": all_texts[idx],
            "source": meta.get("source", "unknown"),
            "score": round(rrf[doc_id], 6),
        })

    # 日志（INFO 级别，DEBUG=true 时在 Streamlit Cloud Logs 可见）
    score_range = (round(min(c["score"] for c in chunks), 6), round(max(c["score"] for c in chunks), 6)) if chunks else (0, 0)
    _logger.info("retrieve: query=%r top_k=%d vec_only=%d bm25_only=%d both=%d total=%d score_range=%s",
                 query, top_k, vec_only, bm25_only, both, len(chunks), score_range)

    # 异步持久化到 Gist（失败静默）
    try:
        import threading as _threading
        from utils.gist_store import add_retrieval_stat
        _threading.Thread(
            target=add_retrieval_stat,
            args=(query, top_k, vec_only, bm25_only, both),
            daemon=True,
        ).start()
    except Exception:
        pass

    return chunks
