"""
文档加载、Chunking、Embedding、存入向量库

向量库后端自动选择：
- 配置了 QDRANT_URL + QDRANT_API_KEY → Qdrant Cloud（持久化）
- 未配置 → 本地 ChromaDB（开发用）
"""
import os
import glob
import functools

# 禁用 chromadb 遥测，必须在 import chromadb 之前设置
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import hashlib
import re
import time
from typing import List, Dict
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

_logger = get_logger(__name__)

MAX_CHUNK_SIZE = 800
COLLECTION_NAME = "rag_docs"
WEB_CACHE_LIMIT = 200
WEB_CACHE_LIMIT_DAYS = 7
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 维度

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # 模型已在本地缓存时，禁止联网检查更新，避免每次 encode 触发超时重试
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        _embedder = SentenceTransformer(model_name)
    return _embedder


# ── 向量库后端抽象 ────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _use_qdrant() -> bool:
    """检测 Qdrant 是否可用：配置存在 + 连通性探测（5s 超时）。失败自动降级 ChromaDB。"""
    if not (os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")):
        return False
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=5,
        )
        client.get_collections()  # 探测连通性
        _logger.info("qdrant: connection ok, using Qdrant Cloud")
        return True
    except Exception as e:
        _logger.warning("qdrant: connection failed (%s), falling back to ChromaDB", e)
        return False


@functools.lru_cache(maxsize=1)
def _get_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=10,
    )


def _ensure_qdrant_collection():
    """确保 Qdrant collection 存在，不存在时创建；确保 payload index 存在。"""
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
    client = _get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        _logger.info("qdrant: created collection %s", COLLECTION_NAME)

    # 确保 source 字段有 keyword index，_delete_by_filter 过滤时需要
    try:
        collection_info = client.get_collection(COLLECTION_NAME)
        indexed_fields = set(collection_info.payload_schema.keys()) if collection_info.payload_schema else set()
        if "source" not in indexed_fields:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="source",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            _logger.info("qdrant: created payload index for 'source'")
    except Exception as e:
        _logger.warning("qdrant: failed to create payload index: %s", e)


@functools.lru_cache(maxsize=1)
def get_collection():
    """
    进程级单例。
    - Qdrant 模式：返回 QdrantClient（调用方通过 _use_qdrant() 判断后端）
    - ChromaDB 模式：返回 chromadb Collection
    """
    if _use_qdrant():
        _ensure_qdrant_collection()
        return _get_qdrant_client()
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── 统一 count / get / upsert / delete 接口 ──────────────────

def _count() -> int:
    if _use_qdrant():
        try:
            _ensure_qdrant_collection()
            client = _get_qdrant_client()
            return client.count(collection_name=COLLECTION_NAME).count
        except Exception as e:
            _logger.error("qdrant _count failed: %s", e)
            return 0
    return get_collection().count()


def _get_all() -> Dict:
    """返回 {ids, documents, metadatas}，统一格式。"""
    if _use_qdrant():
        _ensure_qdrant_collection()
        client = _get_qdrant_client()
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        ids, documents, metadatas = [], [], []
        for r in records:
            ids.append(str(r.id))
            documents.append(r.payload.get("document", ""))
            metadatas.append({k: v for k, v in r.payload.items() if k != "document"})
        return {"ids": ids, "documents": documents, "metadatas": metadatas}
    result = get_collection().get(include=["documents", "metadatas"])
    return result


def _upsert(ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict]):
    if _use_qdrant():
        _ensure_qdrant_collection()
        from qdrant_client.models import PointStruct
        client = _get_qdrant_client()
        # Qdrant id 必须是 uint64 或 UUID，用 md5 转 int
        points = []
        for uid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            numeric_id = int(hashlib.md5(uid.encode()).hexdigest()[:16], 16)
            payload = {"document": doc, "_str_id": uid, **meta}
            points.append(PointStruct(id=numeric_id, vector=emb, payload=payload))
        client.upsert(collection_name=COLLECTION_NAME, points=points)
    else:
        get_collection().upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


def _delete_by_ids(ids: List[str]):
    if not ids:
        return
    if _use_qdrant():
        from qdrant_client.models import PointIdsList
        client = _get_qdrant_client()
        numeric_ids = [int(hashlib.md5(uid.encode()).hexdigest()[:16], 16) for uid in ids]
        client.delete(collection_name=COLLECTION_NAME,
                      points_selector=PointIdsList(points=numeric_ids))
    else:
        get_collection().delete(ids=ids)


def _delete_by_filter(field: str, value: str, op: str = "eq"):
    """按 metadata 字段过滤删除。op: 'eq' 或 'ne'。"""
    if _use_qdrant():
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchExcept
        if op == "eq":
            condition = FieldCondition(key=field, match=MatchValue(value=value))
        else:
            condition = FieldCondition(key=field, match=MatchExcept(**{"except": [value]}))
        client = _get_qdrant_client()
        client.delete(collection_name=COLLECTION_NAME,
                      points_selector=Filter(must=[condition]))
    else:
        where = {field: {"$eq": value}} if op == "eq" else {field: {"$ne": value}}
        col = get_collection()
        existing = col.get(where=where, include=[])
        if existing["ids"]:
            col.delete(ids=existing["ids"])


def _query(query_embedding: List[float], n_results: int) -> Dict:
    """向量检索，返回 {ids, documents, metadatas, distances}。"""
    if _use_qdrant():
        client = _get_qdrant_client()
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=n_results,
            with_payload=True,
        )
        ids, documents, metadatas, distances = [], [], [], []
        for h in result.points:
            ids.append(str(h.id))
            documents.append(h.payload.get("document", ""))
            metadatas.append({k: v for k, v in h.payload.items() if k != "document"})
            # Qdrant cosine score 是相似度（1=完全相同），转为 distance = 1 - score
            distances.append(1.0 - h.score)
        return {"ids": [ids], "documents": [documents], "metadatas": [metadatas], "distances": [distances]}
    result = get_collection().query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return result


# ── 文档处理（与后端无关）────────────────────────────────────

def _parse_frontmatter(content: str):
    """返回 (meta_dict, body_text)"""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_text = content[3:end].strip()
    body = content[end + 4:].lstrip("\n")
    meta = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def load_documents(docs_dir: str) -> List[Dict]:
    docs = []
    for pattern in ["*.txt", "*.md"]:
        for path in glob.glob(os.path.join(docs_dir, "**", pattern), recursive=True):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            meta, text = _parse_frontmatter(content)
            docs.append({"text": text, "source": os.path.basename(path), "meta": meta})
    try:
        from pypdf import PdfReader
        for path in glob.glob(os.path.join(docs_dir, "**", "*.pdf"), recursive=True):
            reader = PdfReader(path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            docs.append({"text": text, "source": os.path.basename(path), "meta": {}})
    except ImportError:
        pass
    return docs


def chunk_text(text: str, source: str) -> List[Dict]:
    def _split_long_paragraph(para: str) -> List[str]:
        sentences = re.split(r'(?<=[。！？\n])', para)
        parts: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= MAX_CHUNK_SIZE:
                current += sent
            else:
                if current.strip():
                    parts.append(current.strip())
                current = sent
        if current.strip():
            parts.append(current.strip())
        return parts if parts else [para]

    paragraphs = re.split(r'\n\n+', text)
    chunks: List[Dict] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > MAX_CHUNK_SIZE:
            if current.strip():
                chunks.append({"text": current.strip(), "source": source})
                current = ""
            for part in _split_long_paragraph(para):
                chunks.append({"text": part, "source": source})
        elif len(current) + len(para) + 2 <= MAX_CHUNK_SIZE:
            current = (current + "\n\n" + para) if current else para
        else:
            if current.strip():
                chunks.append({"text": current.strip(), "source": source})
            current = para

    if current.strip():
        chunks.append({"text": current.strip(), "source": source})
    return chunks


# ── 公开接口 ──────────────────────────────────────────────────

def index_documents(docs_dir: str = "./data/docs") -> int:
    embedder = get_embedder()

    # 清空旧数据
    if _count() > 0:
        _delete_by_filter("source", "", op="ne")

    docs = load_documents(docs_dir)
    if not docs:
        return 0

    all_chunks = []
    for doc in docs:
        for chunk in chunk_text(doc["text"], doc["source"]):
            chunk["meta"] = doc.get("meta", {})
            all_chunks.append(chunk)

    texts = [c["text"] for c in all_chunks]
    sources = [c["source"] for c in all_chunks]
    metas = [c["meta"] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    _upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": s, "topic": m.get("topic", ""), "type": m.get("type", "knowledge_base")}
                   for s, m in zip(sources, metas)],
    )
    _logger.info("index_documents: indexed %d chunks from %d docs in %s", len(all_chunks), len(docs), docs_dir)

    from rag.retriever import invalidate_bm25
    invalidate_bm25()

    return len(all_chunks)


def is_indexed() -> bool:
    return _count() > 0


def index_single_document(filepath: str) -> int:
    embedder = get_embedder()
    filename = os.path.basename(filepath)

    # 删除该文件的旧 chunks
    _delete_by_filter("source", filename, op="eq")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return 0

    meta, text = _parse_frontmatter(content)
    chunks = chunk_text(text, filename)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    ids = [f"chunk_{filename}_{i}" for i in range(len(chunks))]
    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()

    _upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": filename, "topic": meta.get("topic", ""), "type": meta.get("type", "knowledge_base")}
                   for _ in chunks],
    )

    # 索引更新后失效 BM25 缓存，下次查询时重建
    from rag.retriever import invalidate_bm25
    invalidate_bm25()

    return len(chunks)


def add_chunks(chunks: List[Dict]) -> int:
    if not chunks:
        return 0

    embedder = get_embedder()
    texts, sources, ids, metadatas = [], [], [], []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        source = chunk.get("source", "")
        if not text:
            continue
        uid = hashlib.md5(f"{source}:{text[:64]}".encode()).hexdigest()
        ids.append(f"web_{uid}")
        texts.append(text)
        sources.append(source)
        metadatas.append({"source": source, "type": "web_cache", "added_at": int(time.time())})

    if not texts:
        return 0

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    _upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    # web_cache 上限 + TTL 清理
    all_data = _get_all()
    web_pairs = [
        (id_, meta)
        for id_, meta in zip(all_data["ids"], all_data["metadatas"])
        if id_.startswith("web_") or (meta or {}).get("type") == "web_cache"
    ]
    if len(web_pairs) > WEB_CACHE_LIMIT:
        web_pairs.sort(key=lambda x: x[1].get("added_at", 0))
        to_delete = [id_ for id_, _ in web_pairs[:len(web_pairs) - WEB_CACHE_LIMIT]]
        _delete_by_ids(to_delete)
        web_pairs = web_pairs[len(web_pairs) - WEB_CACHE_LIMIT:]

    threshold = int(time.time()) - WEB_CACHE_LIMIT_DAYS * 24 * 3600
    expired = [id_ for id_, meta in web_pairs if meta.get("added_at", 0) < threshold]
    if expired:
        _delete_by_ids(expired)
        _logger.info("add_chunks: removed %d expired web_cache entries", len(expired))

    return len(texts)
