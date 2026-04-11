"""
文档加载、Chunking、Embedding、存入 ChromaDB
"""
import os
import glob
import functools

# 禁用 chromadb 遥测，必须在 import chromadb 之前设置
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
# chromadb 依赖的 opentelemetry 用了旧版 protobuf 生成代码，需强制用纯 Python 实现
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import hashlib
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb

CHUNK_SIZE = 512      # 每个 chunk 的字符数
CHUNK_OVERLAP = 64    # 相邻 chunk 的重叠字符数，保留上下文连贯性
COLLECTION_NAME = "rag_docs"
# 用绝对路径，避免不同调用方工作目录不同导致 ChromaDB "different settings" 冲突
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        # 优先用环境变量指定的镜像，国内用 modelscope 镜像避免访问 HuggingFace 超时
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        _embedder = SentenceTransformer(model_name)
    return _embedder


@functools.lru_cache(maxsize=1)
def get_collection():
    """进程级单例，lru_cache 保证 PersistentClient 只初始化一次。"""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


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
    """加载 docs_dir 下所有 .txt / .md / .pdf 文件（递归子目录），返回 [{text, source, meta}]"""
    docs = []
    patterns = ["*.txt", "*.md"]
    for pattern in patterns:
        for path in glob.glob(os.path.join(docs_dir, "**", pattern), recursive=True):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            meta, text = _parse_frontmatter(content)
            docs.append({"text": text, "source": os.path.basename(path), "meta": meta})

    # PDF 支持（可选，需要 pypdf）
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
    """将长文本切成带 overlap 的 chunks"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_documents(docs_dir: str = "./data/docs") -> int:
    """加载文档 → Chunking → Embedding → 存入 ChromaDB，返回入库 chunk 数量"""
    collection = get_collection()
    embedder = get_embedder()

    # 清空旧数据（重新索引时使用）
    existing = collection.count()
    if existing > 0:
        collection.delete(where={"source": {"$ne": ""}})

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

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": s, "topic": m.get("topic", ""), "type": m.get("type", "knowledge_base")} for s, m in zip(sources, metas)],
    )

    return len(all_chunks)


def is_indexed() -> bool:
    """检查知识库是否已有数据"""
    return get_collection().count() > 0


def index_single_document(filepath: str) -> int:
    """
    增量重索引单个文件：删除该文件的旧 chunks，重新 chunk + embed + 写入。
    返回新写入的 chunk 数量。
    """
    collection = get_collection()
    embedder = get_embedder()
    filename = os.path.basename(filepath)

    # 删除该文件的旧 chunks
    existing = collection.get(where={"source": {"$eq": filename}}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # 读取文件内容
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

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": filename, "topic": meta.get("topic", ""), "type": meta.get("type", "knowledge_base")} for _ in chunks],
    )
    return len(chunks)


def add_chunks(chunks: List[Dict]) -> int:
    """
    增量写入 chunks 到 ChromaDB，自动去重，返回实际新增数量。
    chunks 格式：[{"text": str, "source": str}, ...]
    """
    if not chunks:
        return 0

    collection = get_collection()
    embedder = get_embedder()

    texts, sources, ids, metadatas = [], [], [], []
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        source = chunk.get("source", "")
        if not text:
            continue
        # 用 source + 内容前64字符 生成唯一 id，避免重复写入
        uid = hashlib.md5(f"{source}:{text[:64]}".encode()).hexdigest()
        ids.append(f"web_{uid}")
        texts.append(text)
        sources.append(source)
        metadatas.append({"source": source, "type": "web_cache"})

    if not texts:
        return 0

    embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
    # upsert 天然去重：已存在的 id 更新，不存在的新增
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(texts)
