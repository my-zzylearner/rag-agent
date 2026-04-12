"""
文档加载、Chunking、Embedding、存入 ChromaDB
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
import chromadb

from utils.logger import get_logger

_logger = get_logger(__name__)

MAX_CHUNK_SIZE = 800  # 每个 chunk 的最大字符数
COLLECTION_NAME = "rag_docs"
WEB_CACHE_LIMIT = 200  # web_cache 条目上限，超出时删除最旧的
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
    """按语义边界切分文本：优先在段落处切，单段过长时回退到句子级别。"""

    def _split_long_paragraph(para: str) -> List[str]:
        """单个段落超过 MAX_CHUNK_SIZE 时，按句子边界拆分。"""
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

    # 按双换行切成段落
    paragraphs = re.split(r'\n\n+', text)

    chunks: List[Dict] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > MAX_CHUNK_SIZE:
            # 先把已积累内容入库
            if current.strip():
                chunks.append({"text": current.strip(), "source": source})
                current = ""
            # 长段落拆成句子级 chunks
            for part in _split_long_paragraph(para):
                chunks.append({"text": part, "source": source})
        elif len(current) + len(para) + 2 <= MAX_CHUNK_SIZE:
            # +2 预留段落间分隔符 \n\n
            current = (current + "\n\n" + para) if current else para
        else:
            # 当前积累已满，先提交再开新 chunk
            if current.strip():
                chunks.append({"text": current.strip(), "source": source})
            current = para

    if current.strip():
        chunks.append({"text": current.strip(), "source": source})

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

    _logger.info("index_documents: indexed %d chunks from %d documents in %s", len(all_chunks), len(docs), docs_dir)
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
        metadatas.append({"source": source, "type": "web_cache", "added_at": int(time.time())})

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

    # 超出上限时删除最旧的条目（web_ 前缀的 id 即为 web_cache 条目）
    result = collection.get(include=["metadatas"])
    web_pairs = [
        (id_, meta)
        for id_, meta in zip(result["ids"], result["metadatas"])
        if id_.startswith("web_")
    ]
    if len(web_pairs) > WEB_CACHE_LIMIT:
        web_pairs.sort(key=lambda x: x[1].get("added_at", 0))
        to_delete = [id_ for id_, _ in web_pairs[: len(web_pairs) - WEB_CACHE_LIMIT]]
        collection.delete(ids=to_delete)

    return len(texts)
