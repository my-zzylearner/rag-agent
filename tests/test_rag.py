"""
RAG 核心链路测试：Chunking、索引、检索
"""
import os

import pytest

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from rag.indexer import add_chunks, chunk_text, get_collection, MAX_CHUNK_SIZE
from rag.retriever import retrieve


def test_chunk_text_basic():
    """正常文本能被切成多个 chunk"""
    # 用多个段落（\n\n 分隔），每段接近 MAX_CHUNK_SIZE，确保触发切分
    para = "A" * (MAX_CHUNK_SIZE - 10)
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = chunk_text(text, "test.md")
    assert len(chunks) > 1


def test_chunk_source_preserved():
    """每个 chunk 保留来源文件名"""
    chunks = chunk_text("hello world " * 100, "my_doc.md")
    for c in chunks:
        assert c["source"] == "my_doc.md"


def test_chunk_empty_text():
    """空文本返回空列表"""
    assert chunk_text("", "empty.md") == []


def test_chunk_short_text():
    """短文本只产生一个 chunk"""
    chunks = chunk_text("short text", "short.md")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short text"


# ── 索引与检索测试（需要 embedding 模型，标记为 slow）────────
# 运行：pytest tests/ -m slow
# 快速测试（跳过 slow）：pytest tests/ -m "not slow"

pytestmark_slow = pytest.mark.slow


@pytest.fixture(autouse=True)
def clean_collection():
    """每个测试前清空向量库，保证测试隔离"""
    col = get_collection()
    if col.count() > 0:
        col.delete(where={"source": {"$ne": ""}})
    yield
    if col.count() > 0:
        col.delete(where={"source": {"$ne": ""}})


@pytest.mark.slow
def test_add_chunks_basic():
    """add_chunks 能正常写入并返回数量"""
    chunks = [
        {"text": "RAG is retrieval augmented generation", "source": "test.md"},
        {"text": "Vector databases store embeddings", "source": "test.md"},
    ]
    count = add_chunks(chunks)
    assert count == 2
    assert get_collection().count() == 2


@pytest.mark.slow
def test_add_chunks_dedup():
    """相同内容重复写入不会产生重复记录"""
    chunks = [{"text": "duplicate content for dedup test", "source": "test.md"}]
    add_chunks(chunks)
    add_chunks(chunks)
    assert get_collection().count() == 1


@pytest.mark.slow
def test_add_chunks_empty():
    """空列表写入返回 0"""
    assert add_chunks([]) == 0


@pytest.mark.slow
def test_retrieve_returns_results():
    """写入内容后能检索到相关结果"""
    add_chunks([
        {"text": "RAG combines retrieval with language model generation", "source": "rag.md"},
        {"text": "BM25 is a keyword-based ranking algorithm", "source": "search.md"},
    ])
    results = retrieve("retrieval augmented generation")
    assert len(results) > 0
    assert all("text" in r and "source" in r and "score" in r for r in results)


@pytest.mark.slow
def test_retrieve_score_range():
    """检索结果的相关度分数在 0~1 之间"""
    add_chunks([{"text": "vector similarity search with cosine distance", "source": "vec.md"}])
    results = retrieve("cosine similarity")
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


@pytest.mark.slow
def test_retrieve_empty_collection():
    """空知识库检索返回空列表"""
    results = retrieve("anything")
    assert results == []


@pytest.mark.slow
def test_retrieve_top_k():
    """top_k 参数限制返回数量"""
    chunks = [{"text": f"document number {i} about AI search", "source": f"doc{i}.md"} for i in range(10)]
    add_chunks(chunks)
    results = retrieve("AI search", top_k=3)
    assert len(results) <= 3
