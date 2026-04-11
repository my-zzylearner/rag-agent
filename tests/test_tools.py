"""
Agent 工具层测试：search_knowledge_base、execute_tool
"""
import os
import json

import pytest

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from rag.indexer import add_chunks, get_collection
from agent.tools import search_knowledge_base, execute_tool, TOOLS


@pytest.fixture(autouse=True)
def clean_collection():
    col = get_collection()
    if col.count() > 0:
        col.delete(where={"source": {"$ne": ""}})
    yield
    if col.count() > 0:
        col.delete(where={"source": {"$ne": ""}})


# ── TOOLS 结构校验 ────────────────────────────────────────

def test_tools_schema():
    """TOOLS 列表结构符合 OpenAI function calling 格式"""
    assert len(TOOLS) >= 2
    for tool in TOOLS:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert "required" in fn["parameters"]


def test_tools_names():
    """包含必要的两个工具"""
    names = {t["function"]["name"] for t in TOOLS}
    assert "search_knowledge_base" in names
    assert "search_web" in names


# ── search_knowledge_base 测试 ────────────────────────────

def test_search_kb_empty():
    """知识库为空时返回空结果和提示信息"""
    result = search_knowledge_base("RAG")
    assert result["results"] == []
    assert "message" in result


@pytest.mark.slow
def test_search_kb_with_data():
    """知识库有数据时能返回结果"""
    add_chunks([{"text": "RAG is retrieval augmented generation technology", "source": "rag.md"}])
    result = search_knowledge_base("retrieval augmented generation")
    assert len(result["results"]) > 0
    first = result["results"][0]
    assert "content" in first
    assert "source" in first
    assert "relevance_score" in first


@pytest.mark.slow
def test_search_kb_relevance_score():
    """相关度分数在合理范围内"""
    add_chunks([{"text": "vector database stores high dimensional embeddings", "source": "vec.md"}])
    result = search_knowledge_base("vector database embeddings")
    for r in result["results"]:
        assert 0.0 <= r["relevance_score"] <= 1.0


@pytest.mark.slow
def test_search_kb_top_k():
    """top_k 参数限制返回数量"""
    chunks = [{"text": f"AI search technology document {i}", "source": f"doc{i}.md"} for i in range(10)]
    add_chunks(chunks)
    result = search_knowledge_base("AI search", top_k=2)
    assert len(result["results"]) <= 2


# ── execute_tool 测试 ─────────────────────────────────────

@pytest.mark.slow
def test_execute_tool_kb():
    """execute_tool 调用 search_knowledge_base 返回合法 JSON"""
    add_chunks([{"text": "BM25 is a keyword ranking algorithm", "source": "bm25.md"}])
    result_str = execute_tool("search_knowledge_base", {"query": "BM25 ranking"})
    result = json.loads(result_str)
    assert "results" in result


def test_execute_tool_unknown():
    """未知工具名返回 error 字段"""
    result_str = execute_tool("unknown_tool", {})
    result = json.loads(result_str)
    assert "error" in result


def test_execute_tool_missing_query():
    """query 参数缺失时不抛异常，正常返回"""
    result_str = execute_tool("search_knowledge_base", {})
    result = json.loads(result_str)
    assert "results" in result or "message" in result
