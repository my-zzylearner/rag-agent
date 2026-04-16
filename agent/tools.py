"""
Agent 可调用的工具定义
"""
import os
import json
from typing import Dict
from tavily import TavilyClient

from rag.retriever import retrieve
from rag.indexer import add_chunks
from utils.logger import get_logger

_logger = get_logger(__name__)

# 工具描述（OpenAI tool_call 格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从本地知识库中检索与问题相关的内容。"
                "仅适用于：AI技术、RAG、向量数据库、搜索算法等专业原理性问题。"
                "不适用于：天气、新闻、股价等实时信息，这类请使用 search_web。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用于检索的查询语句，尽量简洁精准",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "从互联网搜索最新信息。"
                "适用于：天气、新闻、股价、实时数据、知识库未覆盖的内容。"
                "当 search_knowledge_base 返回空结果时，必须调用此工具继续搜索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def search_knowledge_base(query: str, top_k: int = 4) -> Dict:
    """调用 RAG 检索，返回结构化结果"""
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception as e:
        _logger.error("search_knowledge_base failed: query=%r error=%s error_type=%s",
                      query, e, type(e).__name__, exc_info=True)
        return {"results": [], "message": "知识库检索异常，建议调用 search_web 继续搜索"}

    if not chunks:
        return {"results": [], "message": "知识库中未找到相关内容，建议调用 search_web 继续搜索"}

    return {
        "results": [
            {
                "content": c["text"],
                "source": c["source"],
                "relevance_score": c["score"],
            }
            for c in chunks
        ]
    }


def search_web(query: str, llm_client=None, llm_model: str = "") -> Dict:
    """调用 Tavily 网络搜索"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {"results": [], "message": "TAVILY_API_KEY 未配置"}

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=4)
        results = [
            {
                "content": r.get("content", ""),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
            }
            for r in response.get("results", [])
        ]
        _logger.info("search_web succeeded: query=%r result_count=%d", query, len(results))

        # 自动内化：将搜索结果写入知识库，实时性强的内容不内化
        _REALTIME_KEYWORDS = ("天气", "weather", "气温", "新闻", "stock", "股价", "今日", "今天", "明天", "实时")
        is_realtime = any(kw in query.lower() for kw in _REALTIME_KEYWORDS)
        if not is_realtime:
            try:
                chunks = [
                    {"text": r["content"], "source": r["source"]}
                    for r in results if len(r["content"]) >= 50
                ]
                add_chunks(chunks)
            except Exception:
                pass  # 内化失败不影响主流程

        # 异步知识内化（提炼后写入 data/docs/）
        if llm_client and results:
            import threading
            from rag.knowledge_internalizer import internalize_async
            _logger.info("internalize_thread_start: query=%r result_count=%d", query, len(results))
            t = threading.Thread(
                target=internalize_async,
                args=(query, results, llm_client, llm_model),
                daemon=True,
            )
            t.start()

        return {"results": results}
    except Exception as e:
        _logger.error("search_web failed: query=%r error=%s", query, e)
        return {"results": [], "message": f"搜索失败: {str(e)}"}


def execute_tool(tool_name: str, tool_args: dict, top_k: int = 4, llm_client=None, llm_model: str = "") -> str:
    """根据工具名执行对应函数，返回 JSON 字符串。任何异常都兜住，不让 agent loop 崩溃。"""
    try:
        if tool_name == "search_knowledge_base":
            result = search_knowledge_base(tool_args.get("query", ""), top_k=top_k)
        elif tool_name == "search_web":
            result = search_web(tool_args.get("query", ""), llm_client=llm_client, llm_model=llm_model)
        else:
            result = {"error": f"未知工具: {tool_name}"}
    except Exception as e:
        _logger.error("execute_tool unexpected error: tool=%s error=%s", tool_name, e)
        result = {"results": [], "message": f"工具执行异常: {str(e)}，建议换用其他工具继续"}

    return json.dumps(result, ensure_ascii=False)
