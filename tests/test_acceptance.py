"""
验收基线测试：核心场景行为验证

目的：防止改 prompt / 换模型 / 调参数后悄悄破坏已有能力（regression 检测）。
与 test_tools.py 的区别：
  - test_tools.py 测结构（格式对不对）
  - test_acceptance.py 测行为（答案对不对、路由走不走对）

运行方式：
  venv/bin/pytest tests/test_acceptance.py -v

所有测试都用 mock 替代真实 LLM / Tavily，完全离线，无需 API Key。
"""
import json
import os
import threading
from unittest.mock import MagicMock, patch

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
# 保证 agent 能构建 candidates（mock 掉真实调用，但需要 env var 存在）
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key")
os.environ.setdefault("LLM", "bailian")


# ── 工具函数 ──────────────────────────────────────────────

def _collect_events(gen) -> list:
    """收集 generator 产生的所有事件。"""
    return list(gen)


def _get_answer(events: list) -> str:
    """从事件列表中拼接完整回答文本。"""
    return "".join(e["content"] for e in events if e["type"] == "answer_chunk")


def _get_tool_calls(events: list) -> list:
    """从事件列表中提取所有工具调用名称。"""
    return [e["tool"] for e in events if e["type"] == "tool_call"]


# ── Mock 构造器 ───────────────────────────────────────────

def _make_tool_call_response(tool_name: str, query: str, call_id: str = "call_001"):
    """构造一个返回工具调用的 LLM 响应 mock。"""
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps({"query": query})

    msg = MagicMock()
    msg.tool_calls = [tc]

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_answer_response(text: str):
    """构造一个直接回答（无工具调用）的 LLM 响应 mock。"""
    msg = MagicMock()
    msg.tool_calls = None

    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def _make_stream_chunks(text: str):
    """构造流式输出的 chunk 列表 mock。"""
    chunks = []
    for char in text:
        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content=char))]
        chunks.append(chunk)
    # 末尾空 chunk
    end_chunk = MagicMock()
    end_chunk.choices = [MagicMock(delta=MagicMock(content=None))]
    chunks.append(end_chunk)
    return chunks


# ── 场景 1：实时信息路由 ──────────────────────────────────
# 验收标准：天气类 query → agent 必须调用 search_web，不能直接回答

class TestRealtimeRouting:
    """实时信息必须走 search_web，不能直接从知识库回答。"""

    @patch("agent.agent._build_candidates")
    @patch("agent.tools.search_web")
    def test_weather_query_calls_search_web(self, mock_search_web, mock_build_candidates):
        """天气查询必须触发 search_web 工具调用。"""
        from agent.agent import run_agent

        mock_search_web.return_value = {
            "results": [{"content": "明天北京天气晴，气温15-28度", "source": "weather.com"}]
        }

        # 第一轮：LLM 决定调用 search_web
        call_resp = _make_tool_call_response("search_web", "明天北京天气")
        # 第二轮：LLM 基于结果给出回答（无工具调用）
        answer_resp = _make_answer_response("明天北京天气晴，气温15-28度")
        stream_chunks = _make_stream_chunks("明天北京天气晴，气温15-28度")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            call_resp,       # 第一轮决策
            answer_resp,     # 第二轮决策（无工具调用）
            iter(stream_chunks),  # 流式输出
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("明天北京天气怎么样", max_tool_rounds=3))
        tool_calls = _get_tool_calls(events)

        assert "search_web" in tool_calls, (
            f"天气查询应调用 search_web，实际工具调用: {tool_calls}"
        )

    @patch("agent.agent._build_candidates")
    @patch("agent.tools.search_web")
    def test_weather_query_does_not_call_kb_first(self, mock_search_web, mock_build_candidates):
        """天气查询不应先查知识库再查网络（路由应直接走 search_web）。"""
        from agent.agent import run_agent

        mock_search_web.return_value = {"results": []}

        call_resp = _make_tool_call_response("search_web", "北京天气")
        answer_resp = _make_answer_response("暂无数据")
        stream_chunks = _make_stream_chunks("暂无数据")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            call_resp,
            answer_resp,
            iter(stream_chunks),
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("北京今天天气", max_tool_rounds=3))
        tool_calls = _get_tool_calls(events)

        assert "search_knowledge_base" not in tool_calls, (
            f"天气查询不应调用知识库，实际工具调用: {tool_calls}"
        )


# ── 场景 2：知识库命中 ────────────────────────────────────
# 验收标准：RAG 原理类 query → 优先调用 search_knowledge_base

class TestKnowledgeBaseRouting:
    """技术原理类问题应优先查本地知识库。"""

    @patch("agent.agent._build_candidates")
    @patch("agent.tools.search_knowledge_base")
    def test_rag_principle_calls_kb(self, mock_search_kb, mock_build_candidates):
        """RAG 原理查询应调用 search_knowledge_base。"""
        from agent.agent import run_agent

        mock_search_kb.return_value = {
            "results": [
                {
                    "content": "RAG（检索增强生成）通过向量检索找到相关文档片段，再将其注入 LLM 上下文生成回答。",
                    "source": "rag_intro.md",
                    "relevance_score": 0.92,
                }
            ]
        }

        call_resp = _make_tool_call_response("search_knowledge_base", "RAG 原理")
        answer_resp = _make_answer_response("RAG是检索增强生成")
        stream_chunks = _make_stream_chunks("RAG是检索增强生成，通过向量检索找到相关文档。")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            call_resp,
            answer_resp,
            iter(stream_chunks),
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("什么是RAG，原理是什么", max_tool_rounds=3))
        tool_calls = _get_tool_calls(events)

        assert "search_knowledge_base" in tool_calls, (
            f"RAG原理查询应调用知识库，实际工具调用: {tool_calls}"
        )


# ── 场景 3：知识库空结果自动降级 ─────────────────────────
# 验收标准：KB 返回空 → agent 必须继续调用 search_web，不能直接回答"未找到"

class TestKbEmptyFallback:
    """知识库无结果时必须降级到 search_web，不能直接回答"未找到"。"""

    @patch("agent.agent._build_candidates")
    @patch("agent.tools.search_web")
    @patch("agent.tools.search_knowledge_base")
    def test_kb_empty_falls_back_to_web(self, mock_search_kb, mock_search_web, mock_build_candidates):
        """KB 返回空时，agent 必须调用 search_web 继续检索。"""
        from agent.agent import run_agent

        mock_search_kb.return_value = {"results": [], "message": "知识库中未找到相关内容"}
        mock_search_web.return_value = {
            "results": [{"content": "协同过滤是推荐系统的核心算法。", "source": "wiki.org"}]
        }

        # 第一轮：调用知识库
        kb_call = _make_tool_call_response("search_knowledge_base", "协同过滤算法", "call_001")
        # 第二轮：KB 空 → 调用 search_web
        web_call = _make_tool_call_response("search_web", "协同过滤算法", "call_002")
        # 第三轮：有结果 → 回答
        answer_resp = _make_answer_response("协同过滤是推荐系统核心算法")
        stream_chunks = _make_stream_chunks("协同过滤是推荐系统的核心算法。")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            kb_call,
            web_call,
            answer_resp,
            iter(stream_chunks),
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("协同过滤算法是什么", max_tool_rounds=3))
        tool_calls = _get_tool_calls(events)

        assert "search_web" in tool_calls, (
            f"KB空结果后应降级调用 search_web，实际工具调用: {tool_calls}"
        )

    @patch("agent.agent._build_candidates")
    @patch("agent.tools.search_knowledge_base")
    def test_kb_empty_no_direct_answer(self, mock_search_kb, mock_build_candidates):
        """KB 空结果时，最终回答不应包含'未找到'或'没有找到'等拒绝语。"""
        from agent.agent import run_agent

        mock_search_kb.return_value = {"results": [], "message": "知识库中未找到相关内容"}

        # LLM 直接回答（模拟 prompt 约束生效，不说"未找到"）
        answer_resp = _make_answer_response("暂无相关信息")
        stream_chunks = _make_stream_chunks("根据检索结果，这个话题目前知识库暂无覆盖。")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _make_tool_call_response("search_knowledge_base", "量子纠缠"),
            answer_resp,
            iter(stream_chunks),
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("量子纠缠是什么", max_tool_rounds=1))
        answer = _get_answer(events)

        # 验证没有出现明显的拒绝语（这类回答对用户体验很差）
        bad_phrases = ["知识库中未找到", "没有找到相关", "未找到相关内容"]
        for phrase in bad_phrases:
            assert phrase not in answer, (
                f"回答中不应直接暴露知识库拒绝语 '{phrase}'，实际回答: {answer!r}"
            )


# ── 场景 4：停止信号 ──────────────────────────────────────
# 验收标准：stop_event 设置后，agent 必须停止并 yield stopped 事件

class TestStopControl:
    """停止信号必须被 agent 响应。"""

    @patch("agent.agent._build_candidates")
    def test_stop_event_yields_stopped(self, mock_build_candidates):
        """stop_event 在第一轮前设置，agent 应立即 yield stopped。"""
        from agent.agent import run_agent

        mock_client = MagicMock()
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        stop = threading.Event()
        stop.set()  # 提前设置停止信号

        events = _collect_events(run_agent("任意问题", stop_event=stop, max_tool_rounds=3))
        event_types = [e["type"] for e in events]

        assert "stopped" in event_types, (
            f"stop_event 已设置，应产生 stopped 事件，实际事件: {event_types}"
        )
        # 不应有工具调用或回答
        assert "tool_call" not in event_types
        assert "answer" not in event_types


# ── 场景 5：流式空内容降级 ────────────────────────────────
# 验收标准：流式返回空内容时，降级为非流式补一次，最终有回答输出

class TestStreamEmptyFallback:
    """流式返回空内容时必须降级非流式，确保有输出。"""

    @patch("agent.agent._build_candidates")
    def test_stream_empty_falls_back_to_non_stream(self, mock_build_candidates):
        """流式 chunk 全为空时，降级非流式调用，最终有 answer_chunk 输出。"""
        from agent.agent import run_agent

        # 第一轮决策：直接回答，不调工具
        answer_resp = _make_answer_response("")
        answer_resp.choices[0].finish_reason = "stop"

        # 流式：所有 chunk content 为 None（模拟空返回）
        empty_chunks = []
        for _ in range(3):
            chunk = MagicMock()
            chunk.choices = [MagicMock(delta=MagicMock(content=None),
                                       finish_reason=None)]
            empty_chunks.append(chunk)

        # 非流式降级调用返回真实内容
        fallback_resp = MagicMock()
        fallback_resp.choices = [MagicMock(
            message=MagicMock(content="这是降级后的回答内容")
        )]

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            answer_resp,        # 第一轮决策（无工具调用）
            iter(empty_chunks), # 流式：空内容
            fallback_resp,      # 非流式降级
        ]
        mock_build_candidates.return_value = [(mock_client, "qwen-turbo", "bailian")]

        events = _collect_events(run_agent("测试问题", max_tool_rounds=3))
        answer = _get_answer(events)
        event_types = [e["type"] for e in events]

        assert "answer" in event_types, f"应有 answer 事件，实际: {event_types}"
        assert answer == "这是降级后的回答内容", f"降级后应有回答内容，实际: {answer!r}"
