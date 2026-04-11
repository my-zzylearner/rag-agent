"""
Agent 主循环：OpenAI 兼容接口 + Tool Calling
支持多 LLM 提供商，通过 LLM_PROVIDER 环境变量切换。
"""
import os
import json
import threading
from typing import Generator, Dict, Any, Optional
from openai import OpenAI

from .tools import TOOLS, execute_tool

MAX_TOOL_ROUNDS = 3  # 最多调用工具轮次，防止死循环

SYSTEM_PROMPT = """你是一个专业的 AI 搜索助手，可以回答各类问题。

你有两个工具：
1. search_knowledge_base：检索本地专业知识库（AI/ML/搜索技术原理）
2. search_web：搜索互联网获取实时信息

工具路由规则（严格遵守）：
- 实时信息类（天气、新闻、股价、今日日期、最新动态等）→ 直接调用 search_web，禁止先查知识库
- AI/ML/搜索技术原理类 → 优先调用 search_knowledge_base
- search_knowledge_base 返回空结果或相关度低时 → 必须继续调用 search_web，不能直接回答"未找到"
- 不确定属于哪类时 → 先查知识库，无结果再查网络

回答规范：
- 回答要准确、有条理
- 引用检索内容时在末尾注明来源
- 用中文回答
"""


def _get_client():
    """
    根据环境变量 LLM 选择提供商和模型，格式：provider/model_name
    支持：qianfan（默认）、bailian
    示例：LLM=bailian/qvq-max-2025-03-25
    返回 (OpenAI client, model_name) 元组。
    """
    PROVIDERS = {
        "qianfan": {
            "env_var": "QIANFAN_API_KEY",
            "base_url": "https://qianfan.baidubce.com/v2",
            "model": "ernie-speed-pro-128k",
        },
        "bailian": {
            "env_var": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.5-flash",
        },
    }

    llm = os.getenv("LLM", "qianfan")
    parts = llm.split("/", 1)
    provider = parts[0].lower()
    if provider not in PROVIDERS:
        raise ValueError(
            f"不支持的 LLM 提供商: '{provider}'，可选值为: {', '.join(PROVIDERS)}"
        )

    cfg = PROVIDERS[provider]
    model = parts[1] if len(parts) == 2 else cfg["model"]

    api_key = os.getenv(cfg["env_var"])
    if not api_key:
        raise ValueError(f"{cfg['env_var']} 未配置，请检查 .env 文件")

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    return client, model


def run_agent(
    user_query: str,
    stop_event: Optional[threading.Event] = None,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
    top_k: int = 4,
) -> Generator[Dict[str, Any], None, None]:
    """
    Agent 主循环，使用 generator 流式返回事件：
    - {"type": "tool_call", "tool": "...", "query": "..."}
    - {"type": "tool_result", "tool": "...", "result": {...}}
    - {"type": "answer", "content": "..."}
    - {"type": "error", "content": "..."}
    - {"type": "stopped"}  ← 用户主动停止时
    """
    try:
        client, model = _get_client()
    except ValueError as e:
        yield {"type": "error", "content": str(e)}
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for round_num in range(max_tool_rounds + 1):
        if stop_event and stop_event.is_set():
            yield {"type": "stopped"}
            return

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as e:
            yield {"type": "error", "content": f"调用 LLM API 失败: {str(e)}"}
            return

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls

        # 没有工具调用，直接输出最终回答
        if not tool_calls:
            yield {"type": "answer", "content": msg.content or ""}
            return

        # 将 assistant 消息加入历史
        messages.append(msg)

        # 逐个执行工具调用
        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            yield {
                "type": "tool_call",
                "tool": tool_name,
                "query": tool_args.get("query", ""),
            }

            tool_result_str = execute_tool(tool_name, tool_args, top_k=top_k, llm_client=client, llm_model=model)
            tool_result = json.loads(tool_result_str)

            yield {
                "type": "tool_result",
                "tool": tool_name,
                "result": tool_result,
            }

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result_str,
            })

        # 达到最大轮次，禁用工具强制 LLM 基于已有结果生成最终回答
        if round_num >= max_tool_rounds:
            try:
                final_resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                yield {"type": "answer", "content": final_resp.choices[0].message.content or ""}
            except Exception as e:
                yield {"type": "error", "content": f"调用 LLM API 失败: {str(e)}"}
            return
