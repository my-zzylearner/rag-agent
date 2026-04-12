"""
Agent 主循环：OpenAI 兼容接口 + Tool Calling
支持多 LLM 提供商，通过 LLM_PROVIDER 环境变量切换。
"""
import os
import json
import threading
import uuid
from typing import Generator, Dict, Any, Optional
from openai import OpenAI

from .tools import TOOLS, execute_tool
from .logger import debug, warning, error as log_error

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
- 引用检索内容时在末尾注明来源（只引用工具实际返回的内容，禁止编造来源或链接）
- 禁止生成任何未经工具返回的 URL、链接或参考资料
- 用中文回答
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


def _build_candidates() -> list:
    """
    解析 LLM / LLM_FALLBACK 环境变量，返回有序候选列表。
    每项为 (OpenAI client, model_name, label) 元组。
    LLM=bailian/qwen-plus
    LLM_FALLBACK=bailian/qwen-turbo,qianfan/ernie-speed-pro-128k
    """
    entries = [os.getenv("LLM", "bailian")]
    fallback = os.getenv("LLM_FALLBACK", "")
    if fallback:
        entries += [s.strip() for s in fallback.split(",") if s.strip()]

    candidates = []
    for entry in entries:
        parts = entry.split("/", 1)
        provider = parts[0].lower()
        if provider not in PROVIDERS:
            raise ValueError(
                f"不支持的 LLM 提供商: '{provider}'，可选值为: {', '.join(PROVIDERS)}"
            )
        cfg = PROVIDERS[provider]
        model = parts[1] if len(parts) == 2 else cfg["model"]
        api_key = os.getenv(cfg["env_var"])
        if not api_key:
            raise ValueError(f"{cfg['env_var']} 未配置，请检查 .env 文件或 Streamlit Secrets")
        client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
        candidates.append((client, model, entry))

    return candidates


def _should_fallback(exc: Exception) -> bool:
    """判断异常是否应触发模型降级（额度不足、模型不存在、限流等）。"""
    msg = str(exc).lower()
    fallback_keywords = (
        # 额度 / 计费
        "quota", "insufficient", "billing", "balance", "credit", "limit exceeded",
        # 限流
        "rate limit", "429", "too many requests",
        # 模型不存在 / 无权限
        "model_not_found", "model not found", "does not exist", "invalid_request_error",
        "no permission", "not have access", "404",
    )
    return any(k in msg for k in fallback_keywords)


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
    - {"type": "answer_chunk", "content": "..."}  ← 流式片段
    - {"type": "answer", "content": ""}           ← 流结束标记
    - {"type": "retry", "from": "...", "to": "...", "reason": "..."}  ← 模型切换通知
    - {"type": "error", "content": "..."}
    - {"type": "stopped"}  ← 用户主动停止时
    """
    trace_id = uuid.uuid4().hex[:12]

    try:
        candidates = _build_candidates()
    except ValueError as e:
        log_error(trace_id, "build_candidates_failed", exc=e)
        yield {"type": "error", "content": str(e), "trace_id": trace_id}
        return

    from datetime import datetime
    today = datetime.now().strftime("%Y年%m月%d日")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\n当前日期：{today}"},
        {"role": "user", "content": user_query},
    ]

    # 选出当前使用的模型，遇到额度错误时切换到下一个
    candidate_idx = 0
    client, model, label = candidates[candidate_idx]
    debug(trace_id, "agent_start", query=user_query, model=label, max_rounds=max_tool_rounds)

    def _next_candidate(exc: Exception):
        """切换到下一个候选模型，返回 retry 事件或 None（已无候选）。"""
        nonlocal candidate_idx, client, model, label
        if not _should_fallback(exc) or candidate_idx + 1 >= len(candidates):
            return None
        from_label = label
        candidate_idx += 1
        client, model, label = candidates[candidate_idx]
        warning(trace_id, "model_fallback", from_model=from_label, to_model=label,
                reason=str(exc)[:200])
        msg_lower = str(exc).lower()
        if any(k in msg_lower for k in ("model_not_found", "does not exist", "not have access", "404", "invalid_request_error")):
            reason_label = "模型不存在或无权限"
        elif any(k in msg_lower for k in ("quota", "insufficient", "billing", "balance", "credit", "limit exceeded")):
            reason_label = "额度不足"
        elif any(k in msg_lower for k in ("rate limit", "429", "too many requests")):
            reason_label = "请求限流"
        else:
            reason_label = "调用失败"
        return {
            "type": "retry",
            "from": from_label,
            "to": label,
            "reason_label": reason_label,
            "reason": str(exc)[:120],
        }

    for round_num in range(max_tool_rounds + 1):
        if stop_event and stop_event.is_set():
            debug(trace_id, "agent_stopped", round=round_num)
            yield {"type": "stopped"}
            return

        # ── 第一步：非流式调用，决策是否需要 tool calling ──────────
        debug(trace_id, "llm_call", round=round_num, model=label)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
            )
        except Exception as e:
            log_error(trace_id, "llm_call_failed", exc=e, round=round_num, model=label, query=user_query)
            retry_event = _next_candidate(e)
            if retry_event:
                yield retry_event
                continue  # 用新模型重跑本轮
            yield {"type": "error", "content": f"调用 LLM API 失败: {str(e)}", "trace_id": trace_id}
            return

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls

        # ── 第二步：无 tool calls，直接流式输出答案 ────────────────
        if not tool_calls:
            debug(trace_id, "stream_answer", round=round_num, model=label)
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    timeout=60,
                )
                for chunk in stream:
                    if stop_event and stop_event.is_set():
                        debug(trace_id, "agent_stopped", round=round_num)
                        yield {"type": "stopped"}
                        return
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield {"type": "answer_chunk", "content": delta}
                debug(trace_id, "agent_done", round=round_num, model=label)
                yield {"type": "answer", "content": ""}
            except Exception as e:
                log_error(trace_id, "stream_failed", exc=e, round=round_num, model=label, query=user_query)
                retry_event = _next_candidate(e)
                if retry_event:
                    yield retry_event
                    continue
                yield {"type": "error", "content": f"调用 LLM API 失败: {str(e)}", "trace_id": trace_id}
            return

        # ── 第三步：执行 tool calls ────────────────────────────────
        messages.append(msg)

        for tc in tool_calls:
            tool_name = tc.function.name
            try:
                tool_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            debug(trace_id, "tool_call", round=round_num, tool=tool_name,
                  query=tool_args.get("query", ""))
            yield {
                "type": "tool_call",
                "tool": tool_name,
                "query": tool_args.get("query", ""),
            }

            tool_result_str = execute_tool(tool_name, tool_args, top_k=top_k, llm_client=client, llm_model=model)
            tool_result = json.loads(tool_result_str)

            result_count = len(tool_result.get("results", []))
            debug(trace_id, "tool_result", round=round_num, tool=tool_name, result_count=result_count)
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

        # ── 第四步：达到最大轮次，强制输出最终答案 ────────────────
        if round_num >= max_tool_rounds:
            debug(trace_id, "stream_final", round=round_num, model=label)
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    timeout=60,
                )
                for chunk in stream:
                    if stop_event and stop_event.is_set():
                        debug(trace_id, "agent_stopped", round=round_num)
                        yield {"type": "stopped"}
                        return
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield {"type": "answer_chunk", "content": delta}
                debug(trace_id, "agent_done", round=round_num, model=label)
                yield {"type": "answer", "content": ""}
            except Exception as e:
                log_error(trace_id, "stream_final_failed", exc=e, round=round_num, model=label, query=user_query)
                retry_event = _next_candidate(e)
                if retry_event:
                    yield retry_event
                    try:
                        stream = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            stream=True,
                            timeout=60,
                        )
                        for chunk in stream:
                            if stop_event and stop_event.is_set():
                                yield {"type": "stopped"}
                                return
                            delta = chunk.choices[0].delta.content
                            if delta:
                                yield {"type": "answer_chunk", "content": delta}
                        yield {"type": "answer", "content": ""}
                    except Exception as e2:
                        log_error(trace_id, "stream_final_retry_failed", exc=e2, round=round_num, model=label, query=user_query)
                        yield {"type": "error", "content": f"调用 LLM API 失败: {str(e2)}", "trace_id": trace_id}
                else:
                    yield {"type": "error", "content": f"调用 LLM API 失败: {str(e)}", "trace_id": trace_id}
            return
