#!/usr/bin/env python3
"""
本地 AI 搜索链路测试脚本（CLI）。

功能：
- 单问题测试：打印工具调用、检索结果统计、最终答案
- 批量测试：从文本文件逐行读取问题并执行
- 可选择仅知识库 / 仅网络工具，便于路由验证
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv


def _load_env() -> None:
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def _read_queries(args) -> List[str]:
    if args.query:
        return [args.query.strip()]

    if args.file:
        fp = Path(args.file)
        if not fp.is_absolute():
            fp = Path.cwd() / fp
        if not fp.exists():
            raise FileNotFoundError(f"问题文件不存在: {fp}")
        lines = [ln.strip() for ln in fp.read_text(encoding="utf-8").splitlines()]
        queries = [ln for ln in lines if ln and not ln.startswith("#")]
        if not queries:
            raise ValueError("问题文件为空（或全是注释）")
        return queries

    raise ValueError("请通过 --query 或 --file 提供测试问题")


def _tool_mode_to_allowed_tools(mode: str):
    if mode == "kb-only":
        return ["search_knowledge_base"]
    if mode == "web-only":
        return ["search_web"]
    return None


def _run_one_query(query: str, top_k: int, max_rounds: int, tool_mode: str) -> bool:
    from agent.agent import run_agent

    allowed_tools = _tool_mode_to_allowed_tools(tool_mode)
    final_answer_chunks: List[str] = []
    source_urls: List[str] = []
    had_error = False

    print("\n" + "=" * 80)
    print(f"Q: {query}")
    print("-" * 80)

    for event in run_agent(
        query,
        top_k=top_k,
        max_tool_rounds=max_rounds,
        allowed_tools=allowed_tools,
    ):
        et = event.get("type")

        if et == "tool_call":
            print(f"[tool_call] {event.get('tool')} -> {event.get('query', '')}")
        elif et == "tool_result":
            result = event.get("result", {}) or {}
            results = result.get("results", []) or []
            msg = result.get("message", "")
            if msg:
                print(f"[tool_result] count={len(results)} msg={msg}")
            else:
                print(f"[tool_result] count={len(results)}")
            for r in results:
                src = r.get("source", "")
                if src:
                    source_urls.append(src)
        elif et == "retry":
            print(f"[retry] {event.get('from')} -> {event.get('to')} ({event.get('reason_label', '调用失败')})")
        elif et == "answer_chunk":
            final_answer_chunks.append(event.get("content", ""))
        elif et == "stopped":
            print("[stopped] 被外部停止")
        elif et == "error":
            had_error = True
            print(f"[error] {event.get('content', '')}")
            trace_id = event.get("trace_id")
            if trace_id:
                print(f"[trace_id] {trace_id}")

    answer = "".join(final_answer_chunks).strip()
    print("-" * 80)
    if answer:
        print(answer)
    else:
        print("(无回答内容)")

    if source_urls:
        uniq = []
        seen = set()
        for s in source_urls:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        print("-" * 80)
        print("Sources:")
        for idx, src in enumerate(uniq, 1):
            print(f"  [{idx}] {src}")

    print("=" * 80)
    return not had_error


def main() -> int:
    parser = argparse.ArgumentParser(description="本地 AI 搜索测试脚本")
    parser.add_argument("--query", type=str, help="单条测试问题")
    parser.add_argument("--file", type=str, help="批量问题文件（每行一条，# 开头为注释）")
    parser.add_argument("--top-k", type=int, default=4, help="知识库检索条数")
    parser.add_argument("--max-rounds", type=int, default=4, help="最大工具调用轮次")
    parser.add_argument(
        "--tool-mode",
        type=str,
        default="auto",
        choices=["auto", "kb-only", "web-only"],
        help="工具模式：auto 自动路由，kb-only 仅知识库，web-only 仅网络",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    _load_env()

    # 运行前快速检查关键配置
    if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("QIANFAN_API_KEY"):
        print("缺少 LLM API Key：请配置 DASHSCOPE_API_KEY 或 QIANFAN_API_KEY", file=sys.stderr)
        return 2
    if args.tool_mode != "kb-only" and not os.getenv("TAVILY_API_KEY"):
        print("警告：未配置 TAVILY_API_KEY，网络搜索工具会返回空结果", file=sys.stderr)

    try:
        queries = _read_queries(args)
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 2

    ok = True
    for q in queries:
        ok = _run_one_query(
            q,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            tool_mode=args.tool_mode,
        ) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

