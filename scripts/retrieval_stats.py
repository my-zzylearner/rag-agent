"""
检索来源分布统计分析

从 GitHub Gist 拉取 rag_agent_retrieval_stats.json，输出：
- 总体 vec_only / bm25_only / both 占比
- 最近 N 条明细
- 按 top_k 分组统计

用法：
  python scripts/retrieval_stats.py              # 汇总 + 最近 10 条
  python scripts/retrieval_stats.py --tail 20    # 最近 20 条明细
  python scripts/retrieval_stats.py --summary    # 只看汇总，不展示明细

依赖：GITHUB_TOKEN 和 GIST_ID 环境变量（与主应用相同）
"""
import os
import sys
import json
import argparse
import urllib.request
from collections import defaultdict
from dotenv import load_dotenv

# 加载 .env（本地开发）
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)

_FILENAME = "rag_agent_retrieval_stats.json"


def _fetch() -> list:
    token = os.getenv("GITHUB_TOKEN")
    gist_id = os.getenv("GIST_ID")
    if not token or not gist_id:
        print("错误：未配置 GITHUB_TOKEN 或 GIST_ID 环境变量", file=sys.stderr)
        sys.exit(1)

    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            gist = json.loads(resp.read())
    except Exception as e:
        print(f"错误：Gist 读取失败 — {e}", file=sys.stderr)
        sys.exit(1)

    files = gist.get("files", {})
    if _FILENAME not in files:
        print(f"错误：Gist 中未找到 {_FILENAME}，请先创建该文件（初始内容：{{\"stats\": []}}）", file=sys.stderr)
        sys.exit(1)

    raw = files[_FILENAME]["content"]
    data = json.loads(raw)
    return data.get("stats", [])


def _bar(ratio: float, width: int = 20) -> str:
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _print_summary(stats: list) -> None:
    total = len(stats)
    if total == 0:
        print("暂无数据")
        return

    vec_only = sum(1 for s in stats if s.get("vec_only", 0) > 0 and s.get("bm25_only", 0) == 0 and s.get("both", 0) == 0)
    bm25_only = sum(1 for s in stats if s.get("bm25_only", 0) > 0 and s.get("vec_only", 0) == 0 and s.get("both", 0) == 0)
    both = sum(1 for s in stats if s.get("both", 0) > 0)
    mixed = total - vec_only - bm25_only - both  # 混合但没有 both 的情况

    # chunk 级别汇总（每条 query 内部的 chunk 来源）
    total_chunks = sum(s.get("vec_only", 0) + s.get("bm25_only", 0) + s.get("both", 0) for s in stats)
    chunk_vec = sum(s.get("vec_only", 0) for s in stats)
    chunk_bm25 = sum(s.get("bm25_only", 0) for s in stats)
    chunk_both = sum(s.get("both", 0) for s in stats)

    print(f"\n{'='*50}")
    print(f"  检索来源分布统计  （共 {total} 次查询）")
    print(f"{'='*50}")

    print("\n【Chunk 级别来源分布】（每个召回 chunk 的来源）")
    if total_chunks > 0:
        for label, count in [("仅向量命中", chunk_vec), ("仅BM25命中", chunk_bm25), ("两路都命中", chunk_both)]:
            pct = count / total_chunks
            print(f"  {label}：{count:4d} / {total_chunks}  {_bar(pct)}  {pct*100:.1f}%")

    print("\n【按 top_k 分组】")
    by_topk = defaultdict(lambda: {"vec": 0, "bm25": 0, "both": 0, "total": 0})
    for s in stats:
        k = s.get("top_k", "?")
        by_topk[k]["vec"] += s.get("vec_only", 0)
        by_topk[k]["bm25"] += s.get("bm25_only", 0)
        by_topk[k]["both"] += s.get("both", 0)
        by_topk[k]["total"] += 1
    for k in sorted(by_topk):
        g = by_topk[k]
        t = g["vec"] + g["bm25"] + g["both"]
        if t == 0:
            continue
        print(f"  top_k={k}（{g['total']}次）: 向量 {g['vec']/t*100:.0f}% / BM25 {g['bm25']/t*100:.0f}% / 双路 {g['both']/t*100:.0f}%")

    if stats:
        print(f"\n  时间范围：{stats[0]['time']} → {stats[-1]['time']}")
    print()


def _print_tail(stats: list, n: int) -> None:
    recent = stats[-n:]
    print(f"\n【最近 {len(recent)} 条明细】")
    print(f"  {'时间':<16} {'top_k':>5}  {'向量':>4} {'BM25':>4} {'双路':>4}  查询")
    print(f"  {'-'*16} {'-----':>5}  {'----':>4} {'----':>4} {'----':>4}  ----")
    for s in reversed(recent):
        q = s.get("query", "")[:30]
        print(f"  {s.get('time',''):<16} {s.get('top_k',0):>5}  {s.get('vec_only',0):>4} {s.get('bm25_only',0):>4} {s.get('both',0):>4}  {q}")
    print()


def main():
    parser = argparse.ArgumentParser(description="检索来源分布统计")
    parser.add_argument("--tail", type=int, default=10, help="展示最近 N 条明细（默认 10）")
    parser.add_argument("--summary", action="store_true", help="只看汇总，不展示明细")
    args = parser.parse_args()

    stats = _fetch()
    _print_summary(stats)
    if not args.summary:
        _print_tail(stats, args.tail)


if __name__ == "__main__":
    main()
