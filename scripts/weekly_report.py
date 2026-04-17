"""
每周检索来源分布报告生成器，供 GitHub Actions 调用。

读取 Gist 中 rag_agent_retrieval_stats.json 的最近 7 天数据，
生成汇总报告并写回 Gist 的 rag_agent_weekly_report.json。

环境变量：
  GITHUB_TOKEN  — 有 gist 读写权限的 PAT（Actions Secret: GIST_GITHUB_TOKEN）
  GIST_ID       — Gist hash ID
"""
import os
import sys
import json
import urllib.request
from datetime import datetime, timezone, timedelta
from collections import defaultdict

_CST = timezone(timedelta(hours=8))
_FILENAME_STATS = "rag_agent_retrieval_stats.json"
_FILENAME_REPORT = "rag_agent_weekly_report.json"


def _gist_request(method: str, body: dict = None):
    token = os.getenv("GITHUB_TOKEN")
    gist_id = os.getenv("GIST_ID")
    if not token or not gist_id:
        print("错误：未配置 GITHUB_TOKEN 或 GIST_ID", file=sys.stderr)
        sys.exit(1)

    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"https://api.github.com/gists/{gist_id}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _fetch_stats() -> list:
    gist = _gist_request("GET")
    files = gist.get("files", {})
    if _FILENAME_STATS not in files:
        print(f"Gist 中未找到 {_FILENAME_STATS}", file=sys.stderr)
        sys.exit(1)
    return json.loads(files[_FILENAME_STATS]["content"]).get("stats", [])


def _filter_last_7_days(stats: list) -> list:
    cutoff = (datetime.now(_CST) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M")
    return [s for s in stats if s.get("time", "") >= cutoff]


def _generate_report(stats: list, week_stats: list) -> dict:
    def _summarize(rows):
        if not rows:
            return {"queries": 0, "chunks": 0, "vec_only": 0, "bm25_only": 0, "both": 0}
        chunks = sum(s.get("vec_only", 0) + s.get("bm25_only", 0) + s.get("both", 0) for s in rows)
        return {
            "queries": len(rows),
            "chunks": chunks,
            "vec_only": sum(s.get("vec_only", 0) for s in rows),
            "bm25_only": sum(s.get("bm25_only", 0) for s in rows),
            "both": sum(s.get("both", 0) for s in rows),
        }

    week = _summarize(week_stats)
    total = _summarize(stats)

    def _pct(part, whole):
        return round(part / whole * 100, 1) if whole > 0 else 0

    return {
        "generated_at": datetime.now(_CST).strftime("%Y-%m-%d %H:%M"),
        "this_week": {
            **week,
            "vec_only_pct": _pct(week["vec_only"], week["chunks"]),
            "bm25_only_pct": _pct(week["bm25_only"], week["chunks"]),
            "both_pct": _pct(week["both"], week["chunks"]),
        },
        "all_time": {
            **total,
            "vec_only_pct": _pct(total["vec_only"], total["chunks"]),
            "bm25_only_pct": _pct(total["bm25_only"], total["chunks"]),
            "both_pct": _pct(total["both"], total["chunks"]),
        },
    }


def _save_report(report: dict):
    _gist_request("PATCH", {
        "files": {
            _FILENAME_REPORT: {"content": json.dumps(report, ensure_ascii=False, indent=2)}
        }
    })


def _print_report(report: dict):
    w = report["this_week"]
    a = report["all_time"]
    print(f"\n{'='*50}")
    print(f"  Weekly Retrieval Report  {report['generated_at']}")
    print(f"{'='*50}")
    print(f"\n本周（近 7 天）：{w['queries']} 次查询 / {w['chunks']} 个 chunk")
    print(f"  仅向量命中：{w['vec_only']:4d}  ({w['vec_only_pct']}%)")
    print(f"  仅BM25命中：{w['bm25_only']:4d}  ({w['bm25_only_pct']}%)")
    print(f"  两路都命中：{w['both']:4d}  ({w['both_pct']}%)")
    print(f"\n累计：{a['queries']} 次查询 / {a['chunks']} 个 chunk")
    print(f"  仅向量命中：{a['vec_only']:4d}  ({a['vec_only_pct']}%)")
    print(f"  仅BM25命中：{a['bm25_only']:4d}  ({a['bm25_only_pct']}%)")
    print(f"  两路都命中：{a['both']:4d}  ({a['both_pct']}%)")
    print()


def main():
    stats = _fetch_stats()
    week_stats = _filter_last_7_days(stats)
    report = _generate_report(stats, week_stats)
    _print_report(report)
    _save_report(report)
    print(f"报告已写入 Gist: {_FILENAME_REPORT}")


if __name__ == "__main__":
    main()
