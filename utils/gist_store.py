"""
GitHub Gist 持久化存储，用于访问统计、留言板和知识内化审计。

环境变量：
  GITHUB_TOKEN  — 有 gist 权限的 Personal Access Token
  GIST_ID       — Gist 的 hash ID

Gist 文件：
  rag_agent_stats.json        — 访问/查询计数 + 留言板
    初始内容：{"visits": 0, "queries": 0, "feedback": []}
  rag_agent_internalized.json — 知识内化审计记录（独立锁，互不阻塞）
    初始内容：{"internalized": []}

未配置时所有操作静默跳过，不影响主功能。
"""
import os
import json
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from typing import Optional

_CST = timezone(timedelta(hours=8))

_FILENAME = "rag_agent_stats.json"
_FILENAME_INTERNALIZED = "rag_agent_internalized.json"
_lock = threading.Lock()
_lock_internalized = threading.Lock()


def _enabled() -> bool:
    return bool(os.getenv("GITHUB_TOKEN")) and bool(os.getenv("GIST_ID"))


def _load_file(filename: str) -> Optional[dict]:
    """从 Gist 读取指定文件，失败返回 None。"""
    if not _enabled():
        return None
    token = os.getenv("GITHUB_TOKEN")
    gist_id = os.getenv("GIST_ID")
    try:
        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            gist = json.loads(resp.read())
        raw = gist["files"][filename]["content"]
        return json.loads(raw)
    except Exception:
        return None


def _save_file_sync(filename: str, data: dict) -> None:
    """同步写入 Gist 指定文件，在后台线程调用。"""
    if not _enabled():
        return
    token = os.getenv("GITHUB_TOKEN")
    gist_id = os.getenv("GIST_ID")
    try:
        body = json.dumps({
            "files": {filename: {"content": json.dumps(data, ensure_ascii=False)}}
        }).encode()
        req = urllib.request.Request(
            f"https://api.github.com/gists/{gist_id}",
            data=body,
            method="PATCH",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def load() -> Optional[dict]:
    """从 Gist 读取统计数据，失败返回 None。"""
    return _load_file(_FILENAME)


def _save_sync(data: dict) -> None:
    """同步写入 Gist stats 文件，在后台线程调用。"""
    _save_file_sync(_FILENAME, data)


def save(data: dict) -> None:
    """异步写入 Gist，不阻塞 UI。"""
    threading.Thread(target=_save_sync, args=(data,), daemon=True).start()


def increment(field: str) -> None:
    """原子性地将某个计数字段 +1 并异步写回。load 失败时跳过，不用空数据覆盖。"""
    if not _enabled():
        return
    with _lock:
        data = load()
        if data is None:
            return  # 读取失败，跳过，避免空数据覆盖 feedback
        data[field] = data.get(field, 0) + 1
        save(data)


def add_internalized(query: str, filename: str, preview: str) -> None:
    """追加一条内化记录到独立文件，异步写回，最多保留 20 条。"""
    if not _enabled():
        return
    with _lock_internalized:
        data = _load_file(_FILENAME_INTERNALIZED)
        if data is None:
            return
        data.setdefault("internalized", []).append({
            "query": query[:80],
            "file": filename,
            "preview": preview[:200],
            "time": datetime.now(_CST).strftime("%Y-%m-%d %H:%M"),
        })
        data["internalized"] = data["internalized"][-20:]
        threading.Thread(
            target=_save_file_sync,
            args=(_FILENAME_INTERNALIZED, data),
            daemon=True,
        ).start()


def add_feedback(content: str) -> None:
    """追加一条留言并异步写回。load 失败时跳过，不用空数据覆盖。"""
    if not _enabled():
        return
    with _lock:
        data = load()
        if data is None:
            return  # 读取失败，跳过，避免空数据覆盖已有留言
        data.setdefault("feedback", []).append({
            "content": content,
            "time": datetime.now(_CST).strftime("%Y-%m-%d %H:%M"),
        })
        # 只保留最新 50 条
        data["feedback"] = data["feedback"][-50:]
        save(data)
