"""
GitHub Gist 持久化存储，用于访问统计和留言板。

环境变量：
  GITHUB_TOKEN  — 有 gist 权限的 Personal Access Token
  GIST_ID       — Gist 的 hash ID

Gist 文件名：rag_agent_stats.json
初始内容：{"visits": 0, "queries": 0, "feedback": []}

未配置时所有操作静默跳过，不影响主功能。
"""
import os
import json
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional

_FILENAME = "rag_agent_stats.json"
_lock = threading.Lock()


def _enabled() -> bool:
    return bool(os.getenv("GITHUB_TOKEN")) and bool(os.getenv("GIST_ID"))


def load() -> Optional[dict]:
    """从 Gist 读取统计数据，失败返回 None。"""
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
        raw = gist["files"][_FILENAME]["content"]
        return json.loads(raw)
    except Exception:
        return None


def _save_sync(data: dict) -> None:
    """同步写入 Gist，在后台线程调用。"""
    if not _enabled():
        return
    token = os.getenv("GITHUB_TOKEN")
    gist_id = os.getenv("GIST_ID")
    try:
        body = json.dumps({
            "files": {_FILENAME: {"content": json.dumps(data, ensure_ascii=False)}}
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


def save(data: dict) -> None:
    """异步写入 Gist，不阻塞 UI。"""
    threading.Thread(target=_save_sync, args=(data,), daemon=True).start()


def increment(field: str) -> None:
    """原子性地将某个计数字段 +1 并异步写回。"""
    if not _enabled():
        return
    with _lock:
        data = load() or {"visits": 0, "queries": 0, "feedback": []}
        data[field] = data.get(field, 0) + 1
        save(data)


def add_feedback(content: str) -> None:
    """追加一条留言并异步写回。"""
    if not _enabled():
        return
    with _lock:
        data = load() or {"visits": 0, "queries": 0, "feedback": []}
        data.setdefault("feedback", []).append({
            "content": content,
            "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        })
        # 只保留最新 50 条
        data["feedback"] = data["feedback"][-50:]
        save(data)
