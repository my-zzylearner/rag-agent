"""
统一日志模块。

环境变量：
  DEBUG=true   输出 DEBUG 及以上级别
  DEBUG=false  只输出 WARNING 及以上（默认）

日志格式：时间 | 级别 | 模块名 | 消息
  - 写入 logs/app.log（按日期轮转，保留 7 天，单文件最大 5MB）
  - 同时输出到 stderr（WARNING 及以上）
"""
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler

# 项目根目录（本文件位于 utils/，上一级即为项目根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOGS_DIR = os.path.join(_PROJECT_ROOT, "logs")
_LOG_FILE = os.path.join(_LOGS_DIR, "app.log")

_CST = timezone(timedelta(hours=8))


class _CSTFormatter(logging.Formatter):
    """输出北京时间（UTC+8）的日志格式化器。"""
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=_CST)
        return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


_FORMATTER = _CSTFormatter(
    fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 是否已完成全局 handler 初始化
_initialized = False


def _init_root_handlers() -> None:
    """在根 logger 上挂载文件 handler 和 stderr handler（只执行一次）。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    debug_mode = os.getenv("DEBUG", "").lower() == "true"
    root_level = logging.DEBUG if debug_mode else logging.WARNING

    root = logging.getLogger()
    # 避免重复添加（多次 import 或 reload 时）
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    root.setLevel(root_level)

    # ── 文件 handler：DEBUG 及以上全记，按大小轮转 ──────────
    os.makedirs(_LOGS_DIR, exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_FORMATTER)
    root.addHandler(file_handler)

    # ── stderr handler：WARNING 及以上 ────────────────────────
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(_FORMATTER)
    root.addHandler(stderr_handler)


def get_logger(name: str) -> logging.Logger:
    """
    返回以 name 命名的 logger。
    首次调用时完成全局 handler 初始化。
    """
    _init_root_handlers()
    return logging.getLogger(name)
