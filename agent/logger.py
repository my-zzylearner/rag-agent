"""
统一日志模块。

环境变量：
  DEBUG=true   输出详细调试日志（tool 参数、模型响应、报错堆栈）
  DEBUG=false  只输出 WARNING 及以上（默认）

日志格式：JSON 单行，方便 Streamlit Cloud Logs 面板过滤。
"""
import os
import json
import logging
import traceback
from datetime import datetime, timezone


def _setup() -> logging.Logger:
    # ERROR 始终开启；DEBUG=true 时输出所有级别
    level = logging.DEBUG if os.getenv("DEBUG", "").lower() == "true" else logging.ERROR
    logger = logging.getLogger("rag_agent")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


_logger = _setup()


def _emit(level: str, trace_id: str, event: str, **kwargs):
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": level,
        "trace_id": trace_id,
        "event": event,
        **kwargs,
    }
    line = json.dumps(record, ensure_ascii=False)
    if level == "DEBUG":
        _logger.debug(line)
    elif level == "WARNING":
        _logger.warning(line)
    elif level == "ERROR":
        _logger.error(line)


def debug(trace_id: str, event: str, **kwargs):
    _emit("DEBUG", trace_id, event, **kwargs)


def warning(trace_id: str, event: str, **kwargs):
    _emit("WARNING", trace_id, event, **kwargs)


def error(trace_id: str, event: str, exc: Exception = None, **kwargs):
    if exc is not None:
        kwargs["error_type"] = type(exc).__name__
        kwargs["error_msg"] = str(exc)
        if os.getenv("DEBUG", "").lower() == "true":
            kwargs["traceback"] = traceback.format_exc()
    _emit("ERROR", trace_id, event, **kwargs)
