"""
统一日志模块。

环境变量：
  DEBUG=true   输出详细调试日志（tool 参数、模型切换、报错堆栈）
  DEBUG=false  只输出 ERROR（默认）

日志格式：JSON 单行，方便 Streamlit Cloud Logs 面板过滤。
"""
import os
import json
import traceback
from datetime import datetime, timezone


def _emit(level: str, trace_id: str, event: str, **kwargs):
    record = {
        "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "level": level,
        "trace_id": trace_id,
        "event": event,
        **kwargs,
    }
    line = json.dumps(record, ensure_ascii=False)
    is_debug = os.getenv("DEBUG", "").lower() == "true"
    if level == "ERROR":
        print(line, flush=True)
    elif level == "WARNING" and is_debug:
        print(line, flush=True)
    elif level == "DEBUG" and is_debug:
        print(line, flush=True)


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
