#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT/.streamlit-local.pid"

if [ -f "$PID_FILE" ]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID"
    echo "已停止本地服务，PID=$PID"
  else
    echo "PID 文件存在，但进程不在运行。"
  fi
  rm -f "$PID_FILE"
else
  pkill -f "streamlit run app.py --server.address 127.0.0.1 --server.port 8501" >/dev/null 2>&1 || true
  echo "未找到 PID 文件，已尝试按命令特征停止。"
fi

echo
echo "按回车退出..."
read -r _

