#!/bin/bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
chmod +x scripts/run_local_ai_search.sh

PY_BIN=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then
    PY_BIN="$c"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo "未检测到 python3.10/3.11/3.12。"
  echo "请先安装其中一个版本（推荐 python3.11）。"
  echo
  echo "按回车退出..."
  read -r _
  exit 1
fi

if ! PYTHON_BIN="$PY_BIN" ./scripts/run_local_ai_search.sh; then
  echo
  echo "启动失败，按回车退出..."
  read -r _
  exit 1
fi

