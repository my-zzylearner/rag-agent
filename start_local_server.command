#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/local-streamlit.log"
PID_FILE="$ROOT/.streamlit-local.pid"

cd "$ROOT"
mkdir -p "$LOG_DIR"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# 优先选择项目支持的 Python 版本
PY_BIN=""
for c in python3.12 python3.11 python3.10; do
  if command -v "$c" >/dev/null 2>&1; then
    PY_BIN="$c"
    break
  fi
done

if [ -z "$PY_BIN" ]; then
  echo "未检测到 python3.10/3.11/3.12。请先安装其中一个版本（推荐 3.11）。"
  echo
  echo "按回车退出..."
  read -r _
  exit 1
fi

# 准备虚拟环境
if [ ! -d "$ROOT/venv" ]; then
  "$PY_BIN" -m venv "$ROOT/venv"
fi
source "$ROOT/venv/bin/activate"

# 若 venv 版本不受支持，自动重建
VENV_PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$VENV_PY_VER" in
  3.10|3.11|3.12) ;;
  *)
    deactivate || true
    rm -rf "$ROOT/venv"
    "$PY_BIN" -m venv "$ROOT/venv"
    source "$ROOT/venv/bin/activate"
    ;;
esac

python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

if [ ! -f "$ROOT/.env" ]; then
  echo "未找到 .env，请先配置。"
  echo
  echo "按回车退出..."
  read -r _
  exit 1
fi

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
  echo "本地服务已在运行，PID=$(cat "$PID_FILE")"
else
  nohup streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501 >"$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 2
fi

open "http://127.0.0.1:8501"
echo "已启动/复用本地服务，日志：$LOG_FILE"
echo "停止服务：双击 stop_local_server.command"
echo
echo "按回车退出..."
read -r _

