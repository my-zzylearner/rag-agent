#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

print_step() {
  echo
  echo "==> $1"
}

cd "$ROOT_DIR"

print_step "检查 Python 环境"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "未找到 $PYTHON_BIN，请先安装 Python 3.10+"
  exit 1
fi

PYTHON_VER="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VER" in
  3.10|3.11|3.12) ;;
  *)
    echo "当前 Python 版本为 ${PYTHON_VER:-unknown}，建议使用 3.10/3.11/3.12。"
    echo "可这样启动：PYTHON_BIN=python3.11 ./scripts/run_local_ai_search.sh"
    exit 1
    ;;
esac

print_step "准备虚拟环境"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

VENV_PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$VENV_PY_VER" in
  3.10|3.11|3.12) ;;
  *)
    print_step "重建虚拟环境（当前 venv Python=$VENV_PY_VER 不受支持）"
    deactivate || true
    rm -rf "$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ;;
esac

print_step "安装依赖"
python -m pip install --upgrade pip
pip install -r requirements.txt

print_step "检查环境变量文件"
if [ ! -f "$ROOT_DIR/.env" ]; then
  cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
  echo "已创建 .env，请先填写 DASHSCOPE_API_KEY（可选 TAVILY_API_KEY）后重试。"
  echo "编辑文件: $ROOT_DIR/.env"
  exit 1
fi

if command -v rg >/dev/null 2>&1; then
  HAS_DASHSCOPE=0
  HAS_TAVILY=0
  rg -q "^\s*DASHSCOPE_API_KEY\s*=\s*\S+" "$ROOT_DIR/.env" || HAS_DASHSCOPE=1
  rg -q "^\s*TAVILY_API_KEY\s*=\s*\S+" "$ROOT_DIR/.env" || HAS_TAVILY=1
else
  HAS_DASHSCOPE=0
  HAS_TAVILY=0
  grep -Eq "^[[:space:]]*DASHSCOPE_API_KEY[[:space:]]*=[[:space:]]*\\S+" "$ROOT_DIR/.env" || HAS_DASHSCOPE=1
  grep -Eq "^[[:space:]]*TAVILY_API_KEY[[:space:]]*=[[:space:]]*\\S+" "$ROOT_DIR/.env" || HAS_TAVILY=1
fi

if [ "$HAS_DASHSCOPE" -ne 0 ]; then
  echo ".env 中未检测到 DASHSCOPE_API_KEY，请先配置后再启动。"
  exit 1
fi

if [ "$HAS_TAVILY" -ne 0 ]; then
  echo "提示：未配置 TAVILY_API_KEY，联网搜索工具可能不可用（本地知识库仍可用）。"
fi

print_step "启动 AI Search（Streamlit）"
exec streamlit run app.py
