"""
文档整理脚本 — 每 3 天低峰期执行一次（由外部 cron 调度）

功能：
  1. 检查时间戳，距上次运行不足 3 天则退出
  2. 遍历 data/docs/*.md，跳过行数少于 MIN_LINES 的文件
  3. 对每个文件调用 LLM 做全文整理：去重、合并相似段落、结构化
  4. 覆写原文件（保留 frontmatter）并重新索引
  5. 更新时间戳

cron 示例（每天凌晨 2:00 运行，脚本内部控制 3 天间隔）：
  0 2 * * * cd /path/to/rag-agent && python scripts/consolidate_docs.py >> logs/consolidate.log 2>&1
"""

import glob
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

DOCS_DIR = _ROOT / "data" / "docs"
LOGS_DIR = _ROOT / "logs"
TIMESTAMP_FILE = _ROOT / ".consolidate_last_run"

# ── 常量 ──────────────────────────────────────────
INTERVAL_DAYS = 3          # 最短执行间隔（天）
MIN_LINES = 60             # 行数低于此值的文件跳过（内容太少，无需整理）
MAX_INPUT_CHARS = 12000    # 传给 LLM 的最大正文字符数（避免超 context）

# ── 日志 ──────────────────────────────────────────
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
_logger = logging.getLogger("consolidate_docs")


# ── LLM 客户端（复用 knowledge_internalizer 的配置）──
def _build_client():
    """从环境变量构建 LLM 客户端，优先使用 LLM_JUDGE，其次 LLM_MODEL。"""
    from openai import OpenAI

    _PROVIDERS = {
        "qianfan": {
            "env_var": "QIANFAN_API_KEY",
            "base_url": "https://qianfan.baidubce.com/v2",
            "default_model": "ernie-speed-pro-128k",
        },
        "bailian": {
            "env_var": "DASHSCOPE_API_KEY",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "default_model": "qwen-turbo",
        },
    }

    for env_key in ("LLM_JUDGE", "LLM_MODEL"):
        entry = os.getenv(env_key, "").strip()
        if not entry:
            continue
        parts = entry.split("/", 1)
        provider = parts[0].lower()
        if provider not in _PROVIDERS:
            continue
        cfg = _PROVIDERS[provider]
        api_key = os.getenv(cfg["env_var"])
        if not api_key:
            continue
        model = parts[1] if len(parts) == 2 else cfg["default_model"]
        _logger.info("Using LLM: %s / %s (from %s)", provider, model, env_key)
        return OpenAI(api_key=api_key, base_url=cfg["base_url"]), model

    _logger.error("No valid LLM client configured. Set LLM_JUDGE or LLM_MODEL.")
    return None, None


# ── frontmatter 解析 ──────────────────────────────
def _split_frontmatter(content: str):
    """返回 (frontmatter_block, body)，frontmatter_block 含首尾 --- 分隔符。"""
    if not content.startswith("---"):
        return "", content
    end = content.find("\n---", 3)
    if end == -1:
        return "", content
    frontmatter = content[: end + 4]   # 包含结尾的 ---
    body = content[end + 4:].lstrip("\n")
    return frontmatter, body


# ── 时间戳检查 ────────────────────────────────────
def _last_run_ts() -> float:
    """返回上次运行的时间戳，未运行过返回 0。"""
    if not TIMESTAMP_FILE.exists():
        return 0.0
    try:
        return float(TIMESTAMP_FILE.read_text().strip())
    except Exception:
        return 0.0


def _updated_sources_since(since_ts: float) -> list:
    """
    返回 since_ts 之后有变更的 source 文件名列表。
    优先查 Qdrant payload 中的 indexed_at（云端内化也能感知），
    降级 fallback：若 Qdrant 不可用则扫描本地文件 mtime。
    """
    try:
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url and qdrant_key:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
            client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=10)
            records, _ = client.scroll(
                collection_name="rag_docs",
                scroll_filter=Filter(must=[
                    FieldCondition(key="type", match=MatchValue(value="knowledge_base")),
                    FieldCondition(key="indexed_at", range=Range(gt=since_ts)),
                ]),
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            sources = list({r.payload["source"] for r in records if "source" in r.payload})
            _logger.info("Qdrant: %d sources updated since last run: %s", len(sources), sources)
            return sources
    except Exception as e:
        _logger.warning("Qdrant query failed, falling back to file mtime: %s", e)

    # fallback：文件系统 mtime
    sources = [p.name for p in DOCS_DIR.glob("*.md") if p.stat().st_mtime > since_ts]
    _logger.info("File mtime fallback: %d sources updated: %s", len(sources), sources)
    return sources


def _should_run() -> bool:
    last_run = _last_run_ts()

    # 1. 间隔未到，直接跳过
    if last_run > 0:
        elapsed_days = (time.time() - last_run) / 86400
        if elapsed_days < INTERVAL_DAYS:
            _logger.info(
                "Last run was %.1f days ago, interval is %d days. Skipping.",
                elapsed_days, INTERVAL_DAYS,
            )
            return False

    # 2. 间隔已到，但若无文档更新则无需整理，节省 LLM 资源
    if last_run > 0 and not _updated_sources_since(last_run):
        _logger.info("No sources updated since last run. Skipping.")
        return False

    return True


def _update_timestamp():
    TIMESTAMP_FILE.write_text(str(time.time()))


# ── 核心整理逻辑 ──────────────────────────────────
def _consolidate_body(body: str, topic: str, client, model: str) -> str | None:
    """
    调用 LLM 对文档正文做整理：
    - 合并语义重复的段落
    - 统一结构（标题/要点/来源）
    - 返回整理后的正文，失败返回 None
    """
    truncated = body[:MAX_INPUT_CHARS]
    if len(body) > MAX_INPUT_CHARS:
        _logger.warning("Body truncated from %d to %d chars for topic=%r", len(body), MAX_INPUT_CHARS, topic)

    messages = [
        {
            "role": "system",
            "content": (
                f"你是知识库编辑，负责整理「{topic}」主题文档。\n"
                "该文档由多次网络搜索结果追加而成，存在重复段落和碎片化内容。\n"
                "请对正文进行整理，要求：\n"
                "1. 合并语义重复的段落，保留最完整的表述\n"
                "2. 删除广告、无关内容、重复的来源声明\n"
                "3. 统一使用 markdown 格式：二级标题 + 要点列表\n"
                "4. 末尾保留一个「参考来源」章节，去重后汇总所有来源 URL\n"
                "5. 不要添加原文没有的新知识\n"
                "6. 只输出整理后的正文，不要包含 frontmatter，不要解释"
            ),
        },
        {
            "role": "user",
            "content": f"原始正文：\n\n{truncated}",
        },
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        _logger.error("LLM consolidation failed for topic=%r: %s", topic, e)
        return None


# ── 文件处理 ──────────────────────────────────────
def _process_file(filepath: Path, client, model: str) -> bool:
    """整理单个文件，成功返回 True。"""
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()

    if len(lines) < MIN_LINES:
        _logger.info("Skipping %s (%d lines < %d)", filepath.name, len(lines), MIN_LINES)
        return False

    frontmatter, body = _split_frontmatter(content)
    if not body.strip():
        _logger.info("Skipping %s (empty body)", filepath.name)
        return False

    # 从 frontmatter 提取 topic
    topic = filepath.stem
    for line in frontmatter.splitlines():
        if line.startswith("topic:"):
            topic = line.split(":", 1)[1].strip()
            break

    _logger.info("Consolidating %s (topic=%r, %d lines)...", filepath.name, topic, len(lines))

    new_body = _consolidate_body(body, topic, client, model)
    if not new_body:
        return False

    # 覆写文件：frontmatter + 整理后正文
    new_content = (frontmatter + "\n\n" + new_body + "\n") if frontmatter else (new_body + "\n")
    filepath.write_text(new_content, encoding="utf-8")

    new_lines = len(new_content.splitlines())
    _logger.info(
        "Consolidated %s: %d → %d lines (%.0f%% reduction)",
        filepath.name, len(lines), new_lines,
        max(0, (1 - new_lines / len(lines)) * 100),
    )
    return True


# ── 重新索引 ──────────────────────────────────────
def _reindex(filepath: Path):
    """调用 indexer 对整理后的文件重新建索引。"""
    try:
        from rag.indexer import index_single_document
        count = index_single_document(str(filepath))
        _logger.info("Reindexed %s: %d chunks", filepath.name, count)
    except Exception as e:
        _logger.error("Reindex failed for %s: %s", filepath.name, e)


# ── 主入口 ────────────────────────────────────────
def main():
    _logger.info("=" * 60)
    _logger.info("consolidate_docs started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if not _should_run():
        return

    # 加载 .env（生产环境通过 cron 环境变量注入，本地开发用 .env）
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    client, model = _build_client()
    if not client:
        sys.exit(1)

    last_run = _last_run_ts()
    # 取有变更的 source 集合（首次运行 last_run=0，返回所有文件）
    updated_sources: set = set(_updated_sources_since(last_run)) if last_run > 0 else set()

    all_md_files = sorted(glob.glob(str(DOCS_DIR / "*.md")))
    # 首次运行处理全部，否则只处理有变更的
    if last_run > 0 and updated_sources:
        md_files = [p for p in all_md_files if Path(p).name in updated_sources]
        _logger.info(
            "Processing %d updated files (out of %d total): %s",
            len(md_files), len(all_md_files),
            [Path(p).name for p in md_files],
        )
    else:
        md_files = all_md_files
        _logger.info("First run or full pass: processing all %d .md files", len(md_files))

    consolidated = 0
    for path_str in md_files:
        filepath = Path(path_str)
        changed = _process_file(filepath, client, model)
        if changed:
            _reindex(filepath)
            consolidated += 1
        # 每个文件处理后稍作停顿，避免 LLM API 限流
        time.sleep(1)

    _update_timestamp()
    _logger.info(
        "Done. Consolidated %d / %d files. Next run in %d days.",
        consolidated, len(md_files), INTERVAL_DAYS,
    )


if __name__ == "__main__":
    main()

