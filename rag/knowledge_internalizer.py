"""
知识内化模块

对通用知识类搜索结果，提炼后增量写入 data/docs/ 下的 .md 文件，并重新索引。
实时类信息（天气/股价/突发新闻等）直接跳过，不做内化。
"""
import glob
import os
import json
import hashlib
from datetime import datetime

from rag.indexer import index_single_document

# 内化状态文件路径，供 app.py 侧边栏读取
STATUS_FILE = os.path.join(os.path.dirname(__file__), "..", ".internalize_status.json")
MAX_STATUS_ENTRIES = 5  # 最多保留最近 5 条


def _write_status(query: str, filename: str) -> None:
    """将内化结果写入状态文件，供前端展示。"""
    try:
        entries = []
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
        entries.insert(0, {
            "query": query[:50],
            "file": filename,
            "time": datetime.now().strftime("%H:%M"),
        })
        entries = entries[:MAX_STATUS_ENTRIES]
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
    except Exception:
        pass

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

REALTIME_KEYWORDS = [
    "天气", "气温", "温度", "下雨", "晴", "阴", "雪",
    "股价", "股票", "今日", "今天", "实时", "最新行情", "涨跌",
    "breaking news", "weather", "stock price", "today's",
]

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")


# ──────────────────────────────────────────────
# 公开入口
# ──────────────────────────────────────────────

def internalize_async(query: str, results: list, client, model: str) -> None:
    """入口函数，供外部在 daemon 线程中调用。所有异常静默处理。"""
    try:
        _internalize(query, results, client, model)
    except Exception:
        pass


# ──────────────────────────────────────────────
# 内部实现
# ──────────────────────────────────────────────

def _internalize(query: str, results: list, client, model: str) -> None:
    """Step 1 ~ Step 5 完整流程。"""

    # Step 1 — 分类判断：实时类直接跳过
    if _is_realtime(query, results, client, model):
        return

    # Step 2 — LLM 提炼
    refined = _refine(query, results, client, model)
    if not refined:
        return

    # Step 3 — 路由到目标文档
    filepath = _route(query, refined, client, model)

    # Step 4 — 增量写入文档
    _append_to_file(filepath, query, refined)

    # Step 5 — 重新索引
    index_single_document(filepath)

    # 写入状态文件，供侧边栏展示
    _write_status(query, os.path.basename(filepath))


# ──────────────────────────────────────────────
# Step 1 — 分类判断
# ──────────────────────────────────────────────

def _is_realtime(query: str, results: list, client, model: str) -> bool:
    """返回 True 表示属于实时类信息，应跳过内化。"""
    titles = " ".join(r.get("title", "") for r in results)
    combined = (query + " " + titles).lower()

    # 硬规则优先
    for kw in REALTIME_KEYWORDS:
        if kw.lower() in combined:
            return True

    # 未命中硬规则，调用 LLM 判断
    messages = [
        {
            "role": "system",
            "content": (
                "判断用户的搜索内容是否属于实时性信息（天气/股价/突发新闻/当日数据等）。"
                "只回答 yes 或 no，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": f"搜索词：{query}\n搜索结果标题：{titles}",
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content.strip().lower()
    return "yes" in answer


# ──────────────────────────────────────────────
# Step 2 — LLM 提炼
# ──────────────────────────────────────────────

def _refine(query: str, results: list, client, model: str) -> str:
    """将搜索结果提炼为结构化 markdown 知识条目。"""
    results_text = "\n\n".join(
        f"标题：{r.get('title', '')}\n摘要：{r.get('snippet', r.get('body', ''))}\nURL：{r.get('url', r.get('href', ''))}"
        for r in results
    )

    messages = [
        {
            "role": "system",
            "content": (
                "你是知识库编辑。将以下搜索结果提炼为简洁的技术知识条目，要求：\n"
                "1. markdown 格式，包含小标题和要点\n"
                "2. 去除广告、导航栏、无关内容\n"
                "3. 保留核心概念、定义、对比、原理\n"
                "4. 末尾附上来源 URL\n"
                "5. 控制在 500 字以内"
            ),
        },
        {
            "role": "user",
            "content": f"搜索词：{query}\n\n搜索结果：\n{results_text}",
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# Step 3 — 路由到目标文档
# ──────────────────────────────────────────────

def _read_frontmatter(content: str) -> dict:
    """解析 markdown 文件头部的 YAML frontmatter，返回 key-value 字典。"""
    if not content.startswith("---"):
        return {}
    end = content.find("\n---", 3)
    if end == -1:
        return {}
    meta = {}
    for line in content[3:end].strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta


def _sanitize_filename(raw: str) -> str:
    """从 LLM 输出中提取合法的 snake_case .md 文件名。"""
    import re
    # 取第一行
    line = raw.splitlines()[0].strip()
    # 只保留字母数字下划线连字符及点
    cleaned = re.sub(r"[^\w\-.]", "_", line)
    # 确保以 .md 结尾
    if not cleaned.endswith(".md"):
        cleaned = cleaned.rstrip(".") + ".md"
    # 防止空文件名
    if cleaned in (".md",):
        cleaned = "misc_knowledge.md"
    return cleaned


def _new_file(query: str, client, model: str) -> str:
    """让 LLM 生成文件名，新建带 frontmatter 的空文件，返回绝对路径。"""
    messages = [
        {
            "role": "user",
            "content": (
                "只返回文件名，如 recommendation_system.md，不要解释。\n"
                f"主题：{query}"
            ),
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    raw_name = response.choices[0].message.content.strip()
    filename = _sanitize_filename(raw_name)
    filepath = os.path.join(DOCS_DIR, filename)

    # 推导 topic：去掉 .md 后缀，下划线换空格
    topic = filename[:-3].replace("_", " ")
    frontmatter = (
        f"---\n"
        f"topic: {topic}\n"
        f"keywords: \n"
        f"description: {query} 相关知识\n"
        f"type: knowledge_base\n"
        f"---\n\n"
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(frontmatter)
    return filepath


def _route(query: str, refined: str, client, model: str) -> str:
    """
    动态读取 DOCS_DIR 下所有 .md 的 frontmatter description，
    让 LLM 选择最合适的目标文件。
    返回目标文件的绝对路径。
    """
    # 1. 扫描所有 .md 文件（递归），读取 frontmatter
    candidates = []  # [{"path": str, "filename": str, "description": str}]
    for path in glob.glob(os.path.join(DOCS_DIR, "**", "*.md"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        meta = _read_frontmatter(content)
        if meta.get("description"):
            candidates.append({
                "path": path,
                "filename": os.path.basename(path),
                "description": meta["description"],
            })

    # 2. 构造候选列表给 LLM
    if not candidates:
        return _new_file(query, client, model)

    options = "\n".join(
        f"{i + 1}. {c['filename']}: {c['description']}"
        for i, c in enumerate(candidates)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "你是知识库管理员。根据搜索内容，从候选文件列表中选择最合适的文件存放该知识。\n"
                "如果没有合适的文件，回答 'new'。\n"
                "只回答文件名（如 rag_introduction.md）或 'new'，不要解释。"
            ),
        },
        {
            "role": "user",
            "content": f"搜索词：{query}\n\n候选文件：\n{options}",
        },
    ]
    response = client.chat.completions.create(model=model, messages=messages, temperature=0)
    answer = response.choices[0].message.content.strip()

    # 3. 匹配回答到候选文件
    for c in candidates:
        if c["filename"] in answer:
            return c["path"]

    # 4. 没有匹配或回答 new，新建文件
    return _new_file(query, client, model)


# ──────────────────────────────────────────────
# Step 4 — 增量写入文档
# ──────────────────────────────────────────────

def _append_to_file(filepath: str, query: str, refined: str) -> None:
    """将提炼内容追加到目标文件，写入前检查重复。"""
    query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()

    # 检查文件末尾 2000 字符，避免重复追加同一 query
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            f.seek(0, 2)  # 移到文件末尾
            size = f.tell()
            tail_start = max(0, size - 2000)
            f.seek(tail_start)
            tail = f.read()
        if query_hash in tail:
            return
        mode = "a"
    else:
        # 新建文件，确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mode = "w"

    today = datetime.now().strftime("%Y-%m-%d")
    block = (
        f"\n\n---\n"
        f"## 补充知识（来自网络搜索）\n"
        f"> query: {query} | 更新时间: {today} | hash: {query_hash}\n\n"
        f"{refined}\n"
    )

    with open(filepath, mode, encoding="utf-8") as f:
        f.write(block)
