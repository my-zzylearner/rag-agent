"""
知识内化模块

对通用知识类搜索结果，提炼后增量写入 data/docs/ 下的 .md 文件，并重新索引。
实时类信息（天气/股价/突发新闻等）直接跳过，不做内化。
"""
import os
import hashlib
from datetime import datetime

from rag.indexer import index_single_document

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

REALTIME_KEYWORDS = [
    "天气", "气温", "温度", "下雨", "晴", "阴", "雪",
    "股价", "股票", "今日", "今天", "实时", "最新行情", "涨跌",
    "breaking news", "weather", "stock price", "today's",
]

TOPIC_MAP = {
    "rag_introduction.md":       ["rag", "检索增强", "embedding", "向量检索", "chunking", "retrieval"],
    "llm_agent_architecture.md": ["agent", "function calling", "react", "工具调用", "规划", "plan", "tool use"],
    "vector_databases.md":       ["向量数据库", "chroma", "pinecone", "milvus", "qdrant", "faiss", "vector db"],
    "search_algorithms.md":      ["bm25", "混合检索", "reranking", "排序", "召回", "hybrid search"],
}

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

def _route(query: str, refined: str, client, model: str) -> str:
    """返回目标文档的绝对路径。"""
    combined = (query + " " + refined).lower()

    best_file = None
    best_score = 0
    for filename, keywords in TOPIC_MAP.items():
        score = sum(1 for kw in keywords if kw.lower() in combined)
        if score > best_score:
            best_score = score
            best_file = filename

    if best_file and best_score > 0:
        return os.path.join(DOCS_DIR, best_file)

    # 无匹配，让 LLM 生成新文件名
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
    # 取最后一个 token，防止 LLM 多余输出
    filename = _sanitize_filename(raw_name)
    return os.path.join(DOCS_DIR, filename)


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
