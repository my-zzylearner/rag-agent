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
from openai import OpenAI

from rag.indexer import index_single_document
from utils.logger import get_logger

_logger = get_logger(__name__)

# LLM provider 配置（与 agent/agent.py 保持一致，避免循环导入）
_PROVIDERS = {
    "qianfan": {
        "env_var": "QIANFAN_API_KEY",
        "base_url": "https://qianfan.baidubce.com/v2",
        "model": "ernie-speed-pro-128k",
    },
    "bailian": {
        "env_var": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
    },
}


def _build_client(entry: str):
    """根据 'provider/model' 格式的字符串构建 (client, model) 元组，失败返回 (None, None)。"""
    parts = entry.strip().split("/", 1)
    provider = parts[0].lower()
    if provider not in _PROVIDERS:
        return None, None
    cfg = _PROVIDERS[provider]
    model = parts[1] if len(parts) == 2 else cfg["model"]
    api_key = os.getenv(cfg["env_var"])
    if not api_key:
        return None, None
    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), model


_FALLBACK_ERRORS = ("quota", "allocationquota", "403", "429", "rate limit",
                    "insufficient", "free tier", "billing", "balance")


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
    except Exception as e:
        _logger.error("knowledge internalization failed: query=%r error=%s", query, e, exc_info=True)


# ──────────────────────────────────────────────
# 内部实现
# ──────────────────────────────────────────────

def _build_judge_candidates() -> list:
    """
    构建质量判断 LLM 候选列表（LLM_JUDGE + LLM_JUDGE_FALLBACK）。
    未配置时返回空列表，调用方降级为纯规则判断。
    """
    candidates = []
    primary = os.getenv("LLM_JUDGE", "")
    if primary:
        c, m = _build_client(primary)
        if c:
            candidates.append((c, m))
        else:
            _logger.warning("LLM_JUDGE=%r invalid or api key missing", primary)
    fallback_str = os.getenv("LLM_JUDGE_FALLBACK", "")
    for entry in [s.strip() for s in fallback_str.split(",") if s.strip()]:
        c, m = _build_client(entry)
        if c:
            candidates.append((c, m))
    return candidates


def _build_internalize_candidates(client, model: str) -> list:
    """
    构建内化 LLM 候选列表：主模型 + LLM_INTERNALIZE_FALLBACK 配置的备用模型。
    返回 [(client, model), ...] 列表，主模型在首位。
    """
    candidates = [(client, model)]
    fallback_str = os.getenv("LLM_INTERNALIZE_FALLBACK", "")
    for entry in [s.strip() for s in fallback_str.split(",") if s.strip()]:
        c, m = _build_client(entry)
        if c:
            candidates.append((c, m))
    return candidates


def _should_internalize_fallback(exc: Exception) -> bool:
    """判断内化 LLM 调用失败是否应切换到备用模型。"""
    msg = str(exc).lower()
    return any(k in msg for k in _FALLBACK_ERRORS)


def _call_with_fallback(fn, query, candidates, *args):
    """
    用 candidates 列表依次尝试调用 fn(query, *args, client, model)。
    遇到额度/限流类错误自动切换下一个模型，其他错误直接抛出。
    所有候选都失败时抛出最后一个异常。
    """
    last_exc = None
    for c, m in candidates:
        try:
            return fn(query, *args, c, m)
        except Exception as e:
            if _should_internalize_fallback(e):
                _logger.warning("internalize fallback: %s -> next candidate, reason: %s", m, e)
                last_exc = e
                continue
            raise
    raise last_exc


def _internalize(query: str, results: list, client, model: str) -> None:
    """Step 1 ~ Step 5 完整流程，内部自动 fallback。"""
    candidates = _build_internalize_candidates(client, model)

    # Step 1 — 分类判断：实时类直接跳过
    if _call_with_fallback(_is_realtime, query, candidates, results):
        return

    # Step 2 — LLM 提炼
    refined = _call_with_fallback(_refine, query, candidates, results)
    if not refined:
        return

    # Step 2.5 — 质量过滤（规则 + 可选 LLM 打分，judge 有自己的 fallback）
    judge_candidates = _build_judge_candidates()
    if not _is_quality_refined(refined, query, judge_candidates):
        return

    # Step 3 — 路由到目标文档
    filepath = _call_with_fallback(_route, query, candidates, refined)

    # Step 4 — 增量写入文档
    _append_to_file(filepath, query, refined)

    # Step 5 — 重新索引
    index_single_document(filepath)

    # Step 6 — 写入 Gist 审计记录（异步，不阻塞）
    filename = os.path.basename(filepath)
    try:
        from utils.gist_store import add_internalized
        add_internalized(query, filename, refined)
    except Exception:
        pass

    _logger.info("knowledge internalization succeeded: query=%r file=%s", query, filename)


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
# Step 2.5 — 质量过滤
# ──────────────────────────────────────────────

# 提炼失败时 LLM 常见的拒绝短语
_REFUSAL_PHRASES = [
    "无法提炼",
    "无相关内容",
    "没有相关内容",
    "无法从",
    "未找到相关",
    "没有找到相关",
    "无有效信息",
    "无法获取",
    "无内容可提炼",
    "抱歉，没有",
    "抱歉，无法",
    "sorry, i cannot",
    "no relevant content",
    "no useful information",
    "unable to extract",
]

def _similarity_ratio(a: str, b: str) -> float:
    """计算两个字符串的字符级重叠比例（简单实现，不依赖外部库）。"""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b
    return len(intersection) / max(len(set_a), len(set_b))


def _is_quality_refined(refined: str, query: str, judge_candidates: list = None) -> bool:
    """
    返回 True 表示提炼结果质量合格，可以继续内化。

    判断流程：
    1. 硬规则：拒绝语检测
    2. 硬规则：与 query 重复度检测
    3. LLM 打分：遍历 judge_candidates，额度/限流错误自动切换下一个，
                 全部失败时降级为硬规则通过（不阻断内化）
    """
    # 规则 1：拒绝语检测
    lower = refined.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase.lower() in lower:
            _logger.info("internalization skipped (refusal): query=%r phrase=%r", query, phrase)
            return False

    # 规则 2：与 query 重复度过高
    ratio = _similarity_ratio(refined[:200], query)
    if ratio >= 0.85:
        _logger.info("internalization skipped (too similar): query=%r ratio=%.2f", query, ratio)
        return False

    # 规则 3：LLM 打分（有自己的 fallback，全部失败时降级通过）
    if judge_candidates:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是知识库质量审核员。判断以下内容是否值得写入知识库。\n"
                    "值得写入的标准：包含有价值的技术知识、概念解释、方法论等。\n"
                    "不值得写入的标准：广告、导航链接、无实质内容的摘要、与主题无关的内容。\n"
                    "只回答 yes 或 no，不要解释。"
                ),
            },
            {"role": "user", "content": f"搜索词：{query}\n\n提炼内容：\n{refined[:500]}"},
        ]
        for jc, jm in judge_candidates:
            try:
                resp = jc.chat.completions.create(model=jm, temperature=0, messages=messages)
                answer = resp.choices[0].message.content.strip().lower()
                if "no" in answer and "yes" not in answer:
                    _logger.info("internalization skipped (llm judge=no): query=%r model=%s", query, jm)
                    return False
                break  # 成功拿到判断结果，不再尝试下一个
            except Exception as e:
                if _should_internalize_fallback(e):
                    _logger.warning("llm judge fallback: %s -> next, reason: %s", jm, e)
                    continue
                # 非额度类错误，降级为硬规则通过
                _logger.warning("llm judge error, rule-only fallback: %s", e)
                break

    return True


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

    # 检查文件末尾 2000 字节，避免重复追加同一 query
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            tail_start = max(0, size - 2000)
            f.seek(tail_start)
            tail = f.read().decode("utf-8", errors="ignore")
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
