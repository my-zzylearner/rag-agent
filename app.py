"""
Streamlit 前端入口
"""
import os
import threading
from dotenv import load_dotenv

# 禁用 chromadb 遥测，避免 opentelemetry 依赖问题（强制覆盖，确保生效）
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
import streamlit as st

# Streamlit Cloud 通过环境变量注入 secrets，不存在 .env 文件
# 本地开发时重新加载 .env，先清除旧值确保注释掉的变量失效
_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip().lstrip("#;").strip()
            if "=" in _line:
                os.environ.pop(_line.split("=", 1)[0].strip(), None)
load_dotenv(_env_file)

from rag.indexer import index_documents, is_indexed, get_collection  # noqa: E402
from agent.agent import run_agent  # noqa: E402
from utils.gist_store import load as gist_load, increment as gist_increment, add_feedback as gist_add_feedback  # noqa: E402

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔍",
    layout="centered",
)

# ── 访问密码校验 ──────────────────────────────────────────
def _check_password() -> bool:
    """返回 True 表示已通过验证。密码存在 st.secrets 或环境变量 APP_PASSWORD 中。"""
    import os
    try:
        correct = st.secrets.get("APP_PASSWORD", "") or ""
    except Exception:
        correct = ""
    correct = correct or os.getenv("APP_PASSWORD", "")
    if not correct:
        return True  # 未配置密码则不拦截（本地开发时）

    if st.session_state.get("authenticated"):
        return True

    st.title("🔍 AI Search Agent")
    pwd = st.text_input("请输入访问密码", type="password", key="pwd_input")
    if st.button("进入"):
        if pwd == correct:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("密码错误")
    st.stop()

_check_password()

st.title("🔍 AI Search Agent")
st.caption("RAG + Web Search · Powered by Qwen & Tavily")

@st.cache_resource
def _load_stats_once():
    """进程级缓存，只在冷启动时加载一次 Gist 数据。"""
    return {"data": gist_load()}

_stats_cache = _load_stats_once()

# ── 初始化知识库（只在第一次运行时索引）────────────────────
@st.cache_resource
def _warm_up_embedder():
    from rag.indexer import get_embedder
    embedder = get_embedder()
    embedder.encode(["warm up"])
    return embedder


@st.cache_resource
def init_index():
    if not is_indexed():
        return index_documents("./data/docs")
    return None


# 冷启动进度展示（只在第一次 session 时展示，之后静默初始化）
if "startup_done" not in st.session_state:
    with st.status("启动中...", expanded=True) as _startup_status:
        st.write("⏳ 正在加载 Embedding 模型...")
        _warm_up_embedder()
        st.write("✅ Embedding 模型已就绪")
        st.write("⏳ 正在检查知识库...")
        chunk_count = init_index()
        if chunk_count is not None:
            st.write(f"✅ 知识库索引完成，写入 {chunk_count} 个文本块")
        else:
            st.write("✅ 知识库已就绪")
        _startup_status.update(label="启动完成", state="complete", expanded=False)
    st.session_state.startup_done = True
else:
    _warm_up_embedder()
    chunk_count = init_index()

# ── 状态初始化 ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "prefill_input" not in st.session_state:
    st.session_state.prefill_input = ""
if "fb_input_key" not in st.session_state:
    st.session_state.fb_input_key = 0
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

# 访问计数：每个 session 只计一次
if "visit_counted" not in st.session_state:
    st.session_state.visit_counted = True
    gist_increment("visits")

# 每次 rerun 都清除停止信号，避免残留
# 注意：agent_running 不在这里重置，由 Agent 执行块自己管理
st.session_state.stop_event.clear()

# ── 参数配置 + 知识库统计（侧边栏读取，需在 run_agent 调用前定义）──
if "cfg_max_rounds" not in st.session_state:
    st.session_state.cfg_max_rounds = 3
if "cfg_top_k" not in st.session_state:
    st.session_state.cfg_top_k = 4

if st.session_state.agent_running:
    st.sidebar.caption(f"最大工具调用轮次：{st.session_state.cfg_max_rounds}")
    st.sidebar.caption(f"知识库检索条数：{st.session_state.cfg_top_k}")
    cfg_max_rounds = st.session_state.cfg_max_rounds
    cfg_top_k = st.session_state.cfg_top_k
else:
    cfg_max_rounds = st.sidebar.slider("最大工具调用轮次", min_value=1, max_value=6,
                                        value=st.session_state.cfg_max_rounds, step=1)
    cfg_top_k = st.sidebar.slider("知识库检索条数", min_value=1, max_value=10,
                                   value=st.session_state.cfg_top_k, step=1)
    st.session_state.cfg_max_rounds = cfg_max_rounds
    st.session_state.cfg_top_k = cfg_top_k

@st.cache_data(ttl=60)
def _get_kb_stats():
    col = get_collection()
    total = col.count()
    web = len(col.get(where={"type": {"$eq": "web_cache"}}, include=[])["ids"])
    return total, web

try:
    _total, _web = _get_kb_stats()
    _local = _total - _web
    st.sidebar.caption(f"知识库：{_local} 本地 + {_web} 网络缓存 = {_total} 条")
except Exception:
    st.sidebar.caption("知识库：统计加载中...")

RELEVANCE_THRESHOLD = 0.3  # 低于此相关度的检索结果不展示


def _render_sources(sources: list, query: str = "") -> None:
    """统一渲染参考来源，过滤低相关度结果。若传入 query 则高亮关键词。"""
    import re

    def _highlight(text: str, q: str) -> str:
        if not q:
            return text
        # 按空格拆词，同时把整个 query 也作为候选词
        # 再用正则把连续英文/数字串单独提取（如 RAG、LLM 等缩写）
        candidates = set(q.split())
        candidates.add(q)
        candidates.update(re.findall(r'[A-Za-z0-9]{2,}', q))
        words = [w for w in candidates if len(w) >= 2]
        # 按长度降序，避免短词先匹配破坏长词
        words.sort(key=len, reverse=True)
        for word in words:
            text = re.sub(
                f"({re.escape(word)})",
                r"**\1**",
                text,
                flags=re.IGNORECASE,
            )
        return text

    filtered = [s for s in sources if s.get("relevance_score", 0) >= RELEVANCE_THRESHOLD]
    if not filtered:
        return
    with st.expander("📎 参考来源", expanded=False):
        for src in filtered:
            score = src.get("relevance_score", "")
            snippet = src["content"].replace("\n", " ").strip()[:120]
            st.caption(f"📄 {src['source']}　相关度 {score}")
            st.markdown(_highlight(snippet, query) + "…")


# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("steps"):
            with st.expander("✅ 完成", expanded=False):
                for step in msg["steps"]:
                    if step["kind"] == "retry":
                        st.warning(step["text"])
                    else:
                        st.write(step["text"])
        if msg.get("sources"):
            _render_sources(msg["sources"])

# ── 用户输入 ──────────────────────────────────────────────
# chat_input 必须每次都渲染，不能被短路
_prefill = st.session_state.pop("prefill_input", "") or ""
_typed = st.chat_input("请输入你的问题...")
prompt = _prefill or _typed

if prompt:
    st.session_state.agent_running = True
    st.session_state.stop_event.clear()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""
        all_sources = []
        was_stopped = False
        steps = []  # 记录思考过程，rerun 后可重建

        with st.status("Agent 思考中...", expanded=True) as status:
            stop_placeholder = st.empty()
            with stop_placeholder:
                if st.button("⏹ 停止", key="stop_btn"):
                    st.session_state.stop_event.set()
            stream_ph = [None]  # 用列表包装，避免闭包赋值问题

            for event in run_agent(
                prompt,
                stop_event=st.session_state.stop_event,
                max_tool_rounds=cfg_max_rounds,
                top_k=cfg_top_k,
            ):
                if event["type"] == "tool_call":
                    tool_label = {
                        "search_knowledge_base": "📚 检索知识库",
                        "search_web": "🌐 搜索网络",
                    }.get(event["tool"], event["tool"])
                    line = f"{tool_label}：`{event['query']}`"
                    st.write(line)
                    steps.append({"kind": "tool_call", "text": line})

                elif event["type"] == "tool_result":
                    results = event["result"].get("results", [])
                    line = f"  → 获取到 {len(results)} 条结果"
                    st.write(line)
                    steps.append({"kind": "tool_result", "text": line})
                    if event["tool"] == "search_knowledge_base":
                        all_sources.extend(results)

                elif event["type"] == "answer_chunk":
                    if stream_ph[0] is None:
                        stream_ph[0] = st.empty()
                    final_answer += event["content"]
                    stream_ph[0].markdown(final_answer + "▌")

                elif event["type"] == "answer":
                    stop_placeholder.empty()
                    status.update(label="完成", state="complete", expanded=False)
                    gist_increment("queries")
                    if _stats_cache["data"] is not None:
                        _stats_cache["data"]["queries"] = _stats_cache["data"].get("queries", 0) + 1

                elif event["type"] == "retry":
                    reason = event.get("reason_label", "调用失败")
                    line = f"⚠️ {event['from']} {reason}，切换到 {event['to']} 重试..."
                    st.warning(line)
                    steps.append({"kind": "retry", "text": line})

                elif event["type"] == "stopped":
                    was_stopped = True
                    stop_placeholder.empty()
                    status.update(label="已停止", state="error", expanded=False)

                elif event["type"] == "error":
                    trace_id = event.get("trace_id", "")
                    msg = event["content"]
                    if trace_id:
                        msg += f"\n\n`trace_id: {trace_id}`"
                    st.error(msg)
                    stop_placeholder.empty()
                    status.update(label="出错", state="error", expanded=True)

        if final_answer:
            st.markdown(final_answer)
            _render_sources(all_sources, query=prompt)
        elif was_stopped:
            st.markdown("_已停止_")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer or ("_已停止_" if was_stopped else ""),
            "sources": all_sources,
            "steps": steps,
        })

    st.session_state.agent_running = False
    st.session_state.stop_event.clear()

# ── 侧边栏 ────────────────────────────────────────────────

with st.sidebar:
    # 留言板（每次直接从 Gist 拉取，不缓存）
    with st.expander("💬 留言板", expanded=False):
        _fb_input = st.text_area("写下你的建议或反馈", key=f"fb_input_{st.session_state.fb_input_key}", height=80)
        if st.button("提交留言", use_container_width=True, disabled=not _fb_input):
            import time  # noqa: E402
            gist_add_feedback(_fb_input.strip())
            time.sleep(0.8)  # 等异步写入 Gist 完成
            st.toast("感谢你的反馈！")
            st.session_state.fb_input_key += 1  # key 变化，强制 text_area 重建
            st.rerun()
        _fb_data = gist_load()
        _feedbacks = (_fb_data or {}).get("feedback", [])
        if _feedbacks:
            st.caption("最近留言：")
            for _fb in reversed(_feedbacks[-3:]):
                st.caption(f"🗨️ {_fb['time']}　{_fb['content'][:40]}")

    st.divider()
    st.header("关于本项目")
    st.markdown("""
**AI Search Agent** 是一个工程化的 RAG + Web Search 智能问答系统。

不同于把搜索路径写死的 Workflow，这里的核心是一个真正的 **Agent Loop**：每轮由 LLM 自主决定是查本地知识库还是联网搜索，工具调用结果再反馈给 LLM 决定下一步，循环直到给出答案。

**工程亮点：**
- 🛡️ **多模型自动兜底**：配置主模型 + 备用模型列表，主模型额度耗尽或不可用时无感切换，不中断对话
- 🧠 **知识自动内化**：每次联网搜索后，结果异步经 LLM 提炼写入本地知识库，下次同类问题优先命中本地，越用越快
- 🔍 **请求全链路追踪**：每次对话生成唯一 trace_id，出错时直接展示给用户，对照后台日志可定位到具体哪一轮工具调用出了问题
- 📊 **数据驱动评测**：离线跑 RAGAS 评估检索质量（Faithfulness / Context Recall），有数字才能判断改动是否真的有效
    """)

    # 历史输入记录（点击直接重发，最多显示最近 10 条）
    _history = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if _history:
        st.divider()
        st.subheader("历史输入")
        for i, q in enumerate(reversed(_history[:10])):
            if st.button(
                q[:40] + ("..." if len(q) > 40 else ""),
                key=f"hist_{i}",
                use_container_width=True,
                disabled=st.session_state.agent_running,
            ):
                st.session_state.prefill_input = q
                st.rerun()

    # 最近内化状态
    import json as _json  # noqa: E402
    _status_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".internalize_status.json")
    if os.path.exists(_status_file):
        try:
            with open(_status_file, "r", encoding="utf-8") as _f:
                _entries = _json.load(_f)
            if _entries:
                st.divider()
                st.caption("最近内化：")
                for _e in _entries:
                    st.caption(f"  {_e['time']} · 📄 {_e['file']}")
        except Exception:
            pass

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 重建知识库", use_container_width=True):
        try:
            col = get_collection()
            if col.count() > 0:
                col.delete(where={"source": {"$ne": ""}})
        except Exception:
            pass
        st.cache_resource.clear()
        st.rerun()
