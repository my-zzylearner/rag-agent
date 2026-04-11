"""
Streamlit 前端入口
"""
import os
import threading
from dotenv import load_dotenv

# 禁用 chromadb 遥测，避免 opentelemetry 依赖问题
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
import streamlit as st

load_dotenv()

from rag.indexer import index_documents, is_indexed  # noqa: E402
from agent.agent import run_agent  # noqa: E402

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Search Agent",
    page_icon="🔍",
    layout="wide",
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
st.caption("RAG + Web Search · Powered by ERNIE & Tavily")

# ── 初始化知识库（只在第一次运行时索引）────────────────────
@st.cache_resource(show_spinner="正在加载知识库...")
def init_index():
    if not is_indexed():
        count = index_documents("./data/docs")
        return count
    return None

chunk_count = init_index()
if chunk_count is not None:
    st.toast(f"本地文档已索引，写入 {chunk_count} 个文本块", icon="✅")

# ── 状态初始化 ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "prefill_input" not in st.session_state:
    st.session_state.prefill_input = ""
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

# 每次 rerun 都清除停止信号，避免残留
# 注意：agent_running 不在这里重置，由 Agent 执行块自己管理
st.session_state.stop_event.clear()

# ── 参数配置 + 知识库统计（侧边栏读取，需在 run_agent 调用前定义）──
_slider_disabled = st.session_state.agent_running
cfg_max_rounds = st.sidebar.slider("最大工具调用轮次", min_value=1, max_value=6, value=3, step=1, disabled=_slider_disabled)
cfg_top_k = st.sidebar.slider("知识库检索条数", min_value=1, max_value=10, value=4, step=1, disabled=_slider_disabled)

# 知识库统计（每次 rerun 刷新）
from rag.indexer import get_collection as _get_col  # noqa: E402
_col = _get_col()
_total = _col.count()
try:
    _web = len(_col.get(where={"type": {"$eq": "web_cache"}}, include=[])["ids"])
except Exception:
    _web = 0
_local = _total - _web
st.sidebar.caption(f"知识库：{_local} 本地 + {_web} 网络缓存 = {_total} 条")

# 展示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源", expanded=False):
                for src in msg["sources"]:
                    score = src.get('relevance_score', '')
                    snippet = src['content'].replace('\n', ' ').strip()[:120]
                    st.caption(f"📄 {src['source']}　相关度 {score}")
                    st.text(snippet + "…")

# ── 用户输入 ──────────────────────────────────────────────
# chat_input 必须每次都渲染，不能被短路
_prefill = st.session_state.pop("prefill_input", "") or ""
_typed = st.chat_input("请输入你的问题...")
prompt = _prefill or _typed

if prompt:
    st.session_state.stop_event.clear()
    st.session_state.agent_running = True
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        final_answer = ""
        all_sources = []
        was_stopped = False

        with st.status("Agent 思考中...", expanded=True) as status:
            stop_col, _ = st.columns([1, 4])
            with stop_col:
                if st.button("⏹ 停止", key="stop_btn"):
                    st.session_state.stop_event.set()

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
                    st.write(f"{tool_label}：`{event['query']}`")

                elif event["type"] == "tool_result":
                    results = event["result"].get("results", [])
                    st.write(f"  → 获取到 {len(results)} 条结果")
                    if event["tool"] == "search_knowledge_base":
                        all_sources.extend(results)

                elif event["type"] == "answer":
                    final_answer = event["content"]
                    status.update(label="完成", state="complete", expanded=False)

                elif event["type"] == "stopped":
                    was_stopped = True
                    status.update(label="已停止", state="error", expanded=False)

                elif event["type"] == "error":
                    st.error(event["content"])
                    status.update(label="出错", state="error", expanded=True)

        if final_answer:
            st.markdown(final_answer)
            if all_sources:
                with st.expander("📎 参考来源", expanded=False):
                    for src in all_sources:
                        st.markdown(f"**{src['source']}** (相关度: {src.get('relevance_score', 'N/A')})")
                        st.markdown(f"> {src['content'][:200]}...")
        elif was_stopped:
            st.markdown("_已停止_")

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_answer or ("_已停止_" if was_stopped else ""),
            "sources": all_sources,
        })

    st.session_state.agent_running = False

    st.session_state.stop_event.clear()

# ── 侧边栏 ────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.header("关于本项目")
    st.markdown("""
**AI Search Agent** 是一个结合 RAG 和 Web Search 的智能问答系统。

**技术架构：**
- 🧠 LLM: ERNIE-Speed (文心一言)
- 📦 向量库: ChromaDB
- 🔗 Embedding: all-MiniLM-L6-v2
- 🌐 网络搜索: Tavily
- 🖥️ 前端: Streamlit

**Agent 工作流：**
1. 分析用户问题
2. 选择工具（知识库 or 网络）
3. 执行检索
4. 融合结果生成回答

**知识库内容：**
- RAG 技术原理
- 向量数据库对比
- 搜索排序算法
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

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 重建知识库", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
