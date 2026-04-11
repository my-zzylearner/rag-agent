# Agent 优化方案

## 概述

对 rag-agent 的六项功能优化：停止控制、联网搜索路由修复、知识自动内化（升级为 LLM 提炼写入文档）、历史输入重发、访问密码校验、运行时参数配置。

## 决策记录

- [x] 已确定：停止按钮使用 `threading.Event` 信号机制 — Streamlit rerun 模型下的最简方案，无需引入异步
- [x] 已确定：联网搜索路由通过 Prompt 工程修复 — 在 SYSTEM_PROMPT 和工具 description 中明确路由规则，成本最低
- [x] 已确定：知识内化升级为 LLM 提炼 + 文档增量写入 — 原始 web_cache 只是碎片，现在异步提炼为结构化 markdown 写入 data/docs/，并重新索引
- [x] 已确定：实时类判断采用硬规则优先 + LLM 兜底 — 天气/股价等关键词命中直接跳过，未命中才调用 LLM 判断，节省 token
- [x] 已确定：历史输入通过侧边栏点击重发 — `st.chat_input` 在独立 iframe 中，JS 跨 iframe 注入被浏览器同源策略阻断，键盘方向键方案不可行
- [x] 已确定：访问密码使用简单密码拦截 — 场景是分享给认识的人，无需 OAuth，密码存 `st.secrets`/环境变量 `APP_PASSWORD`，本地不配置则不拦截
- [x] 已确定：运行时参数配置通过侧边栏滑块 — max_tool_rounds 和 top_k 无需重启即可调整
- [ ] 待定：内化内容的老化清理策略 — 知识库会随使用积累 web_cache 和文档补充内容，暂无过期机制

## 方案细节

### 1. 停止按钮

**实现文件**：`app.py`、`agent/agent.py`

**核心机制**：
- `st.session_state.stop_event`：`threading.Event` 对象，跨 rerun 持久
- `st.session_state.agent_running`：布尔标志，Agent 运行时禁用输入框
- `run_agent(user_query, stop_event)` 每轮循环开头检查 `stop_event.is_set()`
- 用户点击停止按钮 → Streamlit rerun → stop_event 已 set → generator 下一轮检查到后 yield `{"type": "stopped"}` 并退出

**边界情况**：
- 停止后已获取的中间结果（工具调用步骤）仍正常展示
- 停止后输入框重新可用（`agent_running` 重置为 False）

### 2. 联网搜索路由修复

**实现文件**：`agent/agent.py`（SYSTEM_PROMPT）、`agent/tools.py`（工具 description）

**根本原因**：原 SYSTEM_PROMPT 只说"知识库没有时再用网络"，但未定义"没有"的判断标准，也未区分实时信息类问题。LLM 在知识库返回低相关度结果时仍可能直接回答。

**修复方式**：
- SYSTEM_PROMPT 增加明确路由规则：实时信息类直接走 `search_web`
- 知识库返回空/低相关时必须继续调用 `search_web`，不能直接回答"未找到"
- 工具 description 中明确各自的适用和不适用场景

### 3. 知识自动内化

**实现文件**：`rag/indexer.py`（新增 `add_chunks()`）、`agent/tools.py`（`search_web` 调用后触发）

**数据流**：
```
search_web 成功
    ↓
results 列表（content + source URL）
    ↓
add_chunks() — 每条结果作为一个 chunk
    ↓
hash(source + content[:64]) → chunk id
    ↓
ChromaDB upsert（去重，已存在则更新）
    ↓
metadata: {"source": url, "type": "web_cache"}
```

**去重策略**：用 `md5(source:content[:64])` 作为 chunk id，`upsert` 保证同一 URL 同一内容只存一份。

**失败处理**：内化操作包在 `try/except` 中，失败静默忽略，不影响搜索结果返回给用户。

### 4. 历史输入重发

**实现文件**：`app.py`

**方案**：侧边栏展示用户历史输入列表（倒序），点击后写入 `st.session_state.prefill_input`，触发 rerun，主逻辑读取后直接作为 prompt 执行。

**放弃的方案**：键盘上下键 + JS 注入。`st.chat_input` 渲染在独立 iframe 中，`st.components.v1.html` 注入的 JS 也在另一个 iframe，浏览器同源策略阻断了跨 iframe 的 `contentDocument` 访问。

### 5. 访问密码校验

**实现文件**：`app.py`（`_check_password()` 函数）

**机制**：
- 页面最顶部调用 `_check_password()`，未通过则 `st.stop()` 阻断后续渲染
- 密码来源优先级：`st.secrets["APP_PASSWORD"]` > 环境变量 `APP_PASSWORD`
- 未配置密码时直接放行（本地开发不受影响）
- 通过后写入 `st.session_state.authenticated = True`，rerun 后不再拦截

**局限**：无防暴力破解，但对"分享给认识的人"场景足够。

## 待解决问题

- web_cache 条目积累后的清理机制 — 目前无过期策略，长期使用后知识库会持续增长
- 内化内容质量控制 — Tavily 返回的网页摘要质量参差不齐，暂未过滤

## TODO（待实现）

| # | 功能 | 状态 | 说明 |
|---|------|------|------|
| 1 | 来源展示统一 | ✅ 已完成 | 提取 `_render_sources()` 统一渲染，历史消息和当前回答共用 |
| 2 | 移动端适配 | ✅ 已完成 | `layout="wide"` → `layout="centered"` |
| 3 | 知识内化状态反馈 | ✅ 已完成 | 内化完成写 `.internalize_status.json`，侧边栏展示最近 5 条 |
| 4 | `.env` 注释格式修复 | ✅ 已完成 | `;` 改为 `#`，消除 python-dotenv 解析警告 |
| 5 | 相关度阈值过滤 | ✅ 已完成 | `_render_sources()` 内过滤 < 0.3 的结果，不展示低相关度来源 |
| 6 | ChromaDB 持久化 | ⏳ 待实现 | Streamlit Cloud 每次冷启动需重建知识库，可考虑挂载外部存储或用云端向量库 |

### 6. 知识内化升级（LLM 提炼 + 文档增量写入）

**实现文件**：`rag/knowledge_internalizer.py`（新建）、`rag/indexer.py`（新增 `index_single_document`）、`agent/tools.py`（触发异步线程）

**数据流**：
```
search_web 成功
    ↓
daemon 线程启动 internalize_async(query, results, client, model)
    ↓
Step 1: 分类判断
  硬规则关键词（天气/股价等）命中 → 跳过
  未命中 → LLM 判断是否实时类
    ↓
Step 2: LLM 提炼（temperature=0）
  输入：query + 搜索结果原文
  输出：结构化 markdown 知识条目（≤500字，去噪，含来源）
    ↓
Step 3: 路由到目标文档
  TOPIC_MAP 关键词评分 → 匹配已有文档
  无匹配 → LLM 生成文件名，新建 data/docs/<topic>.md
    ↓
Step 4: 增量写入（query md5 去重，避免重复追加）
    ↓
Step 5: index_single_document(filepath) 重索引该文件
```

**关键设计**：
- 异步：daemon 线程，不阻塞 Agent 回答
- 原有 web_cache 写入保留（快速检索用），内化是额外的质量提升
- 所有异常静默忽略，不影响主流程

### 7. 运行时参数配置

**实现文件**：`app.py`

侧边栏顶部两个滑块：`max_tool_rounds`（1-6，默认3）和 `top_k`（1-10，默认4），每次对话读取当前值传给 `run_agent`，无需重启生效。

### 8. 知识内化路由升级（frontmatter 动态路由）

**实现文件**：`data/docs/*.md`（加 frontmatter）、`rag/knowledge_internalizer.py`（路由改造）、`rag/indexer.py`（递归扫描 + frontmatter 解析）

**核心设计**：
- 每个文档头部加 YAML frontmatter，包含 `topic`/`keywords`/`description`/`type` 字段
- `_route()` 动态扫描 `data/docs/**/*.md`，读取各文件 `description`，构造候选列表交给 LLM 判断
- LLM 返回文件名 → 追加到该文件；返回 "new" → 调用 `_new_file()` 新建带 frontmatter 的文件
- `load_documents` 改为递归扫描（支持子目录），解析 frontmatter 后只索引正文，metadata 带 `topic`/`type`

**优势**：
- 新增文档自动纳入路由，无需修改代码
- 原始文档和补充内容在同一文件，主题知识完整
- frontmatter 作为文档的"技能描述"，路由判断更准确

### 9. 工具调用轮次上限行为优化

**实现文件**：`agent/agent.py`

**改动前**：达到 `max_tool_rounds` 时 yield `type: error`，直接报错，已收集的工具结果被丢弃。

**改动后**：达到上限时，去掉 `tools` 参数再调用一次 LLM，强制基于已有工具结果生成最终回答，不再报错。

```python
# 达到最大轮次，禁用工具强制 LLM 基于已有结果生成最终回答
final_resp = client.chat.completions.create(model=model, messages=messages)
yield {"type": "answer", "content": final_resp.choices[0].message.content}
```

**Why**：之前的报错会让用户看到"请重新提问"，但实际上 LLM 已经拿到了足够的检索结果，强制生成一次回答比报错更有价值。

### 10. ChromaDB 路径绝对化

**实现文件**：`rag/indexer.py`

**问题**：`CHROMA_PATH = "./chroma_db"` 相对路径在不同调用方（app.py、retriever.py）工作目录不同时，ChromaDB 认为是两个不同实例，报 `ValueError: An instance of Chroma already exists with different settings`。

**修复**：改为基于 `__file__` 的绝对路径，同时用 `@functools.lru_cache(maxsize=1)` 保证 `get_collection()` 进程级单例。

## 变更记录

- 2026-04-11: 完成三项优化实现（停止控制、联网路由、知识内化 web_cache）
- 2026-04-11: 新增历史输入重发（侧边栏点击）、访问密码校验
- 2026-04-11: 知识内化升级为 LLM 提炼 + 文档增量写入（异步）；新增运行时参数配置滑块
- 2026-04-11: 知识内化路由升级为 frontmatter 动态路由；indexer 支持递归扫描和 frontmatter 解析
- 2026-04-11: 工具调用轮次上限由报错改为强制生成最终回答；修复 ChromaDB 相对路径冲突
- 2026-04-12: 完成 TODO 1-5：来源展示统一、移动端适配、知识内化状态反馈、.env 注释格式、相关度阈值过滤
