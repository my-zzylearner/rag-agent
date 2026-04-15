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

**实现文件**：`app.py`

页面入口增加密码拦截，适用于限定范围分享场景。密码配置方式见 `.env.example`。

## 待解决问题

- web_cache 条目积累后的清理机制 — 目前无过期策略，长期使用后知识库会持续增长
- 内化内容质量控制 — 质量过滤（< 80 字/拒绝语跳过）尚未实现，Tavily 返回的网页摘要质量参差不齐

## 已完成功能

| # | 功能 | 说明 |
|---|------|------|
| 1 | 停止控制 | threading.Event 信号机制，Agent 运行中可随时中止，已有中间结果保留展示 |
| 2 | 联网搜索路由修复 | SYSTEM_PROMPT 增加明确路由规则，实时信息直接走 search_web |
| 3 | 知识自动内化（web_cache） | 搜索结果自动写入 ChromaDB，upsert 去重 |
| 4 | 历史输入重发 | 侧边栏点击历史 query 直接重发 |
| 5 | 访问密码校验 | 密码存 st.secrets/APP_PASSWORD，未配置不拦截 |
| 6 | 知识内化升级（LLM 提炼） | 搜索结果异步经 LLM 提炼后增量写入 data/docs/，frontmatter 动态路由 |
| 7 | 运行时参数配置 | 侧边栏滑块调整 max_tool_rounds / top_k，无需重启 |
| 8 | 工具轮次上限优化 | 达到上限时强制生成最终回答，不再报错 |
| 9 | ChromaDB 路径绝对化 | 基于 `__file__` 绝对路径 + lru_cache 进程级单例 |
| 10 | 来源展示统一 | `_render_sources()` 统一渲染，历史消息和当前回答共用 |
| 11 | 移动端适配 | layout="centered" |
| 12 | 知识内化状态反馈 | 侧边栏展示最近 5 条内化记录 |
| 13 | 相关度阈值过滤 | 过滤 < 0.3 的来源，不展示低相关度结果 |
| 14 | 回答流式输出 | stream=True，answer_chunk 逐 token yield，▌光标指示生成中 |
| 15 | 搜索关键词高亮 | snippet 中匹配 query 词加粗，支持英文缩写单独提取 |
| 16 | 语义 chunk 切分 | 按段落合并（MAX_CHUNK_SIZE=800），超长段落回退句子切分 |
| 17 | web_cache 上限清理 | 超 200 条时按 added_at 删最旧的 |
| 18 | 冷启动进度展示 | 两阶段 status 展示：Embedding 模型加载 → 知识库检查 |
| 19 | 多模型 fallback | LLM_FALLBACK 环境变量，额度不足/模型不存在时自动切换，UI 展示切换原因 |
| 20 | 统一日志系统 | utils/logger.py，关键路径 ERROR 日志，DEBUG=true 输出详细日志 |
| 21 | .env 热重载 | 每次 rerun 重新加载 .env，注释掉的变量实时失效 |
| 22 | 思考过程持久化 | steps 字段存入 session_state，rerun 后历史消息用 expander 重建展示 |
| 23 | 停止按钮自动隐藏 | 完成/停止/出错后 stop_placeholder.empty() 隐藏按钮 |

## TODO（待实现）

| # | 功能 | 优先级 | 状态 | 说明 |
|---|------|--------|------|------|
| 1 | 内化质量过滤 | 🔴 高 | ✅ 已完成 | 拒绝语检测；重复度检测；可选 LLM 打分（长度不作为质量指标） |
| 2 | web_cache 老化清理 | 🔴 高 | ✅ 已完成 | TTL=7天，add_chunks 时删除过期条目；数量上限 200 保留 |
| 3 | 内化匿名化 | 🔴 高 | ✅ 已完成 | 侧边栏只展示时间和文件名，不暴露用户原始 query |
| 4 | 知识库评估（RAGAS） | 🟡 中 | ✅ 已完成 | eval/evaluate.py 离线评估，自动生成问题，输出 eval/report.md |
| 5 | 混合检索 | 🟡 中 | ✅ 已完成 | BM25 + 向量 + RRF 融合（rag/retriever.py），rank_bm25 库，fetch_k=top_k*2 候选 |
| 6 | ChromaDB 持久化（方案B） | 🟡 中 | ✅ 已完成 | 配置 QDRANT_URL+QDRANT_API_KEY 自动切换 Qdrant Cloud，未配置降级本地 ChromaDB；indexer 抽象统一接口 |
| 15 | 连续追问（多轮对话记忆） | 🟡 中 | ✅ 已完成 | run_agent 新增 history 参数，固定窗口 6 轮（12条），app.py 传入 session_state.messages |
| 7 | Reranking | 🟢 低 | ⏳ 待实现 | Cross-Encoder 重排召回结果，适合知识库规模较大后再加 |
| 16 | 思维过程展示（Thinking） | 🟡 中 | ⏳ 待评估 | 展示 LLM 内部推理链路，见设计文档 |
| 8 | 用户隔离 | 🟢 低 | ⏳ 待实现 | 内化内容目前全局共享，考虑按 session 隔离或管理员审核机制 |
| 9 | UI 美化 | 🟢 低 | ⏳ 待实现 | 页面视觉优化：配色、字体、间距、卡片样式；设计专属图标 |
| 10 | 工具结构化错误 | 🔴 高 | ⏳ 待实现 | search_knowledge_base 空结果时给出修正建议（如"建议调用 search_web 继续检索"），减少 Agent 直接回答"未找到"的情况 |
| 11 | 确定性验收基线 | 🔴 高 | ✅ 已完成 | tests/test_acceptance.py，6 个场景：天气路由、RAG路由、KB降级、停止信号，全 mock 离线运行 |
| 12 | 上下文分层管理 | 🟡 中 | ⏳ 待实现 | system prompt 拆分为常驻层（身份/约束）+ 按需加载层（路由规则/领域知识），防止 Context Rot |
| 13 | 多轮对话记忆压缩 | 🟡 中 | ⏳ 待实现 | messages[] 超 token 阈值时自动摘要压缩，防止长对话后决策质量下降 |
| 14 | 评测环境隔离 | 🟡 中 | ⏳ 待实现 | eval/evaluate.py 使用独立 ChromaDB 实例，不复用生产知识库，保证评测结果稳定可重现 |

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
- 2026-04-12: 完成 TODO 7-11：回答流式输出、关键词高亮、语义 chunk 切分、web_cache 上限清理、冷启动优化
- 2026-04-12: 禁用 ChromaDB 遥测（ANONYMIZED_TELEMETRY/CHROMA_TELEMETRY=False），修复 opentelemetry protobuf 冲突（PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python）
- 2026-04-12: 升级 sentence-transformers>=3.0.0、protobuf>=3.20.0,<4.0.0，兼容 Streamlit Cloud Python 3.14 环境
- 2026-04-12: 新增多模型 fallback 机制（LLM_FALLBACK 环境变量），额度不足/模型不存在/限流时自动切换，UI 展示切换原因
- 2026-04-12: 新增统一日志模块 agent/logger.py，每次请求生成 trace_id，ERROR 默认输出，DEBUG=true 时输出详细日志；error 事件携带 trace_id 展示给用户
- 2026-04-12: fallback 触发条件扩展为 _should_fallback()，覆盖 model_not_found/404/无权限等场景，提示文案动态生成
- 2026-04-12: status 框思考过程持久化到 session_state（steps 字段），rerun 后历史消息用 expander 重建展示
- 2026-04-12: agent 运行中 slider 改为静态文字，防止拖动触发 rerun 中断 agent
- 2026-04-12: web_cache TTL 7天老化清理、内化展示匿名化（只显示文件名）；内化质量过滤未实现
- 2026-04-12: 停止按钮在完成/停止/出错后用 stop_placeholder.empty() 隐藏
- 2026-04-12: system prompt 增加禁止编造 URL/链接约束，防止模型幻觉生成假参考资料
- 2026-04-12: 主页面副标题 ERNIE 改为 Qwen，与当前 LLM 配置一致
- 2026-04-12: Streamlit Cloud 改用 Python 3.10，移除 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python（该 workaround 仅针对 3.14 C 扩展兼容问题，3.10 不需要）
- 2026-04-12: 新增访问统计 + 留言板（GitHub Gist 持久化），侧边栏顶部展示访问人数/查询次数，留言板收集优化建议；新增 utils/gist_store.py 封装 Gist 读写
- 2026-04-12: 修复 .env 热重载在 Streamlit Cloud 上误清 secrets 的问题，改为检测 .env 文件存在时才 pop
- 2026-04-12: _build_candidates() 加 lru_cache 进程级缓存，避免每次请求重建 OpenAI client
- 2026-04-12: 新增 eval/evaluate.py 离线评估脚本，自动从文档生成问答对，RAGAS 评估 Faithfulness/Answer Relevancy/Context Recall，输出 eval/report.md
- 2026-04-12: 首次评估结果：Faithfulness=0.948（良好），Context Recall=0.250（偏低），Answer Relevancy=nan（embedding调用失败待修复）；Context Recall 低的根本原因是向量检索召回率不足，混合检索（BM25+向量）可改善
- 2026-04-14: 内化质量过滤补充长度检查后撤回（长度不是质量指标，会激励口水话）；新增 tests/test_acceptance.py 验收基线（6 场景，全 mock 离线）
- 2026-04-15: 混合检索（BM25+向量+RRF）上线；连续追问支持（history 固定窗口 6 轮）；方案E确认：data/docs/ 已在 git，内化文档持久
- 2026-04-15: Qdrant Cloud 接入（方案B）；indexer 抽象 _count/_get_all/_upsert/_delete_by_ids/_delete_by_filter/_query 六个统一接口，上层无感知后端切换；_use_qdrant() 带连通性探测（5s超时），失败自动降级 ChromaDB；app.py 移除直接依赖 get_collection()
